"""Predict robustness from prompt-conditioned representation stability."""

from functools import lru_cache
import gc
import logging
import math
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer


LOGGER = logging.getLogger(__name__)
MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
MODEL_ALIASES = {"qwen3-8b:low": MODEL_ID}
# This is an uncalibrated starting point, not a learned decision boundary.
DISTANCE_THRESHOLD = 0.05
MAX_PROMPT_TOKENS = 2048
PERTURBATION_PREFIXES = (
    "Solve the following mathematical problem:\n\n",
    "Find the answer to the following mathematical problem:\n\n",
    "Determine the solution to the following mathematical problem:\n\n",
)


def make_variants(problem: str) -> list[str]:
    """Vary the instruction wording without rewriting the problem itself."""
    if not problem.strip():
        return []
    return [prefix + problem for prefix in PERTURBATION_PREFIXES]


_loaded_model_id: str | None = None


def _load_model(model_id: str) -> tuple[Any, Any]:
    global _loaded_model_id
    if model_id != _loaded_model_id:
        # Evict before loading, not after: two checkpoints may not fit in VRAM.
        _load_model_cached.cache_clear()
        _loaded_model_id = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = _load_model_cached(model_id)
    _loaded_model_id = model_id
    return result


@lru_cache(maxsize=1)
def _load_model_cached(model_id: str) -> tuple[Any, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    # The base model exposes final hidden states without computing vocabulary logits.
    model = AutoModel.from_pretrained(
        model_id,
        dtype=dtype,
        local_files_only=True,
        device_map="auto" if device == "cuda" else "cpu",
    )
    model.eval()
    return tokenizer, model


def _tokenize(tokenizer: Any, text: str) -> dict[str, torch.Tensor]:
    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        add_special_tokens = False
    else:
        add_special_tokens = True
    return tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
        truncation=False,
    )


def _encode(model: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    inputs = {name: value.to(model.device) for name, value in inputs.items()}
    with torch.inference_mode():
        output = model(**inputs, use_cache=False, return_dict=True)
    # Each sequence is processed individually, so the last token is never padding.
    return output.last_hidden_state[0, -1].detach().float().cpu()


def mean_cosine_distance(vectors: list[torch.Tensor]) -> float:
    """Compare the original vector with each variant; invalid vectors score infinity."""
    if len(vectors) < 2:
        return math.inf
    matrix = torch.stack(vectors).float()
    norms = torch.linalg.vector_norm(matrix, dim=1)
    if not bool(torch.isfinite(matrix).all()) or not bool(torch.isfinite(norms).all()):
        return math.inf
    if bool((norms <= 1e-12).any()):
        return math.inf
    normalized = matrix / norms.unsqueeze(1)
    similarities = (normalized[1:] @ normalized[0]).clamp(-1.0, 1.0)
    return float((1.0 - similarities).mean().item())


def predict_robustness(model_id: str, problems: list[str]) -> list[bool]:
    if not problems:
        return []
    resolved_id = MODEL_ALIASES.get(model_id, model_id)
    if not any(problem.strip() for problem in problems):
        return [False for _ in problems]

    tokenizer, model = _load_model(resolved_id)
    text_config = model.config.get_text_config()
    context_limit = min(
        MAX_PROMPT_TOKENS,
        getattr(text_config, "max_position_embeddings", MAX_PROMPT_TOKENS),
    )
    predictions = []
    for index, problem in enumerate(problems):
        variants = make_variants(problem)
        if not variants:
            predictions.append(False)
            continue
        inputs = [_tokenize(tokenizer, text) for text in [problem, *variants]]
        # Truncating a variant can remove mathematics and invalidate the comparison.
        if any(not 0 < item["input_ids"].shape[1] <= context_limit for item in inputs):
            LOGGER.warning("Problem %d exceeds the prompt limit; predicting False", index)
            predictions.append(False)
            continue
        vectors = [_encode(model, item) for item in inputs]
        score = mean_cosine_distance(vectors)
        predictions.append(bool(math.isfinite(score) and score < DISTANCE_THRESHOLD))
    return predictions
