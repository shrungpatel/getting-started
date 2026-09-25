"""Estimate causal-pathway stability with clean/corrupted activation patching."""

import gc
import inspect
import logging
import math
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)
MODEL_ALIASES = {"qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}
MAX_PROMPT_TOKENS = 2048
MAX_NEW_TOKENS = 128
MAX_ANSWER_TOKENS = 32
NUM_CAUSAL_LAYERS = 4
# Uncalibrated heuristics: tune on public development data before submission.
DISTANCE_THRESHOLD = 0.25
MIN_CAUSAL_EFFECT = 1e-4
MIN_ANSWER_PROBABILITY = 1e-4
MIN_ANSWER_PROBABILITY_RATIO = 0.5
# Set an explicit named_modules() path only for architectures with ambiguous stacks.
BLOCK_STACK_PATH: str | None = None
PERTURBATION_PREFIXES = (
    "Solve the following mathematical problem:\n\n",
    "Determine the answer to the following mathematical problem:\n\n",
)
ANSWER_INSTRUCTION = "\n\nGive only the final answer, without an explanation."
READOUT_PREFIX = "\nFinal answer:"
_loaded_model_id: str | None = None


def make_variants(problem: str) -> list[str]:
    """Keep all mathematical content unchanged and vary only the instruction."""
    return [problem, *(prefix + problem for prefix in PERTURBATION_PREFIXES)]


def _load_model(model_id: str) -> tuple[Any, Any]:
    global _loaded_model_id
    if model_id != _loaded_model_id:
        # Evict before loading: retaining two checkpoints can exhaust GPU memory.
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
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=True,
        dtype=dtype,
        device_map="auto" if device == "cuda" else "cpu",
    )
    if getattr(model.config, "is_encoder_decoder", False):
        raise ValueError("Causal stability requires a decoder-only causal language model")
    model.eval()
    return tokenizer, model


def _text_config(model: Any) -> Any:
    getter = getattr(model.config, "get_text_config", None)
    return getter() if callable(getter) else model.config


def _context_limit(tokenizer: Any, model: Any) -> int:
    limits = [MAX_PROMPT_TOKENS + MAX_NEW_TOKENS]
    for obj, names in (
        (tokenizer, ("model_max_length",)),
        (_text_config(model), ("max_position_embeddings", "n_positions", "max_seq_len")),
    ):
        for name in names:
            value = getattr(obj, name, None)
            if isinstance(value, int) and value > 0:
                limits.append(value)
    return min(limits)


def _discover_blocks(model: Any) -> list[tuple[str, torch.nn.Module]]:
    """Discover a repeated decoder stack structurally, not by model-family name."""
    if BLOCK_STACK_PATH is not None:
        stack = model.get_submodule(BLOCK_STACK_PATH)
        if not isinstance(stack, (torch.nn.ModuleList, torch.nn.Sequential)) or not len(stack):
            raise ValueError("BLOCK_STACK_PATH must identify a nonempty decoder block stack")
        candidates = [(BLOCK_STACK_PATH, stack)]
    else:
        candidates = []
        config = _text_config(model)
        depth = next(
            (getattr(config, key) for key in ("num_hidden_layers", "n_layer", "num_layers", "n_layers")
             if isinstance(getattr(config, key, None), int) and getattr(config, key) > 0),
            None,
        )
        for name, module in model.named_modules():
            if not isinstance(module, (torch.nn.ModuleList, torch.nn.Sequential)) or not len(module):
                continue
            children = list(module.children())
            if depth is not None and len(children) != depth:
                continue
            if depth is None and len({type(child) for child in children}) != 1:
                continue
            if not all(any(child.children()) for child in children):
                continue
            candidates.append((name, module))
        if len(candidates) != 1:
            names = [name for name, _ in candidates]
            raise ValueError(
                f"Cannot uniquely identify the decoder block stack (candidates: {names}). "
                "Set BLOCK_STACK_PATH to its model.named_modules() path."
            )
    name, stack = candidates[0]
    # The terminal block can merely replace the first answer-token readout.
    # Prefer internal sites, retaining a fallback for single-block toy models.
    available = max(1, len(stack) - 1)
    count = min(NUM_CAUSAL_LAYERS, available)
    indices = [round(i * (available - 1) / max(1, count - 1)) for i in range(count)]
    return [(f"{name}.{index}", stack[index]) for index in indices]


def _prompt_ids(tokenizer: Any, text: str) -> list[int]:
    content = text + ANSWER_INSTRUCTION
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": content}]
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        # Some templates always open a thinking section, ignoring the option above.
        if prompt.rstrip().endswith("<think>"):
            prompt += "</think>\n"
        add_special_tokens = False
    else:
        prompt = content
        add_special_tokens = True
    return tokenizer.encode(prompt + READOUT_PREFIX, add_special_tokens=add_special_tokens)


def _make_pair(tokenizer: Any, text: str, empty_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace the variable prompt span with neutral tokens, preserving positions."""
    ids = _prompt_ids(tokenizer, text)
    start = 0
    while start < min(len(ids), len(empty_ids)) and ids[start] == empty_ids[start]:
        start += 1
    suffix = 0
    while (
        suffix < min(len(ids) - start, len(empty_ids) - start)
        and ids[len(ids) - suffix - 1] == empty_ids[len(empty_ids) - suffix - 1]
    ):
        suffix += 1
    stop = len(ids) - suffix
    neutral_ids = tokenizer.encode("?", add_special_tokens=False)
    if not neutral_ids or start >= stop or stop >= len(ids):
        raise ValueError("Tokenizer did not preserve an identifiable problem span and readout suffix")
    corrupted = ids.copy()
    corrupted[start:stop] = [neutral_ids[0]] * (stop - start)
    return torch.tensor([ids], dtype=torch.long), torch.tensor([corrupted], dtype=torch.long)


def _input_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


def _extract_answer(text: str, finished: bool) -> str | None:
    """Prefer a balanced boxed answer; reject ambiguous or unfinished reasoning."""
    if "<think>" in text and "</think>" not in text:
        return None
    text = text.rsplit("</think>", 1)[-1].strip()
    start = text.rfind("\\boxed{")
    if start >= 0:
        depth = 1
        for index in range(start + len("\\boxed{"), len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index + 1] if index > start + len("\\boxed{") else None
        return None
    if not finished:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        return None
    answer = lines[0]
    if answer.lower().startswith("final answer:"):
        answer = answer[len("final answer:"):].strip()
    return answer or None


def _reference_answer(tokenizer: Any, model: Any, prompt: torch.Tensor, budget: int) -> torch.Tensor | None:
    inputs = prompt.to(_input_device(model))
    eos_ids = getattr(model.generation_config, "eos_token_id", None)
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [])
    pad_id = tokenizer.pad_token_id
    if pad_id is None and eos_ids:
        pad_id = eos_ids[0]
    with torch.inference_mode():
        generated = model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=budget,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            use_cache=True,
            pad_token_id=pad_id,
            return_dict_in_generate=False,
            output_scores=False,
            output_logits=False,
        )[0, prompt.shape[1]:].tolist()
    finished = any(token in eos_ids for token in generated)
    if finished:
        generated = generated[:next(i for i, token in enumerate(generated) if token in eos_ids)]
    answer = _extract_answer(tokenizer.decode(generated, skip_special_tokens=True), finished)
    if answer is None:
        return None
    # Use the exact same target tokens in every clean/corrupted/intervened context.
    ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not 0 < len(ids) <= MAX_ANSWER_TOKENS:
        return None
    return torch.tensor([ids], dtype=torch.long)


def _hidden(output: Any) -> torch.Tensor:
    hidden = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError("Decoder blocks must expose batch-first [1, sequence, hidden] outputs")
    return hidden


@contextmanager
def _activation_hooks(
    blocks: list[tuple[str, torch.nn.Module]],
    position: int,
    captured: dict[str, torch.Tensor],
    patch: tuple[str, torch.Tensor] | None,
) -> Iterator[None]:
    handles = []
    calls: dict[str, int] = {}

    def hook_for(name: str) -> Any:
        def hook(module: Any, args: Any, output: Any) -> Any:
            calls[name] = calls.get(name, 0) + 1
            if calls[name] != 1:
                raise ValueError("Shared/recurrent decoder blocks require a custom intervention adapter")
            hidden = _hidden(output)
            if position >= hidden.shape[1]:
                raise ValueError("Decoder block output does not contain the readout position")
            if patch is None:
                captured[name] = hidden[0, position].detach().cpu().clone()
                return None
            donor = patch[1]
            if donor.shape != hidden[0, position].shape:
                raise ValueError("Donor activation and recipient have different hidden dimensions")
            changed = hidden.clone()
            changed[0, position] = donor.to(device=hidden.device, dtype=hidden.dtype)
            if isinstance(output, tuple):
                return (changed, *output[1:])
            if isinstance(output, list):
                return [changed, *output[1:]]
            return changed
        return hook

    selected = [(name, block) for name, block in blocks if patch is None or name == patch[0]]
    try:
        for name, block in selected:
            handles.append(block.register_forward_hook(hook_for(name)))
        yield
        if len(calls) != len(selected) or not selected:
            raise ValueError("Selected decoder blocks did not execute during the forward pass")
    finally:
        for handle in handles:
            handle.remove()


def _answer_probability(
    model: Any,
    prompt: torch.Tensor,
    answer: torch.Tensor,
    blocks: list[tuple[str, torch.nn.Module]],
    patch: tuple[str, torch.Tensor] | None = None,
) -> tuple[float, dict[str, torch.Tensor]]:
    position = prompt.shape[1] - 1
    inputs = torch.cat((prompt, answer[:, :-1]), dim=1).to(_input_device(model))
    captured: dict[str, torch.Tensor] = {}
    forward_options = {}
    parameters = inspect.signature(model.forward).parameters
    for key in ("logits_to_keep", "num_logits_to_keep"):
        if key in parameters:
            forward_options[key] = answer.shape[1]
            break
    with torch.inference_mode(), _activation_hooks(blocks, position, captured, patch):
        output = model(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            use_cache=False,
            return_dict=True,
            **forward_options,
        )
        logits = output.logits[:, -answer.shape[1]:].float()
        if logits.shape[1] != answer.shape[1]:
            raise ValueError("The language model must return logits for every answer position")
        token_log_probs = torch.log_softmax(logits, dim=-1).gather(
            -1, answer.to(logits.device).unsqueeze(-1)
        )
        log_probability = float(token_log_probs.double().sum().item())
    return math.exp(log_probability), captured


def _causal_profile(
    model: Any,
    pair: tuple[torch.Tensor, torch.Tensor],
    answer: torch.Tensor,
    blocks: list[tuple[str, torch.nn.Module]],
) -> tuple[list[float], float]:
    clean, corrupt = pair
    clean_probability, clean_states = _answer_probability(model, clean, answer, blocks)
    corrupt_probability, corrupt_states = _answer_probability(model, corrupt, answer, blocks)
    effects = []
    for name, _ in blocks:
        damaged_probability, _ = _answer_probability(
            model, clean, answer, blocks, patch=(name, corrupt_states[name])
        )
        restored_probability, _ = _answer_probability(
            model, corrupt, answer, blocks, patch=(name, clean_states[name])
        )
        # Hold the input fixed within each difference; change only one mediator.
        effects.extend((clean_probability - damaged_probability, restored_probability - corrupt_probability))
    return effects, clean_probability


def pathway_distance(original: list[float], variant: list[float]) -> float:
    """Signed relative L1 distance; zero-effect profiles are not evidence of stability."""
    if not original or len(original) != len(variant):
        return math.inf
    if not all(math.isfinite(value) for value in (*original, *variant)):
        return math.inf
    if any(
        sum(abs(value) for value in profile) / len(profile) < MIN_CAUSAL_EFFECT
        or max(profile) < MIN_CAUSAL_EFFECT
        for profile in (original, variant)
    ):
        return math.inf
    magnitude = sum(abs(value) for value in (*original, *variant))
    return sum(abs(left - right) for left, right in zip(original, variant)) / magnitude


def predict_robustness(model_id: str, problems: list[str]) -> list[bool]:
    """Predict stability of measured causal effects, not mathematical correctness."""
    if not problems:
        return []
    if not any(problem.strip() for problem in problems):
        return [False for _ in problems]
    tokenizer, model = _load_model(MODEL_ALIASES.get(model_id, model_id))
    blocks = _discover_blocks(model)
    context_limit = _context_limit(tokenizer, model)
    empty_ids = _prompt_ids(tokenizer, "")
    predictions = []
    for index, problem in enumerate(problems):
        if not problem.strip():
            predictions.append(False)
            continue
        pairs = [_make_pair(tokenizer, text, empty_ids) for text in make_variants(problem)]
        longest_prompt = max(clean.shape[1] for clean, _ in pairs)
        budget = min(MAX_NEW_TOKENS, context_limit - longest_prompt)
        if longest_prompt > MAX_PROMPT_TOKENS or budget < MAX_ANSWER_TOKENS:
            LOGGER.warning("Problem %d exceeds the context budget; predicting False", index)
            predictions.append(False)
            continue
        answer = _reference_answer(tokenizer, model, pairs[0][0], budget)
        if answer is None:
            LOGGER.warning("Problem %d has no usable short reference answer; predicting False", index)
            predictions.append(False)
            continue
        original, probability = _causal_profile(model, pairs[0], answer, blocks)
        if (
            not math.isfinite(probability)
            or probability < MIN_ANSWER_PROBABILITY
            or not math.isfinite(pathway_distance(original, original))
        ):
            predictions.append(False)
            continue
        robust = True
        for pair in pairs[1:]:
            variant, variant_probability = _causal_profile(model, pair, answer, blocks)
            distance = pathway_distance(original, variant)
            LOGGER.debug("Problem %d: causal distance=%g, answer probability=%g", index, distance, variant_probability)
            if (
                not math.isfinite(variant_probability)
                or variant_probability < max(MIN_ANSWER_PROBABILITY, probability * MIN_ANSWER_PROBABILITY_RATIO)
                or not math.isfinite(distance)
                or distance > DISTANCE_THRESHOLD
            ):
                robust = False
                break
        predictions.append(bool(robust))
    return predictions
