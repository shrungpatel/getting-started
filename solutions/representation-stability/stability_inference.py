"""Trainable robustness prediction from prompt-conditioned stability."""

from __future__ import annotations

from functools import lru_cache
import gc
import logging
from pathlib import Path
import re
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


LOGGER = logging.getLogger(__name__)

# Compatibility aliases accepted by the benchmark. Any exact Hugging Face
# model ID is accepted without being added to this mapping.
MODEL_ALIASES = {
    "qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
}

MAX_PROMPT_TOKENS = 2048

PERTURBATION_PREFIXES = (
    "Solve the following mathematical problem:\n\n",
    "Find the answer to the following mathematical problem:\n\n",
    "Determine the solution to the following mathematical problem:\n\n",
)

RobustnessExample = tuple[str, int | bool]


def resolve_model_id(model_id: str) -> str:
    """Resolve a compatibility alias while accepting arbitrary model IDs."""
    return MODEL_ALIASES.get(model_id, model_id)


def classifier_artifact_name(model_id: str) -> str:
    """Return a safe, deterministic filename for a model-specific artifact."""
    resolved_id = resolve_model_id(model_id)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "__", resolved_id)
    return f"{safe_name}.pt"


def make_variants(problem: str) -> list[str]:
    """Vary instruction wording without rewriting the problem."""
    if not problem.strip():
        return []

    return [
        prefix + problem
        for prefix in PERTURBATION_PREFIXES
    ]


_loaded_model_id: str | None = None


def _load_model(model_id: str) -> tuple[Any, Any]:
    global _loaded_model_id

    if model_id != _loaded_model_id:
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
        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=True,
    )

    model = AutoModel.from_pretrained(
        model_id,
        dtype=dtype,
        local_files_only=True,
        device_map="auto" if device == "cuda" else "cpu",
    )

    model.eval()

    return tokenizer, model


def _tokenize(
    tokenizer: Any,
    text: str,
) -> dict[str, torch.Tensor]:
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


def _encode(
    model: Any,
    inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Encode one prompt into its final hidden-state vector.

    The backbone is frozen in this implementation, so gradients are not
    retained here. The classifier and projection remain trainable.
    """
    inputs = {
        name: value.to(model.device)
        for name, value in inputs.items()
    }

    with torch.inference_mode():
        output = model(
            **inputs,
            use_cache=False,
            return_dict=True,
        )

    # Each prompt is processed individually, so the final token is not padding.
    return output.last_hidden_state[0, -1].float().cpu()


def _get_problem_vectors(
    tokenizer: Any,
    model: Any,
    problem: str,
    problem_index: int | None = None,
) -> list[torch.Tensor] | None:
    variants = make_variants(problem)

    if not variants:
        return None

    text_config = model.config.get_text_config()
    context_limit = min(
        MAX_PROMPT_TOKENS,
        getattr(
            text_config,
            "max_position_embeddings",
            MAX_PROMPT_TOKENS,
        ),
    )

    texts = [problem, *variants]
    inputs = [
        _tokenize(tokenizer, text)
        for text in texts
    ]

    # Do not truncate because truncation could remove mathematical content.
    if any(
        not 0 < item["input_ids"].shape[1] <= context_limit
        for item in inputs
    ):
        if problem_index is not None:
            LOGGER.warning(
                "Problem %d exceeds the prompt limit; skipping",
                problem_index,
            )
        return None

    return [
        _encode(model, item)
        for item in inputs
    ]


class RobustnessClassifier(nn.Module):
    """
    Trainable robustness classifier.

    It learns:
    1. A projection of the frozen language-model representations.
    2. A robustness classifier using the original representation and
       learned stability distance.
    """

    def __init__(
        self,
        hidden_size: int,
        projection_size: int = 256,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.projection_size = projection_size

        self.projector = nn.Sequential(
            nn.Linear(hidden_size, projection_size),
            nn.LayerNorm(projection_size),
            nn.GELU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(projection_size + 1, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        original: torch.Tensor,
        variants: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logit:
                Robustness logit. Positive means robust.
            distance:
                Differentiable mean cosine distance.
        """
        if not variants:
            raise ValueError("At least one variant is required")

        representations = torch.stack(
            [original, *variants],
            dim=0,
        )

        projected = self.projector(representations)
        projected = F.normalize(
            projected,
            dim=-1,
            eps=1e-8,
        )

        original_projection = projected[0]
        variant_projections = projected[1:]

        similarities = (
            variant_projections
            * original_projection.unsqueeze(0)
        ).sum(dim=-1)

        distance = (
            1.0 - similarities
        ).clamp_min(0.0).mean()

        classifier_features = torch.cat(
            [
                original_projection,
                distance.unsqueeze(0),
            ],
            dim=0,
        ).unsqueeze(0)

        logit = self.classifier(
            classifier_features,
        ).squeeze()

        return logit, distance


def robustness_loss(
    logit: torch.Tensor,
    distance: torch.Tensor,
    label: torch.Tensor,
    lambda_distance: float = 0.1,
    distance_margin: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute:

        L = L_BCE + lambda_distance * L_distance

    The distance term is label-aware:

        robust example:
            minimize distance

        non-robust example:
            push distance above distance_margin
    """
    logit = logit.reshape(())
    label = label.float().reshape(())

    bce_loss = F.binary_cross_entropy_with_logits(
        logit,
        label,
    )

    distance_loss = (
        label * distance
        + (1.0 - label)
        * F.relu(distance_margin - distance)
    )

    total_loss = (
        bce_loss
        + lambda_distance * distance_loss
    )

    return total_loss, bce_loss, distance_loss


def train_robustness_classifier(
    model_id: str,
    examples: list[RobustnessExample],
    classifier: RobustnessClassifier | None = None,
    epochs: int = 3,
    learning_rate: float = 1e-4,
    lambda_distance: float = 0.1,
    distance_margin: float = 0.25,
    classifier_device: str | torch.device | None = None,
) -> RobustnessClassifier:
    """
    Train the robustness classifier.

    Args:
        model_id:
            Hugging Face model ID or configured alias.

        examples:
            List of:
                (problem_text, label)

            where label is:
                1 / True  = robust
                0 / False = non-robust

        classifier:
            Optional existing classifier to continue training.
    """
    resolved_id = resolve_model_id(model_id)
    tokenizer, model = _load_model(resolved_id)

    # Freeze the large language model. Only the robustness projection and
    # classification head are trained.
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    text_config = model.config.get_text_config()
    hidden_size = text_config.hidden_size

    if classifier is None:
        classifier = RobustnessClassifier(
            hidden_size=hidden_size,
        )

    if classifier.hidden_size != hidden_size:
        raise ValueError(
            "Classifier hidden size does not match model hidden size: "
            f"{classifier.hidden_size} != {hidden_size}"
        )

    if classifier_device is None:
        classifier_device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    classifier = classifier.to(classifier_device)
    classifier.train()

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=learning_rate,
    )

    valid_examples = [
        (problem, int(label))
        for problem, label in examples
        if problem.strip()
    ]

    if not valid_examples:
        raise ValueError("No valid training examples were provided")

    for epoch in range(epochs):
        epoch_total = 0.0
        epoch_bce = 0.0
        epoch_distance = 0.0
        update_count = 0

        for index, (problem, label_value) in enumerate(
            valid_examples
        ):
            vectors = _get_problem_vectors(
                tokenizer=tokenizer,
                model=model,
                problem=problem,
                problem_index=index,
            )

            if vectors is None:
                continue

            vectors = [
                vector.to(classifier_device)
                for vector in vectors
            ]

            original = vectors[0]
            variants = vectors[1:]

            logit, distance = classifier(
                original=original,
                variants=variants,
            )

            label = torch.tensor(
                float(label_value),
                device=classifier_device,
            )

            total_loss, bce_loss, distance_loss = robustness_loss(
                logit=logit,
                distance=distance,
                label=label,
                lambda_distance=lambda_distance,
                distance_margin=distance_margin,
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                classifier.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            epoch_total += float(total_loss.detach().item())
            epoch_bce += float(bce_loss.detach().item())
            epoch_distance += float(distance_loss.detach().item())
            update_count += 1

        if update_count:
            LOGGER.info(
                "Epoch %d/%d: total=%.5f bce=%.5f distance=%.5f",
                epoch + 1,
                epochs,
                epoch_total / update_count,
                epoch_bce / update_count,
                epoch_distance / update_count,
            )

    classifier.eval()
    return classifier


def predict_robustness(
    model_id: str,
    problems: list[str],
    classifier: RobustnessClassifier,
) -> list[bool]:
    """
    Predict robustness using the trained classifier.

    No distance threshold is used. The classifier's learned logit is used.
    """
    if not problems:
        return []

    if not any(problem.strip() for problem in problems):
        return [False for _ in problems]

    resolved_id = resolve_model_id(model_id)
    tokenizer, model = _load_model(resolved_id)

    classifier.eval()

    classifier_device = next(
        classifier.parameters()
    ).device

    predictions: list[bool] = []

    for index, problem in enumerate(problems):
        vectors = _get_problem_vectors(
            tokenizer=tokenizer,
            model=model,
            problem=problem,
            problem_index=index,
        )

        if vectors is None:
            predictions.append(False)
            continue

        vectors = [
            vector.to(classifier_device)
            for vector in vectors
        ]

        with torch.inference_mode():
            logit, _distance = classifier(
                original=vectors[0],
                variants=vectors[1:],
            )

            probability = torch.sigmoid(logit).item()

        # This is the classifier's probability decision boundary.
        # It is not a manually selected distance threshold.
        predictions.append(probability >= 0.5)

    return predictions


def save_robustness_classifier(
    classifier: RobustnessClassifier,
    path: str | Path,
) -> None:
    """Save classifier weights and architecture metadata."""
    path = Path(path)

    torch.save(
        {
            "hidden_size": classifier.hidden_size,
            "projection_size": classifier.projection_size,
            "state_dict": classifier.state_dict(),
        },
        path,
    )


def load_robustness_classifier(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> RobustnessClassifier:
    """Load a previously trained robustness classifier."""
    checkpoint = torch.load(
        path,
        map_location=device,
    )

    classifier = RobustnessClassifier(
        hidden_size=checkpoint["hidden_size"],
        projection_size=checkpoint["projection_size"],
    )

    classifier.load_state_dict(
        checkpoint["state_dict"],
    )

    classifier.to(device)
    classifier.eval()

    return classifier
