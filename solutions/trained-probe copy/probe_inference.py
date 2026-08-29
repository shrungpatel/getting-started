"""Artifact loading, hidden-state extraction, and linear-probe scoring.

This module implements the inference half of the trained-probe baseline. The
pipeline for each problem is:

1. **Load an artifact** for the requested ``model_id`` (a pretrained probe
   ensemble exported by the training repo). See :func:`_load_artifact`.
2. **Encode the problem** into a single hidden-state vector by running one
   forward pass of the offline language model and reading the final
   prompt-token representation at the probe's chosen layer. See
   :func:`_encode_problem`.
3. **Score the probes** by computing each linear probe's margin against that
   vector and averaging them. Non-negative mean margin => "robust". See
   :func:`mean_ensemble_margin`.

If no artifact is found for a model, the method degrades gracefully and predicts
``False`` for every problem, so the solution still builds and runs.
"""

from __future__ import annotations

import pickle
import re
import torch
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer


# Filenames/locations searched for a probe artifact, relative to this file.
PICKLE_ARTIFACT_NAME = "probe_artifact.pkl"  # preferred ensemble format
NPZ_ARTIFACT_NAME = "probe_artifact.npz"  # legacy single-probe fallback
MODEL_ARTIFACT_DIR = "probe_artifacts"  # dir holding per-model artifacts
# Only this scoring strategy is supported by mean_ensemble_margin below.
RECOMMENDED_STRATEGY = "best_layer_mean_margin"


@dataclass(frozen=True)
class ProbeArtifact:
    """A loaded probe artifact plus the metadata needed to run it.

    ``kind`` is ``"pickle"`` for the probe ensemble format or ``"npz"`` for the
    legacy single-probe format; it selects the scoring path downstream.
    """

    model_id: str  # HF model id the probe was trained against
    system_prompt: str  # prompt prefix used when the probe was trained
    data: dict[str, Any]  # raw artifact payload (schema depends on kind)
    kind: str


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


def _safe_model_id(model_id: str) -> str:
    """Turn a HF model id into a filesystem-safe token.

    e.g. ``"Qwen/Qwen3-8B"`` -> ``"Qwen_Qwen3-8B"`` so it can be used as a
    per-model artifact filename.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


def _artifact_candidates(model_id: str) -> list[Path]:
    """Return artifact paths to try, most specific first.

    A model-specific artifact (``probe_artifacts/<safe-id>.pkl``) wins over a
    generic one, letting a single submission ship probes for several models.
    """
    solution_dir = Path(__file__).resolve().parent
    safe_model_id = _safe_model_id(model_id)
    candidates = []
    # 1. Per-model artifacts named after the (sanitized) model id.
    if safe_model_id:
        candidates.extend(
            [
                solution_dir / MODEL_ARTIFACT_DIR / f"{safe_model_id}.pkl",
                solution_dir / MODEL_ARTIFACT_DIR / f"{safe_model_id}.npz",
            ]
        )
    # 2. Generic artifacts, checked next to solution.py and in the artifact dir.
    candidates.extend(
        [
            solution_dir / PICKLE_ARTIFACT_NAME,
            solution_dir / MODEL_ARTIFACT_DIR / PICKLE_ARTIFACT_NAME,
            solution_dir / NPZ_ARTIFACT_NAME,
        ]
    )
    return candidates


def _load_artifact(model_id: str) -> ProbeArtifact | None:
    """Load the first existing candidate artifact, or ``None`` if none exist."""
    path = next(
        (candidate for candidate in _artifact_candidates(model_id) if candidate.is_file()),
        None,
    )
    if path is None:
        return None
    # Dispatch on extension: .pkl is the ensemble format, everything else npz.
    return load_pickle_artifact(path) if path.suffix == ".pkl" else _load_npz_artifact(path)


def load_pickle_artifact(path: Path) -> ProbeArtifact:
    """Load and validate a schema-v2 probe-ensemble pickle.

    Raises ``RuntimeError`` on any schema mismatch so a malformed artifact fails
    loudly instead of silently producing garbage predictions.
    """
    with path.open("rb") as handle:
        data = pickle.load(handle)

    # Validate the artifact shape before trusting any of its contents.
    if not isinstance(data, dict):
        raise RuntimeError(f"{path.name} must contain a dictionary artifact")
    if data.get("schema_version") != 2:
        raise RuntimeError(
            f"{path.name} has unsupported schema_version {data.get('schema_version')!r}"
        )
    if data.get("artifact_type") != "all_folds_layers_seed_probe_ensemble":
        raise RuntimeError(
            f"{path.name} has unsupported artifact_type {data.get('artifact_type')!r}"
        )
    if not data.get("groups"):
        raise RuntimeError(f"{path.name} contains no probe groups")
    if "best_layer_index" not in data:
        raise RuntimeError(f"{path.name} is missing best_layer_index")

    # If a strategy is recorded it must be the one we know how to score.
    strategy = data.get("recommended_strategy") or {}
    if strategy and strategy.get("name") != RECOMMENDED_STRATEGY:
        raise RuntimeError(f"{path.name} recommends unsupported strategy {strategy.get('name')!r}")

    return ProbeArtifact(
        model_id=str(data.get("model_id") or ""),
        system_prompt=str(data.get("system_prompt") or ""),
        data=data,
        kind="pickle",
    )


def _load_npz_artifact(path: Path) -> ProbeArtifact:
    """Load the legacy single-probe ``.npz`` format (compatibility fallback)."""
    data = np.load(path, allow_pickle=False)
    required = {"model_id", "layer_index", "weights", "bias", "threshold"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise RuntimeError(f"{path.name} is missing required entries: {missing}")

    # A single probe is stored as flat weight/bias/threshold arrays.
    weights = np.asarray(data["weights"], dtype=np.float32).reshape(-1)
    if weights.size == 0:
        raise RuntimeError(f"{path.name} contains empty probe weights")

    return ProbeArtifact(
        model_id=str(_scalar(data["model_id"])),
        system_prompt=str(_scalar(data["system_prompt"])) if "system_prompt" in data.files else "",
        data={
            "layer_index": int(_scalar(data["layer_index"])),
            "weights": weights,
            "bias": float(_scalar(data["bias"])),
            "threshold": float(_scalar(data["threshold"])),
        },
        kind="npz",
    )


def _scalar(value: Any) -> Any:
    """Unwrap a 0-d numpy array to a Python scalar; pass other values through."""
    return value.item() if hasattr(value, "item") else value


def _layer_index(artifact: ProbeArtifact) -> int:
    """Resolve which model layer's hidden state the probe reads.

    For the ensemble format, prefer the strategy's ``layer_index`` and fall back
    to the globally selected ``best_layer_index``.
    """
    if artifact.kind == "pickle":
        strategy = artifact.data.get("recommended_strategy") or {}
        return int(strategy.get("layer_index", artifact.data["best_layer_index"]))
    return int(artifact.data["layer_index"])


# ---------------------------------------------------------------------------
# Hidden-state extraction
# ---------------------------------------------------------------------------


def _load_model(model_id: str) -> tuple[Any, Any]:
    """Load the tokenizer and model from the offline HF cache.

    ``local_files_only=True`` is essential: the evaluation container has no
    network, so anything not already cached must fail rather than hang.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # Prefer bfloat16 where supported, else float16; both halve memory use.
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        local_files_only=True,
    )
    model.to(device)
    model.eval()  # disable dropout etc. for deterministic representations
    return tokenizer, model


def _build_prompt(tokenizer: Any, problem_text: str, system_prompt: str) -> str:
    """Format the problem exactly as it was formatted during probe training.

    Uses the tokenizer's chat template when a system prompt is present (so the
    encoding matches an instruct model's expected format), and falls back to a
    plain concatenation if templating is unavailable or fails.
    """
    if system_prompt and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": problem_text},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Some tokenizers lack a usable template; degrade to plain text.
            pass
    return f"{system_prompt}\n\n{problem_text}" if system_prompt else problem_text


def _encode_problem(problem_text: str, artifact: ProbeArtifact) -> Any:
    """Run one forward pass and return the probe's input hidden-state vector.

    The vector is the last prompt token's hidden state at the probe's layer,
    mirroring how the probes were trained. Returned as a float32 numpy array.
    """
    tokenizer, model = _load_model(artifact.model_id)
    prompt = _build_prompt(tokenizer, problem_text, artifact.system_prompt)
    model_inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    model_inputs = {name: value.to(model.device) for name, value in model_inputs.items()}

    # No gradients or KV cache needed; we only want the hidden states.
    with torch.no_grad():
        output = model(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    hidden_states = output.hidden_states
    if hidden_states is None:
        raise RuntimeError("model forward pass returned no hidden states")

    # hidden_states is a tuple of (num_layers + 1) tensors, each
    # (batch, seq_len, hidden_dim). Validate the layer index is in range.
    layer_index = _layer_index(artifact)
    if not -len(hidden_states) <= layer_index < len(hidden_states):
        raise RuntimeError(
            f"probe layer {layer_index} is outside model hidden-state range "
            f"0..{len(hidden_states) - 1}"
        )
    # [0, -1, :] => batch item 0, final token, full hidden vector.
    return hidden_states[layer_index][0, -1, :].detach().float().cpu().numpy()


# ---------------------------------------------------------------------------
# Probe scoring
# ---------------------------------------------------------------------------


def _single_probe_margin(vector: Any, artifact: ProbeArtifact) -> float:
    """Signed margin for the legacy single-probe (.npz) format.

    ``margin = weights . vector + bias - threshold``; ``>= 0`` means "robust".
    """
    weights = np.asarray(artifact.data["weights"], dtype=np.float32).reshape(-1)
    if vector.shape != weights.shape:
        raise RuntimeError(
            f"probe dimension mismatch: got {vector.shape[0]}, expected {weights.shape[0]}"
        )
    score = float(np.dot(vector, weights) + float(artifact.data["bias"]))
    return score - float(artifact.data["threshold"])


def mean_ensemble_margin(vector: Any, artifact: ProbeArtifact) -> float:
    """Average signed margin across every probe in the ensemble.

    The ensemble spans groups (e.g. cross-validation folds), each holding a
    stack of per-seed probes for the chosen layer. We compute
    ``weights @ vector + bias - threshold`` for all of them and return the mean;
    a non-negative mean is the "robust" decision.
    """
    layer_index = _layer_index(artifact)
    margins = []
    for group in artifact.data["groups"]:
        probes = group.get("probes", {})
        # Layer keys may be stored as ints or strings; accept either.
        probe = probes.get(layer_index, probes.get(str(layer_index)))
        if probe is None:
            continue

        # weights: (n_seeds, hidden_dim); normalize a 1-D probe to 2-D.
        weights = np.asarray(probe["weights"], dtype=np.float32)
        if weights.ndim == 1:
            weights = weights.reshape(1, -1)
        if weights.ndim != 2 or weights.shape[1] != vector.shape[0]:
            raise RuntimeError(
                f"probe dimension mismatch: got {vector.shape[0]}, expected {weights.shape[-1]}"
            )

        # bias/threshold are per-seed vectors aligned with weights' first axis.
        bias = np.asarray(probe["bias"], dtype=np.float32).reshape(-1)
        threshold = np.asarray(probe["threshold"], dtype=np.float32).reshape(-1)
        if bias.size != weights.shape[0] or threshold.size != weights.shape[0]:
            raise RuntimeError("probe ensemble arrays have inconsistent seed dimensions")

        # One margin per seed in this group; collect across all groups.
        margins.extend((weights @ vector + bias - threshold).astype(float).tolist())

    if not margins:
        raise RuntimeError(f"artifact contains no probes for layer {layer_index}")
    # float64 mean to avoid accumulation error across many seeds/folds.
    return float(np.mean(np.asarray(margins, dtype=np.float64)))


def _predict_problem(problem_text: str, artifact: ProbeArtifact) -> bool:
    """Encode one problem and turn its probe margin into a robustness bool."""
    vector = np.asarray(_encode_problem(problem_text, artifact), dtype=np.float32)
    margin = (
        mean_ensemble_margin(vector, artifact)
        if artifact.kind == "pickle"
        else _single_probe_margin(vector, artifact)
    )
    return bool(margin >= 0.0)


def predict_robustness(model_id: str, problem_texts: list[str]) -> list[bool]:
    """Return one robustness prediction for each problem text.

    Loads the artifact once, then scores every problem. If no artifact exists
    for ``model_id`` the method predicts ``False`` everywhere so the submission
    still runs as a valid (if trivial) baseline.
    """
    artifact = _load_artifact(model_id)
    if artifact is None:
        return [False for _ in problem_texts]
    return [_predict_problem(problem_text, artifact) for problem_text in problem_texts]
