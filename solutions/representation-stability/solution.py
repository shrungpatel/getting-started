"""Codabench entry point for the trained representation-stability probe."""

from pathlib import Path

import torch

from stability_inference import (
    classifier_artifact_name,
    load_robustness_classifier,
    predict_robustness,
)


CLASSIFIER_DIRECTORY = Path(__file__).parent / "stability_artifacts"

_classifier = None
_classifier_model_id: str | None = None


def are_robust(model_id: str, problems: list[str]) -> list[bool]:
    """Return one native Python boolean per problem, preserving input order."""
    global _classifier, _classifier_model_id

    if _classifier is None or _classifier_model_id != model_id:
        classifier_path = (
            CLASSIFIER_DIRECTORY / classifier_artifact_name(model_id)
        )
        if not classifier_path.is_file():
            raise FileNotFoundError(
                "The trained robustness classifier is missing: "
                f"{classifier_path}. Run train_stability.py for model "
                f"{model_id!r} first."
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _classifier = load_robustness_classifier(
            classifier_path,
            device=device,
        )
        _classifier_model_id = model_id

    return predict_robustness(
        model_id=model_id,
        problems=problems,
        classifier=_classifier,
    )
