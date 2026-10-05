"""Train the representation-stability classifier from JSONL examples.

Each input line must be either:

    {"problem": "...", "label": true}

or:

    ["...", true]

Labels may be booleans or 0/1 values. The resulting classifier artifact is
intended to be loaded by solution.py during inference.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from stability_inference import (
    RobustnessExample,
    classifier_artifact_name,
    save_robustness_classifier,
    train_robustness_classifier,
)


LOGGER = logging.getLogger(__name__)


def _read_examples(path: Path) -> list[RobustnessExample]:
    examples: list[RobustnessExample] = []

    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            try:
                record: Any = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from error

            if isinstance(record, dict):
                problem = record.get("problem")
                label = record.get("label")
            elif isinstance(record, list) and len(record) == 2:
                problem, label = record
            else:
                raise TypeError(
                    f"Line {line_number} must contain a problem/label object "
                    "or a two-item array"
                )

            if not isinstance(problem, str):
                raise TypeError(
                    f"Line {line_number} has a non-string problem"
                )

            if isinstance(label, bool):
                normalized_label = label
            elif isinstance(label, int) and label in (0, 1):
                normalized_label = bool(label)
            else:
                raise TypeError(
                    f"Line {line_number} label must be true, false, 0, or 1"
                )

            examples.append((problem, normalized_label))

    if not examples:
        raise ValueError(f"No examples found in {path}")

    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the representation-stability robustness classifier"
    )
    parser.add_argument("examples", type=Path, help="Input JSONL file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output classifier artifact; defaults to a model-specific filename "
            "under stability_artifacts"
        ),
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Cached Hugging Face model ID or configured alias",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lambda-distance", type=float, default=0.1)
    parser.add_argument("--distance-margin", type=float, default=0.25)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.epochs <= 0:
        parser.error("--epochs must be positive")

    examples = _read_examples(args.examples)
    LOGGER.info("Loaded %d training examples", len(examples))

    classifier = train_robustness_classifier(
        model_id=args.model_id,
        examples=examples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        lambda_distance=args.lambda_distance,
        distance_margin=args.distance_margin,
    )

    output_path = args.output
    if output_path is None:
        output_path = (
            Path("stability_artifacts")
            / classifier_artifact_name(args.model_id)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_robustness_classifier(classifier, output_path)
    LOGGER.info("Saved classifier to %s", output_path)


if __name__ == "__main__":
    main()
