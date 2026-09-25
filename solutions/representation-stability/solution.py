"""Codabench entry point for the representation-stability baseline."""

from stability_inference import predict_robustness


def are_robust(model_id: str, problems: list[str]) -> list[bool]:
    """Return one native Python boolean per problem, preserving input order."""
    return predict_robustness(model_id, problems)







