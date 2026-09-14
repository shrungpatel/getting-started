# Representation Stability

A basic, training-free baseline that asks: when the instruction wording changes
without changing the mathematics, does the model's representation stay similar?
This is inspired by the logic-preserving variation idea in the proposed LPDS
notes; it is not a reproduction of that paper.

## Method

For each original problem, the solution:

1. Creates three variants by prepending different instructions to solve the
   problem. The original problem text, including all equations, stays intact.
2. Formats the original and variants with the same user-only chat template.
3. Runs four forward passes and extracts the final-layer hidden state at the
   last prompt token (including the generation prefix when the template adds it).
4. Computes cosine distance between the original representation and each variant:

   `distance = 1 - dot(h_original, h_variant) / (norm(h_original) * norm(h_variant))`

5. Predicts `True` when the mean distance is below `DISTANCE_THRESHOLD`.

**The default threshold of 0.05 is an uncalibrated heuristic.** No predictor has
been trained and no robustness accuracy is claimed. Change the constant in
`stability_inference.py` after calibrating on public development labels, keeping
related problems in the same split and reserving a separate evaluation split.

This measures **prompt-conditioned stability**, not stability while generating a
reasoning trace. Low distance does not prove mathematical correctness or identical
internal computation. Shared chat-template tokens and hidden-state anisotropy can
also yield high similarity. These instruction-only perturbations are deliberately
conservative and much narrower than mathematical paraphrasing or difficulty
scaling; they may not predict robustness to those broader changes.

## Files

- `solution.py`: required `are_robust(model_id: str, problems: list[str]) -> list[bool]` interface.
- `stability_inference.py`: variants, offline model loading, extraction, and scoring.

There are no training artifacts or additional dependencies. Loader regression tests
are in `../../tests/test_representation_stability.py`.

## Runtime behavior

Accepts Hugging Face checkpoint IDs without a model allowlist. The legacy alias
`qwen3-8b:low` resolves to `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`.
Model and tokenizer loading use `local_files_only=True`; each checkpoint must
already be cached and supported by the installed Transformers runtime. The base
model must expose `last_hidden_state` for text inputs. Quantized checkpoints also
require their runtime backends; accepting an ID does not guarantee compatibility.
The current model is reused, with evaluation mode and inference-only forward
passes, and evicted before a different checkpoint loads. On GPU workers,
`device_map="auto"` enables GPU/CPU placement rather than forcing the entire
checkpoint onto one GPU. Offloading can increase runtime substantially.
Sequences are processed individually to keep memory use bounded and avoid padding
alignment issues. No solution text is generated or vocabulary logits computed.

Predictions preserve input order and use native Python booleans. Empty input
returns `[]`. Blank problems, invalid representations, and
problems for which any formatted variant exceeds 2,048 tokens (or the model's
smaller context limit) receive `False`. Inputs are not truncated because dropping
mathematics could invalidate a comparison. Length-limit fallbacks emit warnings.
Model-loading and unexpected inference errors propagate
rather than silently disguising a broken runtime as robustness predictions.

## Run locally

From `getting-started`, after importing the public validation sample and caching
the models present in the input:

```bash
uv run scripts/run_local.py solutions/representation-stability \
  --input-dir data/val-sample/input \
  --reference-dir data/val-sample/reference
```

For submission, place `solution.py` and `stability_inference.py` at the ZIP root.
Do not bundle model weights. Benchmark the four-forward-pass cost against the
evaluation time limit before submitting.
