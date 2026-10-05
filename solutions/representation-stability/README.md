# Representation Stability

This solution trains a small robustness classifier on top of frozen language-model representations. For each problem, it compares the final hidden-state representation of the original prompt with three instruction-only variants.

## Method

For each problem, the implementation:

1. Creates three variants by prepending different instructions without changing the mathematical problem.
2. Formats the original and variants with the tokenizer's user-only chat template.
3. Runs four forward passes and extracts the final prompt-token hidden state.
4. Projects the four representations with a trainable projection.
5. Computes the mean cosine distance between the original and variant projections.
6. Predicts robustness with a trainable classifier using the original projection and distance.

The large language model is frozen. Only the projection and classifier head are trained.

This measures prompt-conditioned representation stability. It does not prove mathematical correctness or identical internal reasoning.

## Files

- `solution.py`: Codabench entry point; loads the trained classifier artifact once.
- `stability_inference.py`: model loading, representation extraction, training, scoring, and artifact serialization.
- `train_stability.py`: command-line training wrapper for JSONL examples.
- `stability_artifacts/<model-id>.pt`: generated model-specific classifier artifact; create one before running inference or packaging a submission.

## Training data

`train_stability.py` accepts one JSON record per line in either format:

```json
{"problem": "Find the value of 2 + 2.", "label": true}
["Solve x^2 - 5x + 6 = 0.", false]
```

`true`/`1` means robust and `false`/`0` means non-robust. Keep related or near-duplicate problems in the same split to avoid data leakage, and reserve a separate evaluation split.

## Train

From this directory, after the model has been cached locally:

```bash
uv run python train_stability.py data/train.jsonl \
  --model-id Qwen/Qwen3-8B \
  --epochs 3
```

By default this writes `stability_artifacts/Qwen__Qwen3-8B.pt`. You can choose a different output path with `--output`, but the submission entry point expects the model-specific filename by default.

The model must already be available in the Hugging Face cache because loading uses `local_files_only=True`. Cache it with:

```bash
uv run hf download deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
```

The legacy model alias `qwen3-8b:low` is also supported. Any exact Hugging Face model ID is accepted without changing the source code, provided that checkpoint is cached locally and supported by Transformers.

The classifier is trained on representations from one base model. Because models can have different hidden sizes and different representation spaces, train a separate classifier artifact for each base model you intend to evaluate. A classifier trained for one model should not be reused for another model merely because both are supported by the loader.

## Run locally

After training and creating the classifier artifact, run from `getting-started`:

```bash
uv run scripts/run_local.py solutions/representation-stability \
  --input-dir data/val-sample/input \
  --reference-dir data/val-sample/reference
```

The submission must include `solution.py`, `stability_inference.py`, and the generated classifier artifact. Do not bundle the base model weights. If evaluating multiple model IDs, package and select a separately trained artifact for each model ID rather than sharing one artifact.

## Runtime behavior

- Each problem requires four individual forward passes.
- GPU workers use automatic device placement and reduced precision when supported.
- CPU inference is supported but can be slow and memory-intensive for the 8B model.
- Blank problems and prompts exceeding 2,048 tokens are predicted as `False`.
- Predictions preserve input order and are returned as native Python booleans.
