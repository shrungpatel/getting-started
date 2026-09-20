# Causal Stability

**Does the same causally important computation remain responsible for the answer?**

This training-free baseline compares **activation-intervention effect profiles**
across instruction perturbations. Unlike representation similarity, it measures
whether deliberately replacing an internal activation changes answer probability.
It is inspired by the causal-mediation and causal-variable perspectives in:

- *A Mechanistic Interpretation of Arithmetic Reasoning in Language Models using
  Causal Mediation Analysis*
- *Internal Causal Mechanisms Robustly Predict Language Model Out-of-Distribution
  Behaviors*

This is an operational approximation, not a reproduction of those papers or a
claim to identify a complete reasoning circuit. There are no trained artifacts,
external services, extra dependencies, or model-specific probes.

## From causal mediation to stability

The motivating graph is `X -> V -> Y`, where `X` is the problem, `V` an internal
computation, and `Y` the answer. Other routes from `X` to `Y` remain possible.
Ideally, one would measure:

```text
CE_v = P(correct answer | do(V = correct))
     - P(correct answer | do(V = incorrect))

D(x, x') = d(c(x), c(x'))
```

Evaluation supplies neither correct answers nor known-correct internal states.
Here, a greedy answer from the original prompt is the fixed reference `y*`.
Clean activations are proxies for correct states; corrupted-prompt activations
are proxies for incorrect states. **Neither proxy is verified to be correct or
incorrect.** The outcome is the teacher-forced sequence probability of `y*`,
not the probability of mathematical correctness:

```text
P(y* | x) = exp(sum_t log P(y*_t | x, y*_<t))
```

The same target token sequence is scored everywhere. The terminating EOS token
is not included, so this is the probability of the reference-answer prefix.

For each selected decoder block `l`, the mediator `V_l` is its output vector at
the **last prompt token**. Two effects hold the recipient input fixed while
replacing only that vector:

```text
necessity_l(x) = P(y* | x)
              - P(y* | x, do(V_l = V_l(corrupt(x))))

recovery_l(x)  = P(y* | corrupt(x), do(V_l = V_l(x)))
              - P(y* | corrupt(x))

c(x) = [necessity_1, recovery_1, ..., necessity_L, recovery_L]
```

The clean/corrupt baseline is equivalent to intervening with its own activation.
Each patch is independent: interventions at different layers are never combined.
Donor states come from the matching clean/corrupted version, not from another
wording variant. Captured states cannot see future answer tokens because the
model is causal; answer tokens are nevertheless teacher-forced for scoring.

Signed relative L1 distance compares both effect allocation and magnitude:

```text
d(a, b) = sum_i |a_i - b_i| / (sum_i |a_i| + sum_i |b_i|)
D       = max over variants x' of d(c(x), c(x'))
```

Finite, sufficiently strong profiles have distances in `[0, 1]`. Signs are
preserved: an intervention that helps instead of hurts is not treated as an
equivalent mechanism. Weak/zero/nonfinite profiles are inconclusive and produce
`False`, rather than being declared perfectly stable.

## Pipeline

1. Keep the original mathematical text intact. Compare it with two variants that
   prepend different requests to solve the problem.
2. Use the tokenizer's user-only chat template when available. Request a short
   answer and append a common `Final answer:` readout. Pass
   `enable_thinking=False` to templates that support it; if the template still
   ends with an open `<think>` tag, close that tag before the readout.
3. Greedily generate at most 128 new tokens from the original prompt. Prefer a
   balanced `\boxed{...}` answer; otherwise require a single nonempty answer line
   followed by EOS. Reject unfinished, ambiguous multiline, or over-32-token
   answers. A complete boxed answer can be used even if generation reached its
   cap. Tokenize the extracted answer once, with a leading space.
4. For each wording, compare its prompt token IDs with the empty-problem prompt
   to locate the variable span. Replace that span with repeated neutral `?`
   tokens. Keep sequence length, token positions, attention mask, and the common
   instruction/readout suffix fixed within that pair. This is an information
   ablation, **not a semantics-preserving perturbation**; only the clean wording
   variants must preserve the mathematics. Token-boundary merges may include
   adjacent whitespace in the corrupted span.
5. Discover the decoder stack structurally and sample up to four evenly spaced
   internal blocks. Exclude the terminal block where possible: patching its
   final readout can trivially replace the first answer-token distribution.
   A single-block model uses that sole block, with this interpretive limitation.
6. Capture clean/corrupted states and run both interventions at each sampled
   layer. Compare the resulting profiles with the original's profile.
7. Return `True` only if **every** variant meets the distance and answer-support
   criteria. Stop early on a failed criterion.

### Default decision rules

These constants live near the top of `causal_inference.py`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `DISTANCE_THRESHOLD` | `0.25` | Maximum signed relative L1 distance |
| `MIN_CAUSAL_EFFECT` | `1e-4` | Minimum mean absolute effect and maximum positive effect, per profile |
| `MIN_ANSWER_PROBABILITY` | `1e-4` | Minimum clean reference-answer sequence probability |
| `MIN_ANSWER_PROBABILITY_RATIO` | `0.5` | Each variant must retain at least this fraction of original answer probability |
| `NUM_CAUSAL_LAYERS` | `4` | Maximum sampled blocks |
| `MAX_PROMPT_TOKENS` | `2048` | Cap per formatted prompt; no truncation |
| `MAX_NEW_TOKENS` | `128` | Reference-generation cap |
| `MAX_ANSWER_TOKENS` | `32` | Maximum extracted reference-answer length |

**These are uncalibrated heuristics. No robustness accuracy is claimed.** Calibrate
on public development labels with related originals/perturbations grouped in the
same split, and evaluate on held-out groups. Both sequence probability and the
effect floor penalize longer or unusually formatted answers; thresholds should
be checked across answer lengths and model/tokenizer families. Distance is not
scale-invariant: confidence changes can count as instability even if the effect
profile has the same shape.

## Model independence and runtime

`solution.py` exposes the exact competition interface:

```python
are_robust(model_id: str, problems: list[str]) -> list[bool]
```

The implementation accepts arbitrary locally cached Hugging Face checkpoint IDs
supported by `AutoModelForCausalLM` and `AutoTokenizer`; there is no checkpoint
allowlist or model-specific layer path. The competition alias `qwen3-8b:low` is
resolved to its checkpoint for compatibility, without restricting other IDs.
Both loaders use `local_files_only=True`, and remote custom code is not enabled.

Supported architectures are decoder-only causal LMs with an identifiable
`ModuleList`/`Sequential` of decoder blocks. Blocks must run once each and return
a batch-first `[batch, sequence, hidden]` tensor, or a tuple/list whose first item
is that tensor. Discovery uses configured layer count where available and
otherwise looks for a homogeneous repeated stack. If several stacks match, set
`BLOCK_STACK_PATH` to the appropriate `model.named_modules()` path. Architectures
with recurrent/shared blocks, different output conventions, encoder-decoder
models, or unusual assistant-channel templates need an adapter; accepting an ID
does not guarantee support for every Transformers architecture. Quantized models
also need their runtime backend installed.

The model runs in evaluation/inference mode. One checkpoint is cached and evicted
before another loads. CPU uses float32; CUDA uses bf16 if supported, otherwise
fp16, with Accelerate automatic placement. Inputs are processed individually,
without padding. Hooks are always removed, including when inference raises.
Only sampled token vectors are stored, on CPU. Models exposing a
`logits_to_keep`/`num_logits_to_keep` forward parameter compute just the required
answer logits; other models compute full-sequence logits.

With four sites and three prompts, the maximum cost is **30 teacher-forced
forward passes plus one greedy generation per problem**. This is considerably
more expensive than representation similarity. Benchmark the actual evaluation
batch against the 3,600-second budget; reduce layers or perturbations if needed,
and recalibrate the decision rule afterward. Full-vocabulary logits, temporary
patched hidden-state copies, long contexts, and CPU offloading can be costly.

Empty batches return `[]`. Blank problems, over-budget prompts, missing short
answers, and weak/nonfinite scores return native Python `False`. Prompt limits
respect known model/tokenizer context limits and reserve answer space. Prompts
are never truncated, since dropping mathematical content invalidates comparison.
Model-loading errors, unsupported architecture errors, and unexpected inference
errors propagate rather than silently masquerading as robustness estimates.

## Interpretation and limitations

- This measures stability of **coarse node-intervention effects**, not equality
  of the full causal pathways. Different circuits may share an effect profile;
  effects from different layers overlap and cannot be added as independent causes.
- The readout is a short direct-answer computation, not an analysis of a complete
  generated reasoning trace. Disabling/closing thinking changes the generation
  protocol, and reasoning-heavy models may fail to provide a short answer.
- Stable wrong answers can appear robust. A clean donor can encode the wrong
  answer; a corrupted donor is not a known incorrect causal-variable value.
- Repeated neutral tokens and transplanted hidden states can be off-distribution.
  They test sensitivity to a chosen ablation, not a formally identified natural
  indirect effect with all mediation assumptions established.
- Only one token position and a few layers are measured. Mechanisms at other
  positions, attention heads, or features can change unnoticed.
- Wording prefixes are deliberately conservative. They do not cover the full
  competition distribution of mathematical paraphrases or other perturbations.
- Reference-answer support is an additional guard, not a check that each variant
  would independently generate that same answer. Greedy decoding is repeatable
  under the same runtime, but bitwise results across GPU backends are not promised.

## Run and submit

From `getting-started`, with public validation data imported and the evaluated
checkpoint already cached:

```bash
uv run scripts/run_local.py solutions/causal-stability \
  --input-dir data/val-sample/input \
  --reference-dir data/val-sample/reference
```

Submit a ZIP with `solution.py` and `causal_inference.py` at its root. The README
is optional. Do not include model weights, caches, or test files. No training
artifact is required.
