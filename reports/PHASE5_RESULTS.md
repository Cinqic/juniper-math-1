# Phase 5 Smoke-Pretraining Results

**SMOKE PIPELINE VALIDATION ONLY.** Every number below demonstrates that a
pipeline stage executed correctly — none of them is a claim about
mathematical capability. See [`docs/TRAINING.md`](../docs/TRAINING.md) for
the design and [`reports/PHASE5_REPORT.md`](PHASE5_REPORT.md) for the
engineering report.

## Experiment identity

- Run ID: `phase5-smoke-v1` (`experiments/phase5-smoke/experiment.yaml`)
- Git commit at run time: `1f5bc9088c4517ca0972f68c11bb7897de51c22c`
- Config: `config/training.yaml` (sha256
  `ceec5c1c4ed2efe7a17fee0290c878ac00b656bf50189612e0ab39434ea0ce60`)
- Seed: `5004032`

## Frozen artifact identities

| Artifact | Identity |
|---|---|
| Architecture | v0.1.0, 5,004,032 trainable parameters (programmatically verified) |
| Tokenizer | `juniper-math-tokenizer-v1` |
| Dataset | `juniper-math-dataset-v1`, whole-dataset identity `bf9933f032a58b4eb618b32156783b8563097a5fc1c0ef26be4f76445128d25a` |
| Evaluation suite (tool-format) | `phase4-tool-use-v2` (`evals/phase4_tool_use_v2.json`) |
| Tool protocol | `juniper-tool-protocol-v1` v1.0.0 |

## Smoke subset

Selected by fixed-stride sampling (see `docs/TRAINING.md`) —
`data/processed/phase5-smoke/smoke_manifest.json` (not committed;
reproducible from the frozen dataset + this config):

| Split | Examples | Stride | Offset | Total tokens |
|---|---|---|---|---|
| train | 2,048 | 716 | 624 | 73,231 |
| validation | 256 | 316 | 172 | 9,604 |

## Environment

| | |
|---|---|
| OS | Linux 7.0.0-30-generic |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| CUDA | 13.0 |
| GPU | NVIDIA GeForce RTX 2060 (6,144 MiB), driver 595.84 |
| CPU | AMD Ryzen 7 5700G (16 logical cores) |
| System RAM | 15.0 GiB |

## Smoke training configuration

Effective batch size 16 (micro-batch 8 × grad-accum 2), sequence length
256, fp32, AdamW (lr 3e-4, weight_decay 0.01, betas 0.9/0.95), cosine
schedule with 20-step warmup down to 10% of peak lr, gradient-norm clip
1.0, 200 optimizer steps.

## Training behavior

| | |
|---|---|
| Initial validation loss (before any training) | 8.3804 (9,860 tokens) |
| Loss after 1st optimizer step | 8.3944 |
| Validation loss @ step 50 | 4.4643 |
| Validation loss @ step 100 | 3.0502 |
| Validation loss @ step 150 | 2.4472 |
| Validation loss @ step 200 (final) | 2.2346 |
| Final training-step loss | 2.2564 |
| Gradient finiteness | PASS — every step, checked before optimizer.step() |
| Parameter finiteness | PASS — every step, checked after optimizer.step() |
| Total tokens_seen (loss-bearing, post-shift) | 117,828 |

Loss decreased monotonically at the validation checkpoints (every 50
steps) from 8.38 to 2.23 — a clear, non-trivial downward trend, not a
single lucky step.

## Resource observations

| | |
|---|---|
| Peak CUDA memory | 494.9 MiB (well within the 6,144 MiB budget) |
| Training throughput | ~6,800 loss-bearing tokens/sec (200 steps in 17.3s) |
| Wall clock, full `train run --evaluate` | ~18-25s |
| Wall clock, `train resume-test` (3 full training passes) | ~37-40s |
| Final checkpoint size | 60,122,115 bytes (~57.3 MiB) |

## Generation: before vs. after training

Fixed prompts, greedy decoding, 32 max new tokens, identical across
initialization and the final checkpoint:

| Prompt | Before training | After training |
|---|---|---|
| `12 + 7 =` | `12 + 7 =ckckckckck...` (degenerate repetition) | `12 + 7 = -111.\n<final>-11` |
| `What is 5 times 6?` | `...` + repeated replacement-character garbage | `What is 5 times 6?\n<final>11` |
| `Convert 3 miles to meters.` | `...ActivityActivityActivity...` (degenerate repetition) | `Convert 3 miles to meters.\n<final>10` |

**Interpretation:** training clearly changed model behavior — the smoke
model learned the dataset's `<final>` answer-tag structural format within
200 steps. It did **not** learn correct arithmetic (`-11`, `11`, `10` are
all wrong). This is the expected and correct outcome for a 2,048-example,
200-step smoke run: it demonstrates the optimization pipeline works, not
that the model can do math.

## Checkpoint save / restore

`python -m juniper_math checkpoint inspect checkpoints/phase5-smoke/step_000200_final.pt`
confirmed schema version, architecture identity match, step 200,
tokens_seen 117,828, seed, git commit, full training config, and the
presence of optimizer/scheduler state — all restorable via the existing
Phase 1 `load_checkpoint` transactional restore path. Checkpoint sha256:
`d662ce1dfb9ced6699ec65de22d4cea144e2a38edaccbd68e4af87f973bd08a9`.

## Resume comparison (Sec. 22 gate)

`python -m juniper_math train resume-test`, run three times independently
on this GPU (once during initial validation, twice during final
verification) — every run:

| | Run A (uninterrupted) | Run B (interrupted @ step 100, resumed) |
|---|---|---|
| Final step | 200 | 200 |
| Tokens seen | 117,828 | 117,828 |
| Final training loss | 2.2564382435793573 | 2.2564382435793573 |
| Loss-history max abs diff (common steps) | 0.0 | |
| Max parameter abs diff | 0.0 | |
| Fixed-prompt generations | identical | identical |

**Result: bitwise-exact equivalence on CUDA**, not merely "training
continued without crashing." `torch.use_deterministic_algorithms(...,
warn_only=True)` is a best-effort request rather than a hardware
guarantee, so this exact equivalence is a genuinely good outcome for this
model size/run length, not something to assume will always hold at larger
scale — documented as a caveat in the tool's own output.

## Evaluation execution

`python -m juniper_math evaluate --checkpoint <final> --sample-size 40`
against `evals/phase4_tool_use_v2.json`:

| | |
|---|---|
| Cases evaluated | 40 |
| Generations that emitted a `<tool_call>` tag | 0 |
| Well-formed (protocol-valid) tool calls | 0 |
| Tool-name matches | 0 |

**0% is the expected and correct result** at this smoke scale (2,048
training examples, 200 steps, no tool-use-specific curriculum) — what
matters is that suite loading, prompt encoding, generation, `<tool_call>`
extraction, and protocol parsing (`juniper_math.tools.protocol.parse_tool_call`)
all executed across 40 cases without a single crash or silently-skipped
case. This is a SMOKE PIPELINE VALIDATION ONLY result, explicitly labeled
as such in the CLI's own output.

## Test suite (full, from a clean state)

```
pytest -v          # 586 passed, 2 warnings (pre-existing CUDA determinism warnings)
ruff check .        # All checks passed
ruff format --check . # 177 files already formatted
mypy                 # Success: no issues found in 58 source files
python -m juniper_math validate-env       # PASS
python -m juniper_math validate-config    # PASS
python -m juniper_math hash verify        # PASS (all artifacts, incl. new training_config entry)
python -m juniper_math manifests-validate # PASS
python -m juniper_math model --device cuda # PASS, 5,004,032 params
python -m juniper_math tokenizer validate # PASS
python -m juniper_math tools self-test    # PASS, 9/9
python -m juniper_math evals validate     # PASS
```

Full `dataset build`/`dataset validate`/`dataset verify` (1.6M examples)
were **not** re-run — the dataset is frozen and unchanged since its Phase 4
approval, and `hash verify` confirms its shard manifest, stats, and
whole-dataset identity files are byte-identical to the approved artifact.

## Known limitations

- Checkpoint binaries (~60MB each) are not committed to Git or otherwise
  archived to a remote store — they are disposable and exactly reproducible
  from the committed config + frozen dataset (demonstrated by the resume
  test's bitwise reproducibility across independent runs). See
  `checkpoints/README.md`.
- The resume-equivalence bitwise match on CUDA is a genuine result for this
  run, not a hardware guarantee for all future runs at any scale — treat
  future resume tests as needing their own verification, with tolerance
  comparison as the documented fallback.
- Generation has no KV cache and is not batched — fine at smoke scale
  (32 max new tokens, a handful of fixed prompts and eval cases), would
  need real work before Phase 6/7 inference-heavy usage.
- 0% tool-format accuracy is exactly what smoke scale should produce; it
  is not evidence about what a real pilot/production run would achieve.
