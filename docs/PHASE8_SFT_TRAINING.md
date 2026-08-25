# Phase 8: Mathematical Instruction and Tool Training (SFT)

Canonical config: [`config/training_phase8_sft.yaml`](../config/training_phase8_sft.yaml).
This document describes the design; see
[`reports/PHASE8_PLAN.md`](../reports/PHASE8_PLAN.md) for the plan,
[`reports/PHASE8_DATASET.md`](../reports/PHASE8_DATASET.md) for the SFT
corpus, [`reports/PHASE8_PREFLIGHT.md`](../reports/PHASE8_PREFLIGHT.md) for
the bounded LR/mixture comparison, and
[`reports/PHASE8_RESULTS.md`](../reports/PHASE8_RESULTS.md) for the
canonical run's measured results.

## Purpose

Phase 7 (`docs/TRAINING.md`, `docs/PILOT_TRAINING.md`) produced a Base with
pretrained representations but no reliable instruction-following or
tool-use behavior (frozen-suite tool-call emission peaked at 80.5% then
*declined* to 37.8% with 0% valid parse throughout — see
`reports/PHASE7_RESULTS.md`). Phase 8 does not pretrain further; it
supervises the Base on the specific interaction loop the project needs:
interpret → decide direct-vs-tool → (call tool | answer | clarify |
refuse) → interpret the real tool result → produce a concise final answer.

## SFT subset selection

[`src/juniper_math/sft_data.py`](../src/juniper_math/sft_data.py) selects a
deterministic, **flattened** (not corpus-proportional) per-category subset
of the frozen `juniper-math-dataset-v1` train/validation splits — see ADR
0011 for why this differs from Phase 6's pilot selection (which is
proportional to token share): Phase 8 needs comparably strong signal for
*not* calling a tool as for calling it correctly, and the frozen corpus's
own mixture is weighted toward direct arithmetic. Every candidate is
masked-tokenized and length-checked at selection time; oversized examples
are rejected (counted), never truncated.

## Loss masking

[`src/juniper_math/sft_rendering.py`](../src/juniper_math/sft_rendering.py)
renders each example into `(text, role)` segments and tokenizes each
segment independently, so the mask is exact by construction:

| Segment | Role | Label |
| --- | --- | --- |
| User prompt | context | `-100` |
| `<tool_call>{...}` | supervised | real token ids |
| `<tool_result>{...}` (from the real runtime) | context | `-100` |
| `<final>`/`<unsupported>`/`<error>` (terminal) | supervised | real token ids |
| BOS | context | `-100` |
| EOS | supervised | real token id (correct stop signal) |
| Padding | — | `-100` |

This reuses the exact tag conventions `dataset.shard.render_training_text`
already froze — it does not introduce a second wire format. 15 tests in
`tests/test_sft_rendering.py` cover every masking invariant (Sec. 11);
segment-wise tokenization was verified byte-identical to the existing
joint-string tokenization on 7,452 sampled examples before any training.

## Initialization

[`src/juniper_math/sft_pipeline.py`](../src/juniper_math/sft_pipeline.py)'s
`init_sft_state` loads the verified Phase 7 Base's **model weights only**
(via `checkpoint.load_checkpoint(..., optimizer=None, scheduler=None,
restore_rng=False)`), then builds a **fresh** AdamW + warmup/cosine
schedule via the same construction `trainer.init_state` uses for a
from-scratch run. Phase 8 never resumes Phase 7's optimizer/scheduler/RNG
trajectory. `sft_training_config.verify_parent_checkpoint` fails loudly
(before any training) if the on-disk Base checkpoint's SHA-256 does not
match the frozen `parent_checkpoint_sha256` the config declares.

## Training loop

Reuses `juniper_math.trainer` **completely unchanged** — the loop is
already generic over `-100`-masked labels, so assistant-focused masking
required zero trainer changes, only a new `Dataset`
(`sft_data.MaskedSftDataset`) that returns real masked labels. No packing
(one example per sequence, unlike Phase 6/7's first-fit packing) — see ADR
0011/`reports/PHASE8_PLAN.md` Sec. 4 for why packing two independently
masked trajectories into one causal sequence was judged an untested
boundary-safety risk not worth taking at SFT's modest scale.

## Evaluation

Every milestone runs: full masked-validation loss + per-category
breakdown; the four frozen Phase 4 v2 suites (regression check against the
Base, unchanged evaluation code); the new end-to-end tool-interaction suite
(`juniper_math.tool_interaction` + `juniper_math.sft_eval`, evaluated over
the held-out `evals/phase8_instruction_v1.json`); and fixed-seed greedy
generations. The tool-interaction harness is the first in this project to
actually *execute* a model-emitted tool call through the real `ToolRuntime`
and feed the runtime's own result back as context — never trusting
anything resembling a `<tool_result>` the model itself generated (verified
directly in `tests/test_tool_interaction.py`).

## Preflight

`config/phase8_preflight/` holds three bounded (300-step) candidates that
determined the frozen learning rate — see
`reports/PHASE8_PREFLIGHT.md`. Contrary to the initial expectation that
fine-tuning would need a much lower LR than Phase 7's from-scratch
pretraining rate, the evidence favored *keeping* Phase 7's 8e-4: it
improved tool-call validity and routing while two lower-LR candidates
regressed on the same metrics.

## Resume check

`config/phase8_sft_resume_check.yaml` bounds the Sec. 18 resume-equivalence
gate to 40 steps (interrupt at 20) against the real SFT pipeline — Phase
7's own resume proof does not establish that the new masked-dataset/
Base-initialization code path resumes correctly, so this is a dedicated
check, not an assumption. See `reports/PHASE8_RESULTS.md` for the result.
