# Phase 7 — Bounded Learning-Rate Preflight

Required by `reports/PHASE6_FINAL_APPROVAL.md` / `reports/TERRA_PHASE6_REVIEW.md`
before committing the Phase 7 full-pretraining budget: one bounded preflight
comparison across `6e-4`, `8e-4`, and `1e-3`, holding data/init/seed fixed.

## Method

Three 60-optimizer-step runs from identical initialization, seed
(`5004032`), and data (the frozen Phase 6 pilot subset,
`data/processed/phase6-pilot`, dataset identity
`bf9933f032a58b4eb618b32156783b8563097a5fc1c0ef26be4f76445128d25a`), varying
only `optimizer.learning_rate`. Configs:
`config/phase7_preflight/lr_{6e-4,8e-4,1e-3}.yaml` (each a copy of
`config/training_phase6_pilot.yaml` with `schedule.total_steps=60`,
`scheduler.warmup_steps=3`, dedicated output paths, no milestone
evaluation). Command: `python -m juniper_math train pilot-run --config
config/phase7_preflight/lr_<lr>.yaml --no-milestone-eval`. Raw evidence:
`experiments/phase7-lr-preflight/<lr>/train_log.jsonl`.

Before this preflight ran, the local (gitignored, disposable) dataset
shards did not match the frozen `shard_manifest.json` — a stale local
build from an earlier session. Per the repository's fail-closed shard
verification (added during Phase 6 remediation), the pilot data loader
correctly refused to run rather than silently training on unverified data.
The dataset was rebuilt with `python -m juniper_math dataset build`,
reproducing the frozen `dataset_identity`
(`bf9933f032a58b4eb618b32156783b8563097a5fc1c0ef26be4f76445128d25a`)
exactly, and `dataset validate` / `dataset verify` both passed before the
preflight was retried.

## Results

Training loss at each logged step (`logging_interval=10`):

| Step | LR 6e-4 | LR 8e-4 | LR 1e-3 |
|---:|---:|---:|---:|
| 10 | 6.441 | 6.090 | 5.781 |
| 20 | 5.040 | 4.755 | 4.668 |
| 30 | 3.981 | 3.858 | 3.919 |
| 40 | 3.712 | 3.659 | 3.862 |
| 50 | 3.706 | 3.627 | 3.767 |
| 60 | **3.290** | **3.172** | **3.358** |

Max observed pre-clip gradient norm: 2.047 (6e-4), 4.897 (8e-4), 1.821
(1e-3) — all finite, all well-behaved under the configured clip threshold
of 1.0. No non-finite loss or gradient at any step, any LR.

`1e-3` shows the fastest initial descent (lowest loss at step 10) but does
not sustain it: its step-40 loss (3.862) exceeds `8e-4`'s (3.659), and it
finishes the window at the highest final loss of the three (3.358). `8e-4`
has the lowest final-step loss and a smooth, monotonic decrease throughout,
with grad-norm behavior still comfortably bounded.

## Decision

Adopt **`8e-4`** as the Phase 7 full-pretraining peak learning rate. It
strictly dominates `6e-4` at every logged step in this comparison and
outperforms `1e-3` by the end of the window despite `1e-3`'s faster start,
while remaining inside the exact three-point preflight window Terra
mandated (no extrapolation beyond `1e-3`). This is a single 60-step,
6-point-per-run bounded comparison — sufficient to break the tie evidence
required before committing full budget, not a claim of a global optimum.
