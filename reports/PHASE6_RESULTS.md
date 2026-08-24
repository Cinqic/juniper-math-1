# Phase 6 Pilot-Pretraining Results

This report supersedes the engineering-candidate result record. During the
independent review, the candidate's ignored local shards were found not to
match the committed frozen shard manifest. The original 137,057-example
result is therefore invalid and must not be used for Phase 7 planning.

## Independently reproduced run

- Source commit: `21ad9f781f972e9802c06867d2a25e2cbff93867`
- Seed: `5004032`; RTX 2060; Python 3.12.3; PyTorch 2.13.0+cu130; CUDA 13.0
- Frozen dataset identity: `bf9933f032a58b4eb618b32156783b8563097a5fc1c0ef26be4f76445128d25a`
- Command: `python -m juniper_math train pilot-run`
- Full structured evidence: `experiments/phase6-pilot/train_log.jsonl`

Before selection, the remediation verifies every ignored JSONL shard's size
and SHA-256 against `shard_manifest.json`; it then verifies the resulting
whole-dataset identity. This run therefore uses the frozen corpus rather
than local-only data.

| Split | Examples | Raw tokens | Packed sequences | Padding |
|---|---:|---:|---:|---:|
| train | 130,492 | 4,997,273 | 5,324 | 3.55% |
| validation | 3,051 | 120,152 | unpacked | n/a |

Both independently generated manifests had SHA-256
`6e335ab5f342eb906f7e26b227a625326e65e27a3bf6a70d761464498151a524`.
All 24 categories are present. The full frozen split contains 1,393
`tool_error` records in train, 72 in validation, and 88 in test; scarcity is
not a frozen-dataset characteristic and does not constrain Phase 7.

## Training and evaluation

FP32 packed training used 1,024-token sequences, micro-batch 4, gradient
accumulation 4, AdamW (peak LR `6e-4`), 16 warmup steps, cosine decay, and
320 optimizer steps. Loss-bearing tokens seen: 5,051,139.

| Step | Validation loss |
|---:|---:|
| 0 | 8.3801 |
| 80 | 2.1327 |
| 160 | 1.1824 |
| 240 | 1.0304 |
| 320 | 0.9777 |

The trajectory is finite and stable. The final category losses remain much
higher for no-answer/calibration behaviors than for templated numeric and
tool patterns; this is format/category learning, not evidence of math skill.

All 725 frozen v2 cases ran at each of five milestones. Math accuracy and
valid tool-call rate were 0% at every milestone. Calibration was 1/130 at
steps 240 and 320; adversarial accuracy was 0%. These isolated outcomes are
noise, not emerging capability.

## Runtime and recovery

The reproduced run took 406.75 seconds total, with 904.5 MiB peak allocated
CUDA memory. Training steps averaged about 0.37 seconds; evaluation accounts
for most non-training wall time. The final checkpoint is 60,123,779 bytes
with SHA-256 `a2061336c270ce325f8444e0e20edb17103c62bb166b294da4cfdd26aee924bc`.
It contains model, optimizer, scheduler, RNG, step, token count, data cursor,
architecture identity, seed, and full configuration. It is reproducible and
is not a Phase 7 starting artifact.

## Resume result

The independent 160-step interruption test reached the same step and token
count on both paths. It recorded maximum parameter difference `0.005875` and
maximum loss-history difference `0.003059`, both under the predeclared
`1e-2` numerical threshold. Fixed greedy generations differed, which is
consistent with PyTorch's warning that the CUDA memory-efficient attention
backward kernel is nondeterministic. Phase 7 may use the resume mechanism,
but must retain tolerance-based checks and must not describe CUDA resume as
bitwise or generation-identical.

## Phase 7 recommendation

Start from fresh random initialization; retain frozen architecture/tokenizer/
dataset and 1,024-token packing; use FP32; use AdamW with beta1 0.9, beta2
0.95, epsilon `1e-8`, weight decay 0.01, clipping 1.0; and begin with a
ratio-based 5% warmup plus cosine decay. `6e-4` is an acceptable conservative
starting LR, but `1e-3` was faster in the candidate's short screen and should
receive one bounded preflight comparison before a serious run. Batch size and
token budget are deferred to that preflight; do not continue from this pilot.
