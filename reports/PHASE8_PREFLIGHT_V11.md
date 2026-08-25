# Phase 8 V11 Last-Two-Block Adaptation Preflight

## Verdict

**Rejected.** Restricting adaptation to the final two transformer blocks
strongly preserves Base loss, but it eliminates held-out direct correctness
and does not restore required-tool parsing or correct-tool selection to the
frozen Base level.

## Provenance

- clean source commit: `dad3e62`;
- run ID: `phase8-sft-v14-last2-safety-tool-preflight`;
- 300 steps at `2.0e-4`; 1,582,336 trainable parameters (last two blocks plus
  final normalization), with embeddings and the first three blocks frozen;
- same V10 safety/tool/replay corpus;
- checkpoint SHA-256:
  `8eab6d23e8509d99202d0137ad2955f557929ef7d17ef9179bcd00312bca0edd`.

## Results

| Metric | Frozen Phase 7 | V11 300-step |
| --- | ---: | ---: |
| Base validation loss | 0.606183 | 0.614801 |
| Base loss delta | — | +0.008618 |
| Math correctness | 1/215 | 0/215 |
| Direct held-out correctness | 0/160 | 0/160 |
| Required tool calls parsed | 51/67 | 36/67 |
| Correct tool | 36/67 | 21/67 |
| Tool end-to-end success | 0/67 | 6/67 |
| Adversarial correctness | 36/195 | 42/195 |

The result supports a narrow conclusion: frozen lower representations alone
cannot support the required direct instruction adaptation at this scope. It
does not justify a checkpoint selection, release, or Phase 9 authorization.
