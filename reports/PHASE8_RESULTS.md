# Phase 8 Results — Base vs. SFT Candidates

## Runs performed

1. **Preflight** (300 steps each, 3 candidates): `reports/PHASE8_PREFLIGHT.md`.
2. **Rejected full run v1** (4,500 steps, LR=8e-4, `run_id: phase8-sft-v1`):
   severe catastrophic forgetting found post hoc (unmasked validation loss
   0.606 → 3.6-4.0, a >5x increase). Preserved as evidence under
   `checkpoints/phase8-sft-v1-lr8e-4-rejected/`,
   `experiments/phase8-sft-v1-lr8e-4-rejected/`,
   `data/processed/phase8-sft-v1-lr8e-4-rejected/`. See
   `reports/PHASE8_REGRESSION.md`.
3. **Corrected full run v2** (4,500 steps, LR=2e-4, `run_id: phase8-sft-v2`,
   the config now frozen at `config/training_phase8_sft.yaml`): the run
   this report evaluates. Checkpoints: `checkpoints/phase8-sft/
   step_000900.pt`, `step_001800.pt`, `step_002700.pt`, `step_003600.pt`,
   `step_004500_final.pt`.

All runs start from the identical verified Phase 7 Base
(`checkpoints/phase7-full-v2/step_007483_final.pt`, sha256
`2e8098ab3a5db3c59a82fad19af2050154637fbe0628f3f6b0ca932d6cb6ea60`), the
identical seed (5,004,032), and the identical SFT dataset selection
(`juniper-math-sft-v1`, sft_identity
`1e55652407d3624a7e7c4d9d849ac6284fba4ce868a18cd59a3bf556a1a2d1b7`).

## Checkpoint-comparison table (n=200 phase8-instruction-v1 sample; n=185-200 per frozen v2 suite; identical decoding — greedy, `max_new_tokens=200`)

| Metric | Base | v2 step 2700 | v2 step 3600 | v2 step 4500 (final) |
| --- | --- | --- | --- | --- |
| Unmasked full-corpus validation loss (Sec. 22) | 0.6062 | 0.7437 (+22.7%) | 0.6881 (+13.5%) | 0.6818 (+12.5%) |
| Frozen `math` accuracy | 0.005 | 0.0 | 0.0 | 0.005 |
| Frozen `calibration` accuracy | 0.0 | 0.0 | 0.0 | 0.0 |
| Frozen `adversarial` accuracy | 0.1846 | **0.2103** | **0.2308** | 0.2154 |
| Frozen `tool_use_format.emitted_rate` | 0.6216 | **0.8595** | **0.8649** | **0.8595** |
| Frozen `tool_use_format.valid_rate` | **0.4865** | 0.3297 | 0.3297 | 0.3027 |
| Frozen `tool_use_format.tool_name_match_rate` | 0.2432 | **0.2757** | 0.2595 | 0.2541 |
| Phase 8 `correct_routing` | 0.735 | **0.765** | 0.720 | 0.685 |
| Phase 8 `call_parsed_valid` | 0.280 | 0.275 | 0.260 | 0.265 |
| Phase 8 `tool_name_correct` | 0.690 | **0.750** | 0.675 | 0.698 |
| Phase 8 `argument_execution_successful` | **0.732** | 0.509 | 0.673 | 0.623 |
| Phase 8 `end_to_end_success_on_tool_required` | 0.0 | 0.0 | 0.0 | 0.0 |
| Phase 8 `direct_answer_correct` | 0.0 | 0.0 | 0.0 | 0.0 |
| Phase 8 `unnecessary_tool_call` | 0.293 | **0.252** | 0.293 | 0.361 |
| Phase 8 `missing_required_call` | 0.189 | 0.189 | 0.245 | 0.189 |
| Phase 8 `fabricated_result_attempted` | 0.425 | **0.380** | 0.405 | 0.465 |
| Phase 8 `terminal_tag_correct` | **0.780** | 0.576 | 0.723 | 0.764 |

(**Bold** marks the best value in each row; ties are not bolded.)

## Selection

`step_002700.pt` (sha256
`41742e554acb6619df954b7425cebe44ed11ee1edceefb4905ae6025287d0361`) wins or
ties on more axes than the other two SFT candidates: best `correct_routing`,
best `tool_name_correct`, best `unnecessary_tool_call`, best
`fabricated_result_attempted`, and second-lowest regression among the three
tested (only step 3600/4500 have slightly lower unmasked-loss regression,
but they lose on nearly every tool metric). It is not chosen because it is
final (it is not — `step_004500_final.pt` is the actual last step) but on
this composite evidence, per Sec. 21's "not merely the final epoch" rule.

**Selected Phase 8 candidate**: `checkpoints/phase8-sft/step_002700.pt`,
sha256 `41742e554acb6619df954b7425cebe44ed11ee1edceefb4905ae6025287d0361`.

## Honest characterization of the result — this is a marginal, mixed outcome, not a clean win

At the smaller evaluation samples used during milestone/preflight
monitoring (n=40-100), the corrected run looked like it was making clear
progress. At the larger, more reliable n=200 sample computed for final
selection, several of those apparent gains **shrink into noise or invert**:

- `tool_use_format.valid_rate` (the frozen Phase 5-7 tool-format metric):
  the Base actually **outperforms every SFT candidate** (0.4865 vs.
  0.30-0.33). This is a real, measured regression on an established metric,
  not an improvement.
- `argument_execution_successful` (does the parsed call, once emitted,
  actually execute successfully): the Base (0.732) also outperforms every
  SFT candidate (0.509-0.673).
- `terminal_tag_correct`: the Base (0.780) outperforms step 2700 (0.576)
  and is comparable to step 3600/4500.
- The genuine, more-likely-real improvements are: `correct_routing`
  (+0.03 for step 2700, modest), `unnecessary_tool_call` (-0.041 for step
  2700), `fabricated_result_attempted` (-0.045 for step 2700), and frozen
  `adversarial` accuracy (+0.026 to +0.046). At n=147-200 these differences
  are on the order of 1-1.5 standard errors under a simple Bernoulli-rate
  model (`sqrt(p(1-p)/n)` ≈ 0.03-0.04) — directionally consistent with a
  real but small effect, not strong statistical evidence of a large one.
- `direct_answer_correct` and `end_to_end_success_on_tool_required` are
  **0.0 for every checkpoint including the Base** — Phase 8 did not confer
  the ability to actually complete a tool-mediated task end to end or
  answer directly with a numerically correct value, at this model scale
  and training budget. This is consistent with the Base's own near-zero
  math baseline (Phase 7: 1/215) — Phase 8 was never scoped to fix raw
  arithmetic capability (Sec. 35, explicit non-goal), but it means the
  "produce a correct final answer" half of the interaction loop remains
  unproven at any point in this project so far.
- The unmasked validation-loss regression (+22.7% at the selected
  checkpoint) exceeds the Sec. 8 pre-committed tolerance of ≤0.05 absolute
  nats. No candidate from either full run meets that tolerance exactly;
  the corrected run's candidates are the closest available and are three
  orders of magnitude better than the rejected run's.

## What Phase 8 did demonstrably achieve

- A complete, tested, working SFT pipeline: masked loss-rendering verified
  byte-for-byte against the existing joint tokenization on thousands of
  examples; a real end-to-end tool-execution/interaction harness that
  provably never trusts a model-fabricated `<tool_result>` (verified by a
  dedicated adversarial unit test); a resume-equivalence gate that passes
  with exact (0.0 diff) reproducibility.
- Discovery and correction of a real, severe failure mode (catastrophic
  forgetting at the naively-carried-over pretraining LR) that the original
  bounded preflight's metric selection was blind to — and a documented,
  evidence-based correction, not a cover-up.
- A modest, directionally-consistent (if not strongly statistically
  significant at this sample size) improvement in routing behavior and
  unnecessary-tool-call avoidance at the selected checkpoint, alongside a
  modest, real regression in argument-execution reliability and the
  established tool-format-validity metric.

## Verdict

Per Sec. 34's explicit instruction not to manufacture a success verdict:
**Phase 8 engineering is complete, but its resulting candidate does not
constitute a clean, confidently-demonstrated capability improvement over
the Phase 7 Base.** The honest result is a small, mixed, and
partially-noise-level shift in tool-interaction behavior, obtained at the
cost of a real (though much-reduced-from-the-first-attempt) general
capability regression. This should be read by GPT-5.6 Terra as a call for
either a substantially larger SFT budget/dataset, a different optimization
regime (e.g. a still-lower LR or more warmup), or accepting that a
5M-parameter model may be intrinsically too small to absorb this
instruction-tuning objective without trading off pretrained capability at
this data scale — not as a phase that is safe to wave through on its
reported numbers alone.
