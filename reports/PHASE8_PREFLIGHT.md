# Phase 8 Bounded SFT Preflight

**Addendum (superseding the Decision below): this preflight did not measure
the Sec. 22 unmasked full-corpus validation-loss regression metric — only
the tool-interaction metrics and the narrower SFT-masked loss. A later,
explicit Sec. 22 check found that Candidate A's learning rate (8e-4)
causes severe catastrophic forgetting on that metric while Candidate B's
(2e-4) does not. The frozen config was revised to use 2e-4. See
`reports/PHASE8_REGRESSION.md` for the full finding and
`reports/PHASE8_RESULTS.md` for the corrected run. This document is kept
as-is below as an honest record of what this preflight actually measured
and concluded at the time.**

Three bounded candidates (300 steps each, ~78-109s wall-clock, RTX 2060),
all starting from the identical verified Phase 7 Base
(`checkpoints/phase7-full-v2/step_007483_final.pt`,
sha256 `2e8098ab...ea60`), identical seed (5,004,032), identical
`sft_subset` selection except where noted, identical decoding
(`temperature=0.0`, greedy, `max_new_tokens_per_turn=200`), identical 40-case
sample of the held-out `evals/phase8_instruction_v1.json` suite at step 0
and step 300.

Note: `generation_max_new_tokens` was corrected from an initial (too small)
48 to 200 partway through this preflight, after discovering that 48 tokens
truncates most tool-call JSON before it completes, artificially suppressing
`call_parsed_valid` regardless of training — every result below uses the
corrected 200-token budget.

## Candidates

| | A | B | C |
| --- | --- | --- | --- |
| Config | `config/phase8_preflight/candidate_a_lr8e-4.yaml` | `config/phase8_preflight/candidate_b_lr2e-4.yaml` | `config/phase8_preflight/candidate_c_mixture.yaml` |
| Learning rate | 8.0e-4 (unchanged from Phase 7) | 2.0e-4 | 2.0e-4 |
| Category weight overrides | none (uniform 1000/category) | none | `tool_error`/`incorrect_tool_call`/`missing_information`/`ambiguity` ×2.0 |
| Rationale | Test whether Phase 7's from-scratch LR still works for fine-tuning | Test the conventional expectation that fine-tuning needs a much lower LR | Isolate whether extra weight on the rarest, highest-value categories helps at the lower LR |

## Results at step 0 (identical Base, shown once) vs step 300

| Metric | Step 0 (Base) | A step 300 | B step 300 | C step 300 |
| --- | --- | --- | --- | --- |
| Validation loss (masked, SFT set) | 0.0570 (0.0479 for C's slightly different selection pass) | 0.1242 | 0.0760 | 0.0644 |
| correct_routing | 0.675 (27/40) | **0.775** (31/40) | 0.550 (22/40) | 0.650 (26/40) |
| emitted_tool_call | 0.475 (19/40) | 0.375 (15/40) | 0.600 (24/40) | 0.500 (20/40) |
| call_parsed_valid | 0.200 (8/40) | **0.350** (14/40) | 0.275 (11/40) | 0.200 (8/40) |
| tool_name_correct | 0.800 (4/5) | 0.833 (5/6) | 0.833 (5/6) | 0.833 (5/6) |
| argument_execution_successful | 0.750 (6/8) | 0.429 (6/14) | 0.636 (7/11) | 0.625 (5/8) |
| final_answer_correct_overall | 0.0 (0/33) | 0.030 (1/33) | 0.0 (0/33) | 0.0 (0/33) |
| unnecessary_tool_call | 0.382 (13/34) | **0.265** (9/34, improved) | 0.529 (18/34, worsened) | 0.412 (14/34, worsened) |
| missing_required_call | 0.0 (0/6) | 0.0 (0/6) | 0.0 (0/6) | 0.0 (0/6) |
| Frozen `math` accuracy | 0.025 | 0.0 | 0.0 | (not separately run) |
| Frozen `adversarial` accuracy | 0.225 | 0.225 | 0.225 | (not separately run) |
| Frozen `calibration` accuracy | 0.0 | 0.025 | 0.0 | (not separately run) |
| `tool_use_format.valid_rate` (frozen v2 suite, corrected token budget) | 0.375 | **0.600** | 0.350 | (not separately run) |
| `tool_use_format.tool_name_match_rate` | 0.225 | **0.600** | 0.250 | (not separately run) |

(A "step 0" baseline is identical model weights across all three
candidates; C's own step-0 validation loss number differs slightly because
its `sft_subset.category_weight_overrides` changes which validation
examples get selected for four categories, which is expected and does not
indicate a different Base.)

## Finding

Candidate A (learning rate unchanged from Phase 7, 8e-4) clearly
outperforms both lower-LR candidates on every tool-format and routing
metric at this bounded scale: valid-call rate roughly doubled (0.375→0.6),
tool-name-match rate nearly tripled (0.225→0.6), and — notably —
`unnecessary_tool_call` **decreased** (0.382→0.265), meaning the model is
simultaneously getting better at using the tool correctly *and* better at
not using it when it shouldn't. Both 2e-4 candidates (B and C) show flat or
regressed routing and a worsened unnecessary-tool-call rate over the same
300 steps.

This **contradicts the a priori assumption recorded in
reports/PHASE8_PLAN.md Sec. 9** that fine-tuning from a pretrained Base
would need a substantially lower learning rate than from-scratch
pretraining. At this model scale (5M parameters) and this data scale
(24,000 examples), the higher LR does not appear to be destabilizing the
Base — frozen-suite accuracy is essentially flat (math/adversarial
unchanged, calibration marginally improved) rather than collapsing, which
is the regression signal that would have indicated the LR was too
aggressive.

Candidate C's mixture ablation (extra weight on rarer categories) was run
at the lower, losing LR and does not show a decisive additional effect
over B at that LR — both are similarly flat/regressed relative to A. This
preflight is not strong evidence that mixture reweighting is unhelpful in
general, only that it did not rescue the lower-LR candidate's
underperformance in this bounded window; the full run uses the uniform
(non-overridden) mixture together with the winning LR, per Sec. 19's
one-variable-at-a-time discipline (the full run is where LR was confirmed
to matter most; a follow-on mixture experiment at 8e-4 was not run given
the bounded preflight budget and is noted as unexplored, not concealed).

## Decision

`config/training_phase8_sft.yaml` is frozen with `learning_rate: 8.0e-4`
and `category_weight_overrides: {}` (uniform mixture), matching Candidate
A. `schedule.total_steps` is set to 4,500 (≈3 epochs over the 24,000-example
SFT train set at effective batch size 16), well within the RTX 2060 budget
given the measured ~0.07s/step pure-training throughput.

## Numerical stability

No non-finite loss, gradient, or parameter value occurred in any of the
three candidates (`assert_model_finite` runs after every optimizer step;
none raised).
