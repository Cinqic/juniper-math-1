# Juniper Math 1 — Final Research Conclusion

## Status

**Research complete.** Juniper Math 1 is a completed research project, not a released Phase 8 model. Phase 7 remains the last approved model-development phase. Its approved Base checkpoint is `checkpoints/phase7-full-v2/step_007483_final.pt` (SHA-256 `2e8098ab3a5db3c59a82fad19af2050154637fbe0628f3f6b0ca932d6cb6ea60`), preserved by tag `phase-7-pretraining`.

No Phase 8 checkpoint is approved. Phase 8 is **research-conclusive, but not approved as a final capability checkpoint**. Phases 9 and 10 are retired for Juniper Math 1.

## Purpose and answer

Juniper Math 1 asked how capable, reliable, efficient, and truthfully calibrated an approximately five-million-parameter purpose-built mathematical language model could become at natural-language mathematics, deterministic tool selection and operation, verification, and truthful handling of ambiguity or unsupported requests.

The project froze a 5,004,032-parameter decoder-only Transformer, a math-specialized tokenizer, a 62.4M-token synthetic corpus, held-out evaluation suites, and a deterministic calculator runtime. Phases 0–7 established and approved the engineering foundation and Base pretraining. Phase 8 tested whether instruction and tool adaptation could produce the required joint capability.

Juniper Math 1 demonstrates that this approximately 5M-parameter Transformer can efficiently acquire meaningful specialized behaviors: mathematical response patterns, structured control tokens, partial tool interaction, calibrated refusals, adversarial/error handling, and limited held-out generalization. Under the frozen architecture and tested training regimes, however, these behaviors could not be made simultaneously robust, generalizable, reliable, and well retained. Stronger instruction learning repeatedly interfered with previously learned behavior; stronger retention constraints limited acquisition of new generalized capability. No Phase 8 checkpoint satisfied the complete acceptance criteria.

This is a finding about Juniper Math 1's architecture, objectives, data, and experimental regime. It is **not** a universal scaling law for every five-million-parameter neural architecture.

## Evidence and central trade-off

The final conclusion preferentially uses the later corrected Phase 8 evaluation surface. Earlier reports use 160 direct / 67 tool-required cases; the Terra remediation report uses the corrected 115 direct / 53 tool-required subsets at generation budget 200, with exact-argument semantics and numeric-parser fixes. Historical values remain preserved and are not directly interchangeable with the corrected evaluation.

| Experiment | Main intervention | Direct | Tool E2E | Base retention | Adversarial | Important result | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original SFT v2 | Full SFT, LR 2e-4 | 0/115 | 0/53 | +0.137536 nats | historical mixed result | Candidate failed both actual-task gates | Rejected |
| Terra remediation v3 | Corrected post-tool-error supervision, LR 5e-5 | 0/115 | 9/53 | +0.006737 nats | tool-error completion improved | Strong Base preservation; no direct generalization | Rejected |
| Diversified recovery v4 | Broader recovery mixture | 1/160 | 10/67 | within gate | n/a | First nonzero direct/tool preflight | Preflight only |
| Independent direct curriculum v6 | Added 12,000 independent direct records | 0/160 | 0/67 | +0.002265 nats | n/a | Fabrication 48/271 while losing gains | Rejected |
| Higher-LR curriculum v8 | Balanced curriculum at 2e-4 | 5/160 | 4/67 | +0.118104 nats | 1/195 | Capability gains with destructive regression | Rejected |
| Frozen-Base replay v9 | Replay-balanced adaptation | 12/160 | 4/67 | +0.051328 nats | 3/195 | Best preflight direct result; misses retention and safety gates | Rejected |
| Safety replay v11 | Safety data plus replay | 8/160 | 6/67 | +0.017621 nats | 120/195 | Strong direct/safety point; tool selection worsened | Rejected |
| Explicit runtime-tool v12 | Executed tool trajectories | 3/160 | 8/67 | +0.029491 nats | 55/195 | E2E increased; no Base tool-metric recovery | Rejected |
| Partial-layer v14 | Last two layers trainable | 0/160 | 6/67 | +0.008618 nats | 42/195 | Preserved Base but limited direct learning | Rejected |
| Staged tool-call v16 | Decomposed call construction + replay | 7/160 | 10/67 | +0.019456 nats | 72/195 | Parsing 40/67, tool selection 25/67 below Base 51/67, 36/67 | Rejected |

The recurring pattern matters more than any single score. The recovery full run stayed within the +0.05-nat Base ceiling and reached 6/67 end-to-end, but remained 0/160 direct. Safety replay reached 16/215 math, 27/130 calibration, and 120/195 adversarial correctness at step 400, while valid calls and correct-tool selection stayed below Base. Staged tool-call supervision reduced fabricated-result attempts from 110/271 at Base to 0/271 at step 200 and 7/271 at step 600, while failing required-call gates.

These are genuine learned behaviors, not an untrained model. They are also not jointly sufficient capability. The evidence is consistent with severe capacity/interference constraints: component skills were easier to learn than to retain together. It does not establish a definitive mechanism or exclude data diversity and objective design as contributors.

## Research checkpoints, not releases

- **Base preservation / corrected tool-error completion:** remediation v3, step 300, SHA-256 `012d00c42dd044fab0d0bdc3f84dab4cd3bc3eb09351ce0070e41f1866ae562d`; +0.006737 nats, 9/53 tool end-to-end, 0/115 direct.
- **Direct/safety balance:** safety-replay v11, step 400, SHA-256 `b0728c539006e5736cffd53e49a56c5ba20df8304f294bd857816d68771b51a2`; 8/160 direct, 120/195 adversarial, +0.017621 nats.
- **Fabrication resistance / final bounded result:** staged-call v16, step 600, SHA-256 `03896f06f0e89a9096a9e948c6a4e3728c440814649cdbc9b4c7cc5edc9502bc`; 7/160 direct, 10/67 tool E2E, 7/271 fabricated-result attempts.

None is an approved Phase 8 production or release checkpoint.

## Why development stops here

The project corrected material pipeline defects and then tested low and high learning rates, replay, curriculum balance, safety supervision, explicit runtime-tool supervision, partial-layer adaptation, staged call supervision, and bounded milestones. It repaired CI, enforced source-tree reproducibility, corrected SFT representation hashing and runtime formatting, and improved evaluator semantics. These materially different interventions repeatedly exposed trade-offs rather than a path through all acceptance gates. Further iteration on the frozen architecture would have diminishing research value. This is a useful stopping criterion, not giving up: the primary question has been answered.

## Future work

The evidence supports future investigation of greater representational capacity, pretraining/SFT co-design, more semantically diverse data, downstream-aware curricula, decomposed tool continuation, continual-learning strategies, modular specialization, and stronger held-out linguistic evaluation. The next effort must be a new Cinqic language-model research project, not Juniper Math 1 Phase 9. No future architecture, model size, launch date, or capability is promised here.

Development on Juniper Math 1 is complete. Its evidence is preserved in the historical reports and indexed experiments; its lessons will inform a future Cinqic language-model project.

## Primary evidence

- [Phase 7 final approval](PHASE7_FINAL_APPROVAL.md)
- [Original Phase 8 results](PHASE8_RESULTS.md)
- [Terra Phase 8 review](TERRA_PHASE8_REVIEW.md)
- [Phase 8 remediation](PHASE8_REMEDIATION.md)
- [V3 full recovery](PHASE8_RESULTS_V3_FULL.md)
- [V4 safety-replay milestones](PHASE8_RESULTS_V4_MILESTONES.md)
- [V5 staged tool-call milestones](PHASE8_RESULTS_V5_CALL_STAGE.md)
- [Experiment index](../experiments/README.md)
