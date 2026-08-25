# Phase 8 Remediation Plan

## Pre-registered acceptance gates

This bounded remediation starts from the approved Phase 7 Base and uses the
same frozen tokenizer, architecture, tool protocol, parent corpus, held-out
cases, and deterministic seed as the historical Phase 8 run.  It is not
eligible for approval unless all gates below hold under the same evaluator:

- Unmasked fixed-stride Base-validation loss delta is at most +0.050 nats.
- No material regression occurs on the frozen Phase 4 suites.
- Tool routing, valid-call, tool-name, exact-argument, or fabrication metrics
  show a meaningful improvement with raw numerators and denominators.
- Direct-answer correctness and end-to-end tool-required success are both
  nonzero on the held-out Phase 8 suite.
- The frozen tool trust boundary and all relevant tests remain intact.

## Intervention

The historical renderer trained EOS directly after every context-only trusted
tool error.  The versioned `juniper-math-sft-v2` derived representation instead
adds a supervised `<error>CODE: message` response derived solely from the
recorded runtime result.  The candidate uses LR `5e-5`, 900 steps, and
milestones at 0/300/600/900.  Every milestone now records the unmasked
fixed-stride Base-validation loss in addition to masked SFT validation.

The existing `phase8-sft-v2` run and candidate release remain historical,
rejected evidence and are not overwritten.

## Result: rejected

The 900-step remediation preserved Base capability on the fixed 2,000-example
unmasked validation sample: 0.610917 versus the independently reproduced Base
0.606183 (+0.004733 nats).  The selected 300-step state had 0.612920
(+0.006737 nats).  At the corrected evaluator's n=200 held-out suite, each
remediation milestone produced 9/53 end-to-end successes, all from useful
tool-error handling, but **0/115 direct-answer correctness**.  The Phase 8
acceptance gate requires both direct-answer and tool-required end-to-end
success to be nonzero, so no remediation checkpoint is approvable.

The best retained remediation checkpoint for further research is
`checkpoints/phase8-sft-v3-remediated/step_000300.pt`:

- SHA-256: `012d00c42dd044fab0d0bdc3f84dab4cd3bc3eb09351ce0070e41f1866ae562d`
- Base regression: +0.006737 nats
- exact expected arguments: 9/53
- end-to-end tool-required success: 9/53
- direct-answer correctness: 0/115

It is rejected and non-release.  The phase remains unapproved.
