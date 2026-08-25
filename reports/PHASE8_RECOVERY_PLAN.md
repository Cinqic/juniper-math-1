# Phase 8 Recovery Plan

## Status

**PLANNED — NOT AN APPROVAL CANDIDATE.** This plan supersedes neither the
historical Phase 8 reports nor the rejected `phase-8-terra-remediation-rejected`
evidence. Phase 7 remains the approved foundation.

## Reproduced starting evidence

- The Phase 7 release asset `step_007483_final.pt` hashes to
  `2e8098ab3a5db3c59a82fad19af2050154637fbe0628f3f6b0ca932d6cb6ea60`.
- The existing remediation result preserved Base loss but reached 0/115
  direct held-out correctness; it is therefore not approvable.
- A clean clone previously failed its unit suite because a unit test expected
  the release-only parent checkpoint to be present locally.

## Recovery design

The next derived representation is `juniper-math-sft-v3`. It has two distinct
identities:

- `selection_identity` (`sft_identity`): selected parent example IDs.
- `sft_representation_identity`: parent identity, selection identity, tokenizer,
  renderer schema, maximum length, and per-split hashes of exact token IDs and
  supervision labels.

The new representation identity makes an otherwise invisible masking or renderer
change auditable. Training candidates must now start from a clean committed tree;
dirty-tree work belongs to a separately labeled development workflow and cannot
be used for approval.

## Pre-registered next experiment

No full training has been run from this plan. Before one is launched, create a
versioned corpus with substantially broader direct-answer wording and structural
families, then record its mixture by *supervised token count*. Keep tool routing,
call construction, trusted-result continuation, and trusted-error continuation
as distinct example types. The frozen parent corpus, tokenizer, architecture,
tool protocol, Phase 7 release, and historical Phase 8 evidence must not change.

Use a small bounded ablation around the evidence-supported `5e-5` learning rate
before a full run. Every milestone must report masked SFT validation, unmasked
Base validation, direct held-out correctness, tool routing/call metrics, trusted
result completion, and end-to-end task success with raw numerators and
denominators.

## Approval gates

Select a checkpoint only if all of the following were defined before its final
evaluation and pass without changing frozen suites:

- clean committed and pushed training source; verified parent and candidate hashes;
- full CI and security suites green;
- unmasked Base validation regression at most +0.050 nats;
- materially nonzero direct held-out correctness, not a single lucky case;
- meaningful tool-routing and end-to-end improvement over Phase 7 Base;
- no model-authored trusted tool results, no security regression, and no
  unacceptable tool-call loops;
- fresh-clone recovery, artifact reconstruction, checkpoint download, and hash
  verification all succeed.

Until then, Phase 8 remains unapproved and Phase 9 remains unauthorized.
