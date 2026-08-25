# Terra Phase 8 Independent Review

## Verdict

**NOT APPROVED.** Phase 8 remains blocked at Phase 7; Phase 9 is not
authorized. The reviewed source started at `6c6135dd40bba7d43c46b651aec13630ca7450e1`.
The original Sonnet candidate tag `phase-8-math-sft-candidate` resolves to
`a0c4ec4f1b937be912af66f37998f19552e18e84` and is preserved unchanged.

## Independent findings

- The GitHub candidate release asset was freshly downloaded and matched
  `41742e554acb6619df954b7425cebe44ed11ee1edceefb4905ae6025287d0361`.
- The independently reproduced unmasked Base-regression values were Base
  0.606183, historical selected candidate 0.743720 (+0.137536), and the
  historical 300-step 2e-4 checkpoint 0.628191 (+0.022008).
- The historical candidate violates the pre-committed +0.05-nat gate and
  has 0/115 direct-answer and 0/53 end-to-end held-out success at n=200.
- `sft_rendering.py` trained EOS directly after context-only trusted results
  for all 1,553 source `tool_error` examples. This prevented supervision of
  the required response after an execution failure.
- Base-regression loss was absent from normal milestone evaluation.
- The v1 interaction evaluator conditioned tool-name correctness on fully
  parsed calls and counted arbitrary runtime success as argument execution;
  evaluator v2 now reports exact expected-argument matches over all required
  tool cases. It also accepts valid currency, commas, fractions, scientific
  notation, units, and explanatory final-answer text.
- `phase4_calibration_v2` has 30 duplicate IDs across 130 distinct cases.
  They do not affect the list-based current scorer, but v2 is historically
  frozen and must not be used as a unique-ID keyed metric without a separately
  versioned correction.
- The claimed SFT identity `1e556…` is the train-split ID hash, not the
  combined manifest identity. The reproducible combined identity is
  `1fbcaf6afe623529badf2c2e2fd7faf5e541928e239359152b70ba2973681f1e`.

## Remediation and result

The versioned `juniper-math-sft-v2` derived representation adds a supervised
`<error>CODE: message` completion derived only from each recorded trusted
runtime error; no frozen Phase 4 artifact changed. A 900-step, 5e-5 run from
the verified Phase 7 Base recorded unmasked regression at every milestone.
The best (300-step) checkpoint has SHA-256
`012d00c42dd044fab0d0bdc3f84dab4cd3bc3eb09351ce0070e41f1866ae562d` and +0.006737
nats regression. It achieved 9/53 tool-required end-to-end successes, but
0/115 direct-answer correctness on the corrected n=200 held-out evaluation.

It is remotely preserved only as the clearly labeled prerelease
`phase-8-terra-remediation-rejected`; a fresh asset download re-hashed to the
same SHA-256. This is non-release research evidence, not an approval tag.

That fails the pre-registered actual-task-completion gate. The remediation
checkpoint, historical 8e-4 run, and original candidate are preserved as
non-release research evidence. No final Phase 8 tag, release, or recovery
approval has been created because those steps require an approvable checkpoint.
