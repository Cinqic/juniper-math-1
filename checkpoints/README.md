# checkpoints/

Checkpoint **metadata** only. See
[`docs/CHECKPOINT_POLICY.md`](../docs/CHECKPOINT_POLICY.md) for the full
policy. Real checkpoint binaries (`.pt`, `.pth`, `.safetensors`, `.ckpt`) are
excluded from Git by `.gitignore` and must be preserved through a versioned
remote artifact mechanism (Git LFS, GitHub Releases, or another
Cinqic-approved store) once they exist. Phase 0 produces no checkpoints.

## Phase 5 smoke checkpoints

Phase 5 (`python -m juniper_math train run`) produces real checkpoints
under `checkpoints/phase5-smoke/` (~60MB each, `.pt`, `.gitignore`d). These
are disposable, exactly-reproducible artifacts, not preserved through Git
LFS or a GitHub release: `config/training.yaml`'s `seed`, smoke-subset
selection, and schedule are committed and frozen, so re-running `train run`
against the same frozen dataset build reproduces an equivalent checkpoint
(the resume-equivalence gate in `reports/PHASE5_RESULTS.md` demonstrates
this reproducibility directly). What is preserved in Git is the checkpoint
*metadata* (`python -m juniper_math checkpoint inspect <path>` output),
recorded in `reports/PHASE5_RESULTS.md` and `experiments/phase5-smoke/`.

## Phase 6 pilot checkpoints

Phase 6 (`python -m juniper_math train pilot-run`) produces real
checkpoints under `checkpoints/phase6-pilot/` (`.pt`, `.gitignore`d), one
per milestone plus a final checkpoint. These are also treated as
**disposable, not archived** to Git LFS or a GitHub release, for two
independent reasons documented here per `docs/CHECKPOINT_POLICY.md` and
Sec. 29 of the Phase 6 instructions:

1. **Exact reproducibility.** `config/training_phase6_pilot.yaml`'s seed,
   category-stratified pilot-subset selection, and schedule are committed
   and frozen, so re-running `train pilot-run` against the same frozen
   dataset build reproduces an equivalent run — demonstrated directly by
   the pilot resume-comparison gate (`train pilot-resume-test`,
   `reports/PHASE6_RESULTS.md` §Resume verification).
2. **Phase 7 restart policy.** Per Sec. 28, the recommended default is for
   Phase 7 to begin from a clean, deliberately chosen initialization —
   not to silently continue from whatever pilot checkpoint happens to
   exist. Nothing about the frozen architecture, tokenizer, special
   tokens, sequence representation, or training objective changed during
   Phase 6 (see `reports/PHASE6_RESULTS.md` §Phase 7 restart rule), so
   there is no correctness reason a pilot checkpoint would need to survive
   into Phase 7 either.

What is preserved in Git is the checkpoint *metadata*
(`python -m juniper_math checkpoint inspect <path>` output) and every
checkpoint's SHA-256, recorded in `reports/PHASE6_RESULTS.md` and
`experiments/phase6-pilot/`.
