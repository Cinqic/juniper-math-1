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
