# Checkpoint Policy

## Storage

Real model checkpoints (`.pt`/`.pth`/`.safetensors`/`.ckpt`) must **never**
be committed directly into Git history — `.gitignore` blocks these
extensions under `checkpoints/`. Once training exists (Phase 1+), important
checkpoints must be preserved through a versioned remote mechanism:

- Git LFS, or
- GitHub Release assets, or
- another Cinqic-approved versioned artifact store.

A checkpoint that exists only on local disk after a phase is officially
complete is not considered preserved, per the project's canonical-storage
principle.

## No fake checkpoints

Phase 0 does not produce checkpoints. Do not commit placeholder/fake large
binaries merely to populate `checkpoints/` — the directory currently holds
only this policy documentation (and, going forward, real checkpoint
metadata).

## Checkpoint metadata

Each real checkpoint must eventually have an accompanying metadata record
(JSON/YAML, tracked in Git even though the binary itself is not) containing:

`checkpoint_id`, `model_version`, `git_commit`, `architecture_version`,
`tokenizer_id`, `dataset_id`, `step`, `tokens_seen`, `training_config`,
`sha256`, `storage_location`.
