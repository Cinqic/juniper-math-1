# checkpoints/

Checkpoint **metadata** only. See
[`docs/CHECKPOINT_POLICY.md`](../docs/CHECKPOINT_POLICY.md) for the full
policy. Real checkpoint binaries (`.pt`, `.pth`, `.safetensors`, `.ckpt`) are
excluded from Git by `.gitignore` and must be preserved through a versioned
remote artifact mechanism (Git LFS, GitHub Releases, or another
Cinqic-approved store) once they exist. Phase 0 produces no checkpoints.
