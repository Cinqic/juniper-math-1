# ADR 0010: Checkpoint schema versioning and trust model

**Context.** Full training checkpoints must carry model weights, optimizer
state, scheduler/scaler state, RNG state across four generators, and
metadata (step, tokens seen, architecture identity, git commit) — plain
`model.state_dict()` cannot represent this. `torch.load` offers a
`weights_only=True` mode that restricts deserialization to plain
tensors/state-dicts for safety, but that mode cannot load the Python-native
objects (optimizer state dicts, RNG tuples, arbitrary training-config
dicts) a full checkpoint requires.

**Decision.** Checkpoints are a single `dict` written via `torch.save`,
loaded via `torch.load(..., weights_only=False)`, carrying an explicit
`schema_version` field (`checkpoint.CHECKPOINT_SCHEMA_VERSION`) and an
`architecture_identity` block (architecture version, vocab size, d_model,
n_layers, parameter target) checked against the currently loaded
architecture config before any state is restored. A schema or identity
mismatch raises `CheckpointError` rather than partially restoring state.
Saves are atomic: write to a temp file in the checkpoint's own directory,
then `os.replace` into place, so an interrupted save cannot corrupt a good
checkpoint.

Trust model: `weights_only=False` uses `pickle`, which can execute
arbitrary code when deserializing a maliciously crafted file. Checkpoints
are treated as trusted, project-generated artifacts only — this project
never loads a checkpoint from an untrusted or unverified source. See
`docs/CHECKPOINT_POLICY.md` and the `juniper_math.checkpoint` module
docstring.

**Consequences.** Loading an incompatible checkpoint (wrong architecture,
wrong schema version, corrupted file, or a file missing the
`schema_version` marker entirely) fails loudly with `CheckpointError`
rather than silently restoring a partially-wrong model. Any future schema
change must bump `CHECKPOINT_SCHEMA_VERSION` and document the migration
(or explicit non-migration) rather than reusing the same version number for
an incompatible layout.
