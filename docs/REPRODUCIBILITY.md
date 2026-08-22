# Reproducibility

Every significant future experiment must be able to record:

- Git commit
- Phase
- Experiment ID (see [`EXPERIMENT_NAMING.md`](EXPERIMENT_NAMING.md))
- Architecture configuration identity (`config/architecture.yaml` version)
- Tokenizer identity (Phase 2+)
- Dataset identity (Phase 4+, via `manifests/sources.yaml`)
- Evaluation suite identity (`evals/*.json` version + SHA-256)
- Seed (`juniper_math.seed.DEFAULT_PROJECT_SEED` or an override)
- Environment (`python -m juniper_math validate-env` output)
- Starting checkpoint (if resuming)
- Training parameters
- Output artifact identities (checkpoint hash, logs)

## What Phase 0 establishes toward this

- A single canonical seed helper (`juniper_math.seed`) instead of scattered
  seeding logic.
- Versioned, hashable configuration (`config/architecture.yaml`,
  `config/project.yaml`) and the tooling to verify hashes
  (`juniper_math.hashing`, `manifests/artifacts.yaml`).
- A versioned, hashed evaluation suite.
- An environment-reporting command that captures the facts needed to
  reproduce a run's environment.

## Determinism honesty

Seeding Python/NumPy/PyTorch RNGs gives reproducible results for CPU-bound,
non-parallel-reduction operations. `torch.use_deterministic_algorithms`
forces deterministic kernels where PyTorch provides them. It does **not**
guarantee bitwise-identical results across different GPUs, driver versions,
or CUDA library versions — floating-point reduction order can still differ.
See [`src/juniper_math/seed.py`](../src/juniper_math/seed.py) for the exact
scope of what is and isn't covered.
