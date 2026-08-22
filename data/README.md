# data/

Data workspace. Subdirectories (`raw/`, `interim/`, `processed/`,
`external/`, `cache/`) hold disposable, regenerable content and are
excluded from Git via `.gitignore` — only this README and future
`README.md`/provenance files under each subdirectory are tracked.

Provenance for anything placed here belongs in
[`manifests/sources.yaml`](../manifests/sources.yaml), not in this
directory itself.

- `raw/` — unmodified acquired data.
- `interim/` — intermediate transformation output.
- `processed/` — final, model-ready data.
- `external/` — third-party reference data not part of the training corpus.
- `cache/` — disposable computation caches.

No training corpus exists yet — full dataset construction is Phase 4 work.
