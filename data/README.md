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

The approved Phase 4 corpus is `juniper-math-dataset-v1`. Its shards remain
disposable and are deterministically reconstructible; the approved identity
and metadata are tracked under `data/processed/juniper-math-dataset-v1/`.
