# Manifests

Machine-readable provenance and integrity metadata, loaded/validated by
[`src/juniper_math/manifests.py`](../src/juniper_math/manifests.py).

## `manifests/sources.yaml`

Tracks provenance of external material (datasets, corpora, reference data)
as it is acquired. Phase 0 does not acquire the training corpus — this
establishes the schema ahead of Phase 4. Each entry:

`source_id`, `source_name`, `publisher`, `source_url`, `source_version`,
`acquisition_date`, `intended_use`, `license_id` (cross-references
`licenses.yaml`), `redistribution_status`
(`ALLOWED`/`NOT_ALLOWED`/`REQUIRES_REVIEW`/`NOT_APPLICABLE`),
`transformation_status`, `checksum`, `notes`.

## `manifests/licenses.yaml`

Tracks licensing for project code, dependencies, and (later) datasets and
third-party assets. Each entry: `license_id`, `scope`, `spdx_identifier`,
`reference`, `attribution_required`, `restrictions`,
`redistribution_status`, `notes`. Unknown licenses are recorded as
`UNKNOWN` or `REQUIRES_REVIEW` — never guessed.

## `manifests/artifacts.yaml`

Records SHA-256 hashes of frozen Phase 0 artifacts (architecture config,
project metadata, evaluation suite, other manifests) so a fresh clone can
verify byte-for-byte integrity via `python -m juniper_math hash verify`.
Hashes are generated from actual file bytes — never hand-typed.

## Regenerating artifact hashes

If a frozen artifact intentionally changes (with an accompanying version
bump and, where applicable, an ADR), regenerate its hash:

```bash
python -m juniper_math hash file <path>
```

and update the corresponding entry in `manifests/artifacts.yaml` by hand.
