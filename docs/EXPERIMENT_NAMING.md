# Experiment Naming Convention

Avoid vague, undecodable directory names like `test2_final_REAL_fixed`.

## Format

```
<phase>-<yyyymmdd>-<short-slug>
```

Example: `p1-20260901-baseline-forward-pass`.

The slug describes *what the experiment was*, not a version counter. Every
experiment directory under `experiments/` must contain a small metadata file
(e.g. `experiments/<experiment-id>/experiment.yaml`) recording at minimum:
experiment ID, phase, git commit, architecture/tokenizer/dataset/eval
identities, seed, and a one-paragraph description of intent and outcome —
so a human never has to reverse-engineer intent from the directory name
alone. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the full field
list expected once training experiments begin.

Phase 0 does not create real experiments; this convention is documented so
Phase 1+ work starts with it in place.
