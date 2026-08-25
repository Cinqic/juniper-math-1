# Phase 8 SFT Dataset V3

## Status

`juniper-math-sft-v3` is a new, derived training representation. It does not
modify the frozen `juniper-math-dataset-v1` corpus, tokenizer, architecture,
tool protocol, evaluations, or any historical Phase 8 artifact. It has not
yet produced an approval candidate.

## Lineage and identities

The parent corpus was freshly reconstructed with identity
`bf9933f032a58b4eb618b32156783b8563097a5fc1c0ef26be4f76445128d25a`.

The deterministic SFT parent selection remains:

- selection identity: `1fbcaf6afe623529badf2c2e2fd7faf5e541928e239359152b70ba2973681f1e`;
- train parent-ID hash: `1e55652407d3624a7e7c4d9d849ac6284fba4ce868a18cd59a3bf556a1a2d1b7`.

The effective rendered representation is distinct:

- representation identity: `24502369b203ec3c7a17d57cb64bb358aa4cdcdaced58624662f527505a0ed47`;
- train token/label hash: `09398a410720a4d6c9b69fe21892951677831addf4b6ab3bfc3be558d983fd24`.

Representation hashing includes the selected parent identities, tokenizer,
renderer schema, maximum sequence length, exact token IDs, and supervision
labels. A change to masking or a terminal completion therefore changes the
representation identity even if the parent selection remains constant.

## Composition

The base selection has 24,000 train records (1,000 per frozen category). For
every concrete direct-answer parent record, one deterministic derived record
adds an independently selected instructional frame such as a homework check,
a student question, or an imperative mathematical request. The v3 train split
therefore contains **39,000** records. Each derived record has a new ID,
versioned generator/family/template lineage, and unchanged answer, split,
verification expression, and tool-trace policy.

Tool-required trajectories, semantic/error trajectories, and trusted host
results are never copied into a model-authored result. The assistant-only loss
mask remains unchanged: prompts and host results are context-only; calls and
assistant terminal completions are supervised.

## Efficiency

V3 stores independent variable-length trajectories and forms deterministic
length-bucketed batches with per-batch right padding. It does not concatenate
conversations. The pre-augmentation measurement was 4.64% padding at
micro-batch size 8; the final v3 value must be recorded from the exact
preflight run before a candidate is selected.

## Limitation and next experiment

The instructional frames are a targeted diversity intervention, not grounds to
claim semantic generalization. The recovery experiment must still demonstrate
nonzero, material held-out direct correctness and end-to-end tool completion,
within the preregistered Base-regression gate, before approval is possible.
