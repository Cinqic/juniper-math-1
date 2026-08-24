# Phase 7 Final Approval

## Verdict

**APPROVED with remediation.** Phase 8 is **AUTHORIZED — NOT STARTED**.
The original `phase7-full-v1` candidate is invalidated: it trained on four
unmanifested shards, exactly 151,171 extra records (5,516,887 tokens), and
recorded a dirty source tree. It is historical evidence only.

## Approved Base freeze

| Item | Value |
| --- | --- |
| Run | `phase7-full-v2` |
| Checkpoint | `checkpoints/phase7-full/step_007483_final.pt` |
| SHA-256 | `2e8098ab3a5db3c59a82fad19af2050154637fbe0628f3f6b0ca932d6cb6ea60` |
| Clean training source | `0a34581ccd07de12b229528de45abbf5cb5a3a5d` |
| Architecture | v0.1.0; decoder-only, 5 layers, d_model 256, 4 Q/KV heads, SwiGLU, RMSNorm, RoPE 10,000, no biases, tied 4,096-token embedding/output; 5,004,032 parameters |
| Tokenizer | `juniper-math-tokenizer-v1` (frozen) |
| Dataset identity | `bf9933f032a58b4eb618b32156783b8563097a5fc1c0ef26be4f76445128d25a` |
| Counts | train 1,466,970; validation 81,094; test 81,014 |
| Training config | `config/training_phase7_full.yaml`, SHA-256 `d6d047472c7a606d03e826a60f1b8baf0fdebe4f20e9f110684f4e03d429efe0` |
| Seed / budget | 5,004,032; 7,483 steps; 118,167,384 loss-bearing tokens |
| Remote artifact | GitHub release `phase-7-pretraining`, verified by fresh download and SHA-256 |

The checkpoint contains model, AdamW optimizer, scheduler, CPU/CUDA/Python/
NumPy RNG states, data stream position (epoch 2, position 0), scaler field,
step/tokens, raw configuration, seed, architecture identity, and source
commit. It loads successfully; the output projection is tied to
`embed_tokens.weight` by design.

## Selection evidence

All retained candidates used identical full frozen validation, frozen v2
evaluation suites, decoding, and fixed generations. The final candidate was
selected, not merely accepted because it was last: it gives the lowest overall
validation loss (0.600296 versus 0.607735 at 80%, 0.622091 at 60%, and
0.643330 at 40%) and the best final held-out loss for most categories. The
40% candidate's 98.4% tool-call emission exceeded the final's 62.2%, but both
had 0% valid parsed calls and 0% tool-name match; selecting it would sacrifice
substantial broad validation improvement for a non-functional smoke metric.

Final frozen-suite results are deliberately modest: math 1/215 (0.47%),
calibration 0/130, adversarial 36/195 (18.46%), tool-call emission 115/185
(62.16%), valid parse 0%, tool-name match 0%. These are base-pretraining
diagnostics, not claims of mathematical or reliable tool-use ability.

## Recovery, testing, and recovery from a wipe

The independent 200-step CUDA interrupted/resumed check restored exact step
and tokens (200; 3,157,988), had maximum logged-loss difference 0.0001933610
and parameter difference 0.0002411418, and generated identical fixed outputs.
It passes the existing `< 1e-2` gate without threshold changes. PyTorch warns
that memory-efficient CUDA attention is not bitwise deterministic; the bounded
evidence demonstrates no trajectory discontinuity.

A clean processed-directory rebuild produced the manifest's exact 34 files,
hashes, counts, and identity; the hardened loader rejects unmanifested JSONL
instead of ingesting it. Repository recovery requires cloning the final tag,
installing `requirements-lock.txt`, rebuilding the dataset with
`python -m juniper_math dataset build`, retrieving the release checkpoint,
checking its SHA-256 above, and using `train full-evaluate` or `full-infer`
with `config/training_phase7_full.yaml`. Architecture, tokenizer, frozen
evaluation suites, configuration, and all recovery instructions are committed.

The final independently run regression command was
`PYTHONPATH="$PWD/src" /home/cinqic/Documents/Juniper\ Math\ 1/.venv/bin/python -m pytest -q`:
**660 passed, 0 failed, 2 documented CUDA nondeterministic-attention warnings**.

## Limits and authorization

The Base is frozen for Phase 8; architecture, tokenizer, dataset, training
configuration, evaluation suites, and this checkpoint must not be silently
changed. Phase 8 is authorized but was not started during this review.
