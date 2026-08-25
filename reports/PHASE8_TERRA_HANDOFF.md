# Phase 8 — GPT-5.6 Terra Handoff

Candidate tag: `phase-8-math-sft-candidate`. Resolve its exact commit via
`git rev-parse phase-8-math-sft-candidate^{commit}` rather than a
hardcoded SHA in this file — the same reason Phase 3's handoff used this
pattern (`config/project.yaml`'s comment on `phase_3_approval`): a commit
cannot correctly self-reference its own resulting hash.

## What changed

Phase 8 added, and did not modify anything frozen:

- `src/juniper_math/sft_rendering.py` — assistant-focused loss-masking
  (renders each example into `(text, role)` segments, tokenizes each
  segment independently, builds `-100`-masked labels).
- `src/juniper_math/sft_data.py` — deterministic, category-flattened SFT
  subset selection from the frozen `juniper-math-dataset-v1` train/
  validation splits, with length-based rejection (never truncation).
- `src/juniper_math/sft_training_config.py` — SFT config loader, fails
  loudly on any architecture/tokenizer/tool-protocol/parent-checkpoint
  identity mismatch.
- `src/juniper_math/sft_pipeline.py` — orchestration: loads the Base's
  weights only (fresh optimizer/scheduler), reuses `juniper_math.trainer`
  completely unchanged, milestone evaluation (validation loss + category
  breakdown + frozen v2 suites + the new tool-interaction suite + fixed
  generations), resume support.
- `src/juniper_math/tool_interaction.py` — the first genuine end-to-end
  tool-execution loop in this project: generate → detect `<tool_call>` →
  parse via the frozen protocol → execute via the real `ToolRuntime` →
  append the runtime's own `<tool_result>` → continue generation → extract
  final answer. Never trusts a model-generated `<tool_result>`, verified
  by a dedicated adversarial unit test.
- `src/juniper_math/sft_eval.py` — Sec. 23's 18 numerator/denominator tool
  metrics, run over a new held-out suite.
- `evals/phase8_instruction_v1.json` — new held-out suite (271 cases, all
  24 categories, seed namespace disjoint from every existing suite and
  the training corpus — verified zero overlap).
- CLI: `train sft-run`, `train sft-resume-test`, `sft-evaluate`,
  `sft-infer` (`src/juniper_math/cli.py`).
- `docs/adr/0011-*.md`, `docs/PHASE8_SFT_TRAINING.md`.
- `config/training_phase8_sft.yaml`, `config/phase8_preflight/*.yaml`,
  `config/phase8_sft_resume_check.yaml`.
- Tests: `tests/test_sft_rendering.py`, `tests/test_sft_data.py`,
  `tests/test_tool_interaction.py`, `tests/test_sft_eval.py`,
  `tests/test_sft_training_config.py` (44 new tests), plus a legitimate
  update to `tests/test_metadata.py` reflecting the new Phase 8 status.

## Where to inspect it

Start with `reports/PHASE8_PLAN.md` (what was planned before training),
then `reports/PHASE8_RESULTS.md` (what actually happened — read this one
carefully; it is not a success narrative) and `reports/PHASE8_REGRESSION.md`
(the catastrophic-forgetting finding and correction). `reports/
PHASE8_SELF_REVIEW.md` lists what was checked, what was found and fixed
during this session, and what remains unresolved.

## What was trained, on what data, and what was excluded

24,000 SFT train examples (1,000/category, all 24 categories, uniform not
corpus-proportional — see ADR 0011), 3,437 validation examples from the
frozen validation split. Excluded: frozen test split, the four frozen
Phase 4 v2 suites, the new Phase 8 suite (all verified zero-overlap).

## What the loss mask does

Prompt and `<tool_result>` tokens: `label=-100` (context-only). `<tool_call>`
tokens and the terminal `<final>`/`<unsupported>`/`<error>` tag: real
labels (supervised). BOS masked, EOS supervised, padding masked. See
`docs/PHASE8_SFT_TRAINING.md` for the full table and
`tests/test_sft_rendering.py` for the 15 tests covering every case.

## How tool execution works

`tool_interaction.run_tool_interaction`: the model generates, the harness
looks for the first `<tool_call>` block, parses it through the unmodified
Phase 3 protocol parser, executes it through the unmodified `ToolRuntime`,
and resumes generation from `prompt + model's own <tool_call> text + the
REAL runtime result` — never from anything the model generated past its
own call (verified: `tests/test_tool_interaction.py::
test_fabricated_tool_result_is_discarded_and_never_trusted`).

## Which checkpoint was selected and why

`checkpoints/phase8-sft/step_002700.pt` (of the corrected `phase8-sft-v2`
run, LR=2e-4), SHA-256
`41742e554acb6619df954b7425cebe44ed11ee1edceefb4905ae6025287d0361`. Chosen
over `step_003600.pt`/`step_004500_final.pt` on a composite of tool
metrics at n=200 (best or tied-best `correct_routing`, `tool_name_correct`,
`unnecessary_tool_call`, `fabricated_result_attempted`) — not because it
is the final step (it is not).

## Where it's preserved and its hash

GitHub release attached to tag `phase-8-math-sft-candidate` in
`https://github.com/Cinqic/juniper-math-1`. SHA-256
`41742e554acb6619df954b7425cebe44ed11ee1edceefb4905ae6025287d0361`.
Verify: download the asset and run `sha256sum` — this session did exactly
that for the *parent* Phase 7 Base checkpoint before use (matched exactly)
and re-verifies the Phase 8 asset below (§ Verification).

## Base-vs-Phase-8 results (full table in `reports/PHASE8_RESULTS.md`)

| Metric | Base | Selected candidate |
| --- | --- | --- |
| Unmasked full-corpus validation loss | 0.6062 | 0.7437 (+22.7%) |
| `correct_routing` | 0.735 | 0.765 |
| `tool_name_correct` | 0.690 | 0.750 |
| `argument_execution_successful` | **0.732** | 0.509 |
| `unnecessary_tool_call` | 0.293 | 0.252 |
| `fabricated_result_attempted` | 0.425 | 0.380 |
| `tool_use_format.valid_rate` (frozen metric) | **0.4865** | 0.3297 |
| `end_to_end_success_on_tool_required` | 0.0 | 0.0 |

## Known weaknesses / implementation risks Terra should inspect

1. **The regression tolerance was violated.** The plan pre-committed to a
   ≤0.05-absolute-nat tolerance on unmasked validation loss; the selected
   checkpoint is +0.14 nats over Base. Terra should judge whether this is
   acceptable at this model scale or whether Phase 8 needs a substantially
   different training regime.
2. **Gains are small and partially within noise** at the sample sizes
   used. Terra may want to re-evaluate with a larger/different sample or
   different seeds to check robustness.
3. **The frozen `evals/phase4_calibration_v2.json` has a latent duplicate-
   id bug** (documented in `reports/PHASE8_DATASET.md` and
   `reports/PHASE8_SELF_REVIEW.md`) that predates Phase 8 and was not
   fixed, per the frozen-artifact boundary — Terra may want a future phase
   to address it.
4. **No end-to-end task completion was demonstrated** at any checkpoint.
   Phase 8 shifted routing/format behavior; it did not make the model
   capable of finishing a tool-mediated task correctly.
5. **Only one LR correction was tried.** A fuller sweep was not run given
   the project's own bounded-preflight discipline; this is a legitimate
   area for remediation or a follow-up phase.

## Tests run and how to reproduce them

```bash
PYTHONPATH="$PWD/src" python -m pytest -q
# 704 passed, 0 failed, 2 documented CUDA-nondeterminism warnings

PYTHONPATH="$PWD/src" python -m juniper_math train sft-resume-test --config config/phase8_sft_resume_check.yaml
# PASS: resume comparison equivalent (0.0 max loss/param diff)

PYTHONPATH="$PWD/src" python -m juniper_math sft-evaluate --checkpoint checkpoints/phase8-sft/step_002700.pt --sample-size 200
```

## Candidate commit and tag

- Candidate tag: `phase-8-math-sft-candidate`
- Resolve the exact commit with: `git rev-parse phase-8-math-sft-candidate^{commit}`

## Recovery test performed this session

A genuine fresh `git clone` of `https://github.com/Cinqic/juniper-math-1.git`
into a scratch directory, checked out at `phase-8-math-sft-candidate`
(`git log -1`: `a0c4ec4 phase 8: implement and run mathematical
instruction/tool SFT`), confirmed:

1. `requirements-lock.txt` in the clone is byte-identical to the working
   copy's (same locked environment).
2. `juniper_math.pilot_data.verify_parent_dataset_shards` passes against
   the clone's own committed `shard_manifest.json`/`DATASET_IDENTITY.sha256`
   — **with one documented limitation**: the actual `.jsonl` shard bytes
   (correctly not committed to Git, per this project's own
   local-storage-is-disposable convention) were copied in from this
   session's already-verified local copy rather than regenerated from
   scratch via `dataset build`, because a full 1.46M-example regeneration
   was judged too time-expensive for this session's bounded recovery
   check. The hash-verification step itself is genuine and unmodified;
   only the *source* of the shard bytes it checked was not a from-scratch
   rebuild. This is a real limitation, stated rather than concealed.
3. The Phase 7 Base checkpoint and the selected Phase 8 checkpoint were
   both retrieved fresh from their respective GitHub releases (not copied
   from the working copy) and independently re-hashed inside the clone:
   both matched their recorded SHA-256 exactly.
4. The clone's own `src/` (via `PYTHONPATH`) successfully: loaded the
   Phase 8 model from the retrieved checkpoint; ran direct inference
   (`"What is 5 times 6?"` → `<final>30`); ran the full end-to-end
   tool-interaction loop (`run_tool_interaction`) on a tool-use prompt;
   and ran the Phase 8 eval suite (`run_phase8_eval_suite`) on a 10-case
   sample.
5. The Python environment itself was **not** rebuilt from
   `requirements-lock.txt` in a fresh venv — this session reused the
   already-validated `.venv` for the clone's execution, given time
   constraints. This is the other explicitly documented limitation of
   this recovery check.

Exact commands:

```bash
git clone https://github.com/Cinqic/juniper-math-1.git recovery_clone
cd recovery_clone && git checkout phase-8-math-sft-candidate

# (shard bytes copied in locally — see limitation above)
PYTHONPATH="$PWD/src" python -c "
from juniper_math.dataset.config import load_dataset_config
from juniper_math.pilot_data import verify_parent_dataset_shards
verify_parent_dataset_shards(load_dataset_config())"

gh release download phase-7-pretraining --repo Cinqic/juniper-math-1 \
  --pattern step_007483_final.pt --dir checkpoints/phase7-full-v2/
gh release download phase-8-math-sft-candidate --repo Cinqic/juniper-math-1 \
  --pattern step_002700.pt --dir checkpoints/phase8-sft/
sha256sum checkpoints/phase7-full-v2/step_007483_final.pt checkpoints/phase8-sft/step_002700.pt

PYTHONPATH="$PWD/src" python -m juniper_math sft-infer \
  --checkpoint checkpoints/phase8-sft/step_002700.pt --prompt "What is 5 times 6?"
```

## Repository status

`config/project.yaml`: `current_phase: 7` (last independently approved
phase, unchanged), `next_phase.status: "ENGINEERING COMPLETE — AWAITING
GPT-5.6 TERRA INDEPENDENT REVIEW"`, `phase_8_engineering.terra_independent_
review: "pending"`, `phase_8_engineering.terra_final_approval: "pending"`.
Phase 9 is not authorized. GPT-5.6 Terra has not yet independently
approved Phase 8 — this handoff exists so Terra does not need to
reverse-engineer the phase before reviewing it.
