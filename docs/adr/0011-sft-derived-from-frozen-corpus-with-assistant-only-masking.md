# ADR 0011: Phase 8 SFT data is selected from the frozen corpus, not
generated fresh, and loss is masked to assistant-authored tokens only

**Context.** Phase 8 needs supervised examples covering direct answers,
tool use, error correction, clarification, and unsupported requests
(reports/PHASE8_PLAN.md Sec. 4). The frozen `juniper-math-dataset-v1`
train split already contains verified examples in all 24 categories
(`VALID_CATEGORIES`), each with ground truth established the same way
(deterministic recomputation or a real `ToolRuntime` execution trace).
Separately, Phase 5-7 tokenize and train on an example's *entire* rendered
text (prompt + tool trace + answer) as one undifferentiated loss-bearing
sequence, which is correct for base pretraining but wrong for instruction
tuning: it would train the model to reproduce the user's own prompt and the
runtime's own tool result, not just to produce the assistant's turns.

**Decision.** Two decisions, made together:

1. Phase 8's SFT corpus (`juniper-math-sft-v1`) is a deterministic,
   category-flattened *subset selection* over the frozen corpus's existing
   train/validation splits (`juniper_math.sft_data`), not a new generator
   corpus. No new generator module is introduced; ADR 0004's "deterministic
   tools over neural guessing" invariant carries over unchanged because
   every selected tool-required example's ground truth is still a real,
   previously-executed `ToolRuntime` trace.
2. Training uses assistant-focused loss masking (`juniper_math.
   sft_rendering`): the user's prompt and the runtime's `<tool_result>`
   blocks are label `-100` (context-only); the assistant's `<tool_call>`
   block and its terminal `<final>`/`<unsupported>`/`<error>` tag are the
   only loss-bearing positions. Segment-wise tokenization (never
   re-tokenizing the joined string) makes this exact at the token level by
   construction, verified in this session against zero mismatches on 7,452
   sampled examples versus the existing joint-string tokenization Phase 5-7
   use.

**Consequences.** Phase 8 cannot silently regress the frozen corpus (it is
read-only input, never mutated) or the frozen tool protocol (every
supervised tool-call target is byte-identical to what the real runtime
already validated at Phase 4 build time). The tradeoff is that Phase 8's
selection is bounded by whatever behavioral diversity the Phase 4 generators
already produced; a genuinely new behavior Phase 4's generators cannot
express (not identified during Phase 8) would require either a new,
separately-reviewed generator module in a later phase, or documenting the
gap rather than working around it inside Phase 8's frozen-data boundary
(reports/PHASE8_PLAN.md Sec. 13).
