# Phase 8 SFT Dataset V4: Independent Direct Curriculum

## Purpose

V3's frame rewrites left full-run held-out direct correctness at 0/160. V4
adds a separate, deterministic direct-mathematics curriculum to address that
specific generalization failure. It does not alter the frozen parent corpus or
reuse any frozen evaluation prompt.

## Construction

For each train/validation split, the curriculum creates 1,000 / 150 new
records per direct category for arithmetic, operator precedence, negative
values, decimals, fractions, percentages, ratios, scientific notation,
algebra, expression translation, word problems, and multi-step problems.
Each is constructed from a closed verification expression and uses new family
IDs and prompt structures, including surveys, dives, recipes, scaled mixtures,
linear rules, inventories, tanks, and two-leg travel.

All examples are direct only: no tool call, tool trace, or model-authored tool
result is introduced. The frozen tool/error/semantic records remain in the
parent selection for behavioral coverage.

## Identity and composition

- parent selection identity: `1fbcaf6afe623529badf2c2e2fd7faf5e541928e239359152b70ba2973681f1e`;
- effective V4 representation identity:
  `ebb7039bac1386e1108765f23c71585d4a4a0202e09b6242fe20fdbb86324f3c`;
- V4 train records: 51,000.

The actual token/label representation hash changes whenever the independent
curriculum, prompt frames, tokenizer, renderer, or mask changes. The parent
selection identity remains separate because it describes only frozen parent
record selection.

## Gate

V4 is untrained. Its preflight must first show Base regression within +0.05
nats and a material direct-held-out gain before a full candidate is justified.
