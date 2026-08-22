# Phase 2 Tokenizer Manual Inspection Report

Mandatory manual inspection (Sec. 58) — metrics can conceal absurd
segmentation, so this report reads actual pieces and IDs by hand, not just
aggregate benchmark numbers. Reproduce any row with:

```bash
python -m juniper_math tokenizer encode "<input text>"
```

`▁` is SentencePiece's whitespace marker (a leading space belongs to the
piece that follows it, not a separate token).

| Input | Pieces | IDs | Round trip | Comment |
|---|---|---|---|---|
| `What is 84317 * 9926?` | `What`, `▁is`, `▁`, `8`, `4`, `3`, `1`, `7`, `▁*`, `▁`, `9`, `9`, `2`, `6`, `?` | `314, 308, 4030, 17, 13, 12, 10, 16, 515, 4030, 18, 18, 11, 15, 4054` | OK | Both integers decompose into fully atomic digits — `84317` is five separate digit tokens, never a memorized number piece. |
| `2x - 5 = 11` | `2`, `x`, `▁-`, `▁`, `5`, `▁=`, `▁`, `1`, `1` | `11, 4052, 412, 4030, 14, 317, 4030, 10, 10` | OK | `11` is two atomic `1` tokens, not a two-digit piece — this is the digit-atomicity policy working exactly as designed. |
| `increase by 14%` | `increase`, `▁by`, `▁`, `1`, `4`, `%` | `588, 380, 4030, 10, 13, 4064` | OK | Percent sign is its own piece; `14` splits into `1`, `4`. |
| `3/4` | `3`, `/`, `4` | `12, 4057, 13` | OK | Fraction slash is a standalone piece, not fused with either digit. |
| `6.022e23` | `6`, `.`, `0`, `2`, `2`, `e`, `2`, `3` | `15, 4041, 9, 11, 11, 4031, 11, 12` | OK | Every digit in the mantissa and exponent is atomic; `e` (scientific-notation marker) is its own piece. |
| `$1,250.00` | `$`, `1`, `,`, `2`, `5`, `0`, `.`, `0`, `0` | `4062, 10, 4059, 11, 14, 9, 4041, 9, 9` | OK | Currency symbol, thousands separator, and decimal point are all standalone pieces; every digit is atomic even inside a comma-grouped amount. |
| `5kg` | `5`, `kg` | `14, 1032` | OK | Unit `kg` merges as a whole learned piece; digit stays atomic and separate. |
| `First divide both sides by 3.` | `First`, `▁divide`, `▁both`, `▁sides`, `▁by`, `▁`, `3`, `.` | `770, 700, 494, 495, 380, 4030, 12, 4041` | OK | Common explanation vocabulary merges into whole-word pieces (`First`, `divide`, `both`, `sides`) — this is the natural-language category paying off. |
| `division by zero` | `division`, `▁by`, `▁zero` | `939, 380, 603` | OK | Entire error phrase compresses to 3 tokens. |
| `<tool_call>{"expression":"2+2"}</tool_call>` | `<tool_call>`, `{"`, `expression`, `":"`, `2`, `+`, `2`, `"}</`, `tool`, `_`, `call`, `>` | `4, 387, 492, 490, 11, 4056, 11, 491, 375, 4077, 485, 4071` | OK | The opening `<tool_call>` is a single reserved atomic piece (id 4, exactly as frozen); the JSON body and closing tag are ordinary text and tokenize/decode normally — no corruption of the reserved token boundary. |
| `€42 你好 🤖 ∂f/∂x` | 27 pieces, mostly `<0xXX>` byte-fallback pieces (see full list via CLI) | — | OK | None of `€`, `你好`, `🤖`, `∂` appear as direct vocabulary pieces (rare in the training corpus at `character_coverage=0.9995`), so byte fallback correctly decomposes them into raw UTF-8 bytes and reconstructs the exact original string on decode — this is byte fallback working as intended, not a failure. |
| `The values are insufficient to determine a unique answer.` | `The`, `▁values`, `▁are`, `▁insufficient`, `▁to`, `▁determine`, `▁a`, `▁unique`, `▁answer`, `.` | `419, 795, 406, 751, 307, 753, 277, 749, 497, 4041` | OK | A full explanatory sentence compresses to 10 whole-word pieces — no character-level fallback needed for ordinary English. |

## Summary judgment

No absurd segmentation was found. Digits are atomic in every case tested,
including inside currency, percentages, scientific notation, and comma-grouped
numbers. Reserved control tokens tokenize as single atomic pieces and don't
bleed into surrounding text. Byte fallback activates only for genuinely rare
characters and round-trips exactly. Common English and math-explanation
vocabulary merges into efficient whole-word pieces rather than falling back
to character-level tokenization.
