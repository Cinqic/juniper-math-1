"""Phase 2 tokenizer validation battery.

Implements the property-style checks required before a tokenizer candidate
can be handed to GPT-5.6 Terra: vocab size, ID range, special-token ID
stability, digit atomicity, the no-uncontrolled-multi-digit-token policy,
byte fallback / unknown-token rate, round trips, and edge cases (empty and
long input). Returns (name, passed, detail) tuples so both the CLI and the
test suite can drive the same checks without duplicating logic.
"""

from __future__ import annotations

import random

from juniper_math.tokenizer import JuniperTokenizer, TokenizerTrainingConfig, audit_vocabulary

Result = tuple[str, bool, str]

_VALIDATION_SEED = 909090

# Distinct from the tokenizer-training corpus's random inputs (Sec. 23, 42):
# a frozen, deterministic property-test set of numeric strings covering the
# required edge cases.
_NUMERIC_TEST_CASES = [
    "0",
    "0007",
    "-18392",
    "12345678901234567890",
    "3.14159265",
    "-0.00042",
    "6.022e23",
    "$12,345.67",
    "99.95%",
    "7",
    "42",
    "100",
    "1000000",
    "-1",
    "-999999",
    "+42",
    "00123",
    "1,234.56",
    ".5",
    "5.",
]

_UNICODE_TEST_CASES = [
    "€42",
    "₹500",
    "你好",
    "résumé",
    "α + β",
    "∂f/∂x",
    "∫",
    "🤖",
    "π",
    "√2",
    "x²",
    "x³",
    "a≤b",
    "x≠y",
    "a≈b",
    "±5",
    "∞",
    "∑",
    "θ",
    "Δ",
]

_TOOL_SYNTAX_CASES = [
    "<tool_call>",
    "<tool_result>",
    "<final>",
    "<unsupported>",
    "<error>",
    '<tool_call>{"expression":"2+2"}</tool_call>',
    '<tool_result>{"value":-42}</tool_result>',
]

# Round-trip corpus: representative lines across every required category
# (Sec. 41-42), independent of both the training corpus and the numeric
# test set above.
_ROUND_TRIP_CASES = [
    "What is 12 plus 37?",
    "x + 4 = 19",
    "2x - 5 = 11",
    "increase by 14%",
    "3/4",
    "3:4",
    "5kg",
    "60 mph",
    "$1,250.00",
    "6.022e23",
    "First divide both sides by 3.",
    "division by zero",
    *_TOOL_SYNTAX_CASES,
    *_UNICODE_TEST_CASES,
]


def _random_numeric_strings(count: int, seed: int = _VALIDATION_SEED) -> list[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        kind = rng.random()
        if kind < 0.3:
            out.append(str(rng.randint(-(10**18), 10**18)))
        elif kind < 0.5:
            out.append(f"{rng.uniform(-1e6, 1e6):.{rng.randint(0, 8)}f}")
        elif kind < 0.65:
            out.append(f"{rng.randint(0, 9)}" * rng.randint(1, 30))  # leading zeros / long runs
        elif kind < 0.8:
            out.append(f"{rng.uniform(1, 9.999):.4f}e{rng.randint(-30, 30)}")
        elif kind < 0.9:
            out.append(f"${rng.uniform(0, 999999):,.2f}")
        else:
            out.append(f"{rng.uniform(0, 100):.2f}%")
    return out


def check_vocab_size(tok: JuniperTokenizer, config: TokenizerTrainingConfig) -> Result:
    ok = tok.vocab_size == config.vocab_size == 4096
    return ("vocab_size", ok, f"tokenizer={tok.vocab_size} config={config.vocab_size} expected=4096")


def check_id_range(tok: JuniperTokenizer) -> Result:
    bad = [i for i in range(tok.vocab_size) if not (0 <= i < 4096)]
    # Also probe encode output on a broad text sample for out-of-range ids.
    sample_ids = tok.encode(" ".join(_ROUND_TRIP_CASES + _random_numeric_strings(200)))
    bad_encoded = [i for i in sample_ids if not (0 <= i < 4096)]
    ok = not bad and not bad_encoded
    return ("id_range", ok, f"vocab_range_violations={len(bad)} encoded_violations={len(bad_encoded)}")


def check_special_token_ids(tok: JuniperTokenizer, config: TokenizerTrainingConfig) -> Result:
    mismatches = []
    expected_core = {
        "<unk>": config.unk_id,
        "<s>": config.bos_id,
        "</s>": config.eos_id,
        "<pad>": config.pad_id,
    }
    for token, expected_id in expected_core.items():
        actual = tok.token_to_id(token)
        if actual != expected_id:
            mismatches.append(f"{token}: expected {expected_id}, got {actual}")
    for spec in config.special_tokens:
        actual = tok.token_to_id(spec.token)
        recorded = next((e["id"] for e in tok.special_tokens if e["token"] == spec.token), None)
        if recorded is None or actual != recorded:
            mismatches.append(f"{spec.token}: live id {actual} != recorded id {recorded}")
    ok = not mismatches
    return ("special_token_ids", ok, "stable" if ok else "; ".join(mismatches))


def check_digit_atomicity(tok: JuniperTokenizer) -> Result:
    failures = []
    contexts = ["{d}", " {d}", "{d}.", "-{d}", "{d}5", "5{d}", "{d}e3", "${d}", "{d}%"]
    for digit in "0123456789":
        for ctx in contexts:
            text = ctx.format(d=digit)
            pieces = tok.encode_pieces(text)
            digit_pieces = [p for p in pieces if p.replace("▁", "") == digit]
            if not digit_pieces:
                failures.append(f"{digit!r} in {text!r} did not produce an atomic digit piece: {pieces}")
    ok = not failures
    return (
        "digit_atomicity",
        ok,
        "all 10 digits atomic in all contexts" if ok else f"{len(failures)} failures",
    )


def check_no_unauthorized_multi_digit(tok: JuniperTokenizer) -> Result:
    stats = audit_vocabulary(tok)
    violations = stats["unauthorized_multi_digit_pieces"]
    ok = len(violations) == 0
    return (
        "no_unauthorized_multi_digit_pieces",
        ok,
        f"violations={violations}" if violations else "0 violations",
    )


def check_random_numeric_segmentation(tok: JuniperTokenizer, count: int = 2000) -> Result:
    failures = []
    for text in _random_numeric_strings(count):
        pieces = tok.encode_pieces(text)
        digit_runs = [p.replace("▁", "") for p in pieces if p.replace("▁", "").isdigit()]
        if any(len(run) > 1 for run in digit_runs):
            failures.append(text)
        if tok.decode(tok.encode(text)) != text:
            failures.append(f"roundtrip:{text}")
    ok = not failures
    return (
        "random_numeric_segmentation",
        ok,
        f"{count - len(failures)}/{count} passed" if not ok else f"all {count} passed",
    )


def check_byte_fallback_and_unk(tok: JuniperTokenizer) -> Result:
    unk_id = tok.token_to_id("<unk>")
    total = 0
    unk_count = 0
    for text in _UNICODE_TEST_CASES:
        ids = tok.encode(text)
        total += len(ids)
        unk_count += sum(1 for i in ids if i == unk_id)
    rate = unk_count / total if total else 0.0
    ok = rate == 0.0
    return ("byte_fallback_unk_rate", ok, f"unk_rate={rate:.4f} ({unk_count}/{total})")


def check_round_trips(tok: JuniperTokenizer) -> Result:
    failures = []
    for text in _ROUND_TRIP_CASES:
        decoded = tok.decode(tok.encode(text))
        if decoded != text:
            failures.append((text, decoded))
    ok = not failures
    return ("round_trips", ok, "all passed" if ok else f"{len(failures)} mismatches: {failures[:5]}")


def check_random_round_trips(tok: JuniperTokenizer, count: int = 1000) -> Result:
    rng = random.Random(_VALIDATION_SEED + 1)
    pool = _ROUND_TRIP_CASES + _random_numeric_strings(500)
    failures = 0
    for _ in range(count):
        n = rng.randint(1, 6)
        text = " ".join(rng.choice(pool) for _ in range(n))
        if tok.decode(tok.encode(text)) != text:
            failures += 1
    ok = failures == 0
    return ("random_round_trips", ok, f"{count - failures}/{count} passed")


def check_empty_and_whitespace_input(tok: JuniperTokenizer) -> Result:
    try:
        for text in ["", " ", "   ", "\t", "\n", "\n\n"]:
            ids = tok.encode(text)
            _ = tok.decode(ids)
        return ("empty_whitespace_input", True, "no crash on empty/whitespace-only input")
    except Exception as exc:  # noqa: BLE001 - this check exists to prove nothing raises
        return ("empty_whitespace_input", False, f"raised: {exc!r}")


def check_long_input(tok: JuniperTokenizer) -> Result:
    long_text = " ".join(_ROUND_TRIP_CASES * 500)  # far beyond the model's 1024-token context
    try:
        ids = tok.encode(long_text)
        decoded = tok.decode(ids)
        roundtrip_ok = decoded == long_text
        ok = roundtrip_ok and len(ids) > 1024
        detail = f"encoded {len(ids)} ids without truncation, round trip {'ok' if roundtrip_ok else 'FAILED'}"
        return ("long_input", ok, detail)
    except Exception as exc:  # noqa: BLE001
        return ("long_input", False, f"raised: {exc!r}")


def run_full_validation(tok: JuniperTokenizer, config: TokenizerTrainingConfig) -> list[Result]:
    return [
        check_vocab_size(tok, config),
        check_id_range(tok),
        check_special_token_ids(tok, config),
        check_digit_atomicity(tok),
        check_no_unauthorized_multi_digit(tok),
        check_random_numeric_segmentation(tok),
        check_byte_fallback_and_unk(tok),
        check_round_trips(tok),
        check_random_round_trips(tok),
        check_empty_and_whitespace_input(tok),
        check_long_input(tok),
    ]
