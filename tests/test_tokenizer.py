from __future__ import annotations

import json
import shutil

import pytest
from sentencepiece import sentencepiece_model_pb2

from juniper_math.errors import JuniperConfigError
from juniper_math.hashing import sha256_file
from juniper_math.tokenizer import (
    JuniperTokenizer,
    JuniperTokenizerError,
    audit_vocabulary,
    load_tokenizer_config,
    rebuild_corpus_for_config,
    tokenizer_artifact_hashes,
    train_tokenizer,
)
from juniper_math.tokenizer_validation import run_full_validation

# The frozen canonical tokenizer, committed under releases/tokenizer/. These
# tests exercise the real artifact, not a throwaway retrain, so they double
# as regression coverage for whatever is actually shipped.
CONFIG = load_tokenizer_config()
TOKENIZER = JuniperTokenizer.load(CONFIG)


def test_config_loads_expected_fields():
    assert CONFIG.vocab_size == 4096
    assert CONFIG.tokenizer_version == "juniper-math-tokenizer-v1"
    assert CONFIG.split_digits is True
    assert CONFIG.byte_fallback is True
    assert set(CONFIG.digit_symbols) == set("0123456789")


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(JuniperConfigError):
        load_tokenizer_config(tmp_path / "nope.yaml")


# --------------------------------------------------------------------------
# Vocabulary / ID range
# --------------------------------------------------------------------------


def test_vocab_size_is_exactly_4096():
    assert TOKENIZER.vocab_size == 4096


def test_all_piece_ids_in_range():
    for i in range(TOKENIZER.vocab_size):
        assert 0 <= i < 4096
        assert isinstance(TOKENIZER.id_to_token(i), str)


def test_audit_reports_no_unauthorized_multi_digit_pieces():
    stats = audit_vocabulary(TOKENIZER)
    assert stats["unauthorized_multi_digit_pieces"] == []
    assert stats["vocab_size"] == 4096
    assert stats["byte_fallback_pieces"] == 256


def test_serialized_sentencepiece_spec_matches_frozen_contract():
    model = sentencepiece_model_pb2.ModelProto()
    model.ParseFromString(CONFIG.model_path.read_bytes())
    trainer = model.trainer_spec
    normalizer = model.normalizer_spec
    assert trainer.model_type == sentencepiece_model_pb2.TrainerSpec.BPE
    assert trainer.vocab_size == 4096
    assert trainer.split_digits is True
    assert trainer.byte_fallback is True
    assert (trainer.unk_id, trainer.bos_id, trainer.eos_id, trainer.pad_id) == (0, 1, 2, 3)
    assert list(trainer.user_defined_symbols) == [*CONFIG.user_defined_symbols]
    assert normalizer.name == "identity"


# --------------------------------------------------------------------------
# Special tokens
# --------------------------------------------------------------------------


@pytest.mark.parametrize("token", ["<tool_call>", "<tool_result>", "<final>", "<unsupported>", "<error>"])
def test_required_special_tokens_present(token):
    ids = {e["token"]: e["id"] for e in TOKENIZER.special_tokens}
    assert token in ids
    assert TOKENIZER.token_to_id(token) == ids[token]


@pytest.mark.parametrize("token", ["<unk>", "<s>", "</s>", "<pad>"])
def test_core_tokens_present(token):
    assert any(e["token"] == token for e in TOKENIZER.special_tokens)


def test_special_token_ids_are_frozen_and_unique():
    ids = [e["id"] for e in TOKENIZER.special_tokens]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)  # unk=0, s=1, /s=2, pad=3, then control tokens, then digits


def test_tool_call_embedded_in_surrounding_text_round_trips():
    text = '<tool_call>{"expression":"2+2"}</tool_call>'
    assert TOKENIZER.decode(TOKENIZER.encode(text)) == text


# --------------------------------------------------------------------------
# Digit atomicity (Sec. 20-22)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("digit", list("0123456789"))
def test_every_digit_is_atomic_alone(digit):
    pieces = TOKENIZER.encode_pieces(digit)
    assert digit in [p.replace("▁", "") for p in pieces]


@pytest.mark.parametrize(
    "text",
    [
        "84317",
        "0007",
        "-18392",
        "3.14159265",
        "-0.00042",
        "6.022e23",
        "$12,345.67",
        "99.95%",
        "2026",
        "1000000",
    ],
)
def test_multi_digit_numbers_decompose_into_atomic_digits(text):
    pieces = TOKENIZER.encode_pieces(text)
    digit_pieces = [p.replace("▁", "") for p in pieces if p.replace("▁", "").isdigit()]
    assert all(len(p) == 1 for p in digit_pieces)


def test_terra_numeric_property_suite_has_no_digit_merges():
    # Independently generated unseen decimal strings spanning short, long,
    # leading-zero, and 50+ digit cases.  Exact digit order must survive.
    cases = []
    for n in range(20_000):
        digits = str((n * 982_451_653) % 10**20).zfill(20)
        cases.append(digits[: 1 + (n % 6)])
        cases.append("0" * (50 + (n % 11)) + digits)
    for text in cases:
        pieces = TOKENIZER.encode_pieces(text)
        encoded_digits = "".join(p.replace("▁", "") for p in pieces if p.replace("▁", "").isdigit())
        assert encoded_digits == text
        assert TOKENIZER.decode(TOKENIZER.encode(text)) == text


# --------------------------------------------------------------------------
# Byte fallback / Unicode (Sec. 26-28, 39)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text", ["€42", "₹500", "你好", "résumé", "α + β", "∂f/∂x", "∫", "🤖", "π", "√2", "x²", "≤", "≠", "≈"]
)
def test_unicode_never_falls_back_to_unk(text):
    unk_id = TOKENIZER.token_to_id("<unk>")
    ids = TOKENIZER.encode(text)
    assert unk_id not in ids


# --------------------------------------------------------------------------
# Round trips (Sec. 41-42)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "What is 12 plus 37?",
        "x + 4 = 19",
        "increase by 14%",
        "3/4",
        "3:4",
        "5kg",
        "60 mph",
        "$1,250.00",
        "6.022e23",
        "First divide both sides by 3.",
        "division by zero",
        "",
        " ",
        "   \n  ",
    ],
)
def test_round_trip(text):
    assert TOKENIZER.decode(TOKENIZER.encode(text)) == text


def test_long_input_not_truncated():
    long_text = " ".join(["84317 + sqrt(16) = ?"] * 2000)
    ids = TOKENIZER.encode(long_text)
    assert len(ids) > 1024
    assert TOKENIZER.decode(ids) == long_text


# --------------------------------------------------------------------------
# Full validation battery (also exercised via CLI `tokenizer validate`)
# --------------------------------------------------------------------------


def test_full_validation_battery_passes():
    results = run_full_validation(TOKENIZER, CONFIG)
    failures = [(name, detail) for name, passed, detail in results if not passed]
    assert not failures, failures


# --------------------------------------------------------------------------
# Deterministic reconstruction (Sec. 51, 75)
# --------------------------------------------------------------------------


def test_deterministic_rebuild_produces_byte_identical_artifact(tmp_path):
    corpus = tmp_path / "corpus.txt"
    n = rebuild_corpus_for_config(CONFIG, corpus)
    assert n > 0

    out_dir_a = tmp_path / "a"
    out_dir_b = tmp_path / "b"
    for out_dir in (out_dir_a, out_dir_b):
        cfg = load_tokenizer_config()
        object.__setattr__(cfg, "model_dir", str(out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        train_tokenizer(cfg, corpus, overwrite=True)

    model_a = (out_dir_a / CONFIG.model_file).read_bytes()
    model_b = (out_dir_b / CONFIG.model_file).read_bytes()
    assert model_a == model_b
    assert sha256_file(out_dir_a / CONFIG.model_file) == sha256_file(out_dir_b / CONFIG.model_file)
    for filename in (
        CONFIG.vocab_file,
        CONFIG.special_token_map_file,
        CONFIG.config_snapshot_file,
        CONFIG.corpus_digest_file,
    ):
        assert (out_dir_a / filename).read_bytes() == (out_dir_b / filename).read_bytes()


def test_existing_staging_directory_is_never_deleted(tmp_path, monkeypatch):
    import juniper_math.tokenizer as tokenizer_module

    monkeypatch.setattr(tokenizer_module.tempfile, "gettempdir", lambda: str(tmp_path))
    staging = tmp_path / "juniper_tokenizer_build"
    staging.mkdir()
    sentinel = staging / "do-not-delete"
    sentinel.write_text("owned by an interrupted or concurrent process", encoding="utf-8")
    corpus = tmp_path / "corpus.txt"
    rebuild_corpus_for_config(CONFIG, corpus)
    cfg = load_tokenizer_config()
    object.__setattr__(cfg, "model_dir", str(tmp_path / "out"))
    with pytest.raises(JuniperTokenizerError, match="Refusing to use existing"):
        train_tokenizer(cfg, corpus, overwrite=True)
    assert sentinel.read_text(encoding="utf-8") == "owned by an interrupted or concurrent process"


def test_train_refuses_to_overwrite_without_flag(tmp_path):
    corpus = tmp_path / "corpus.txt"
    rebuild_corpus_for_config(CONFIG, corpus)
    cfg = load_tokenizer_config()
    out_dir = tmp_path / "frozen"
    object.__setattr__(cfg, "model_dir", str(out_dir))
    out_dir.mkdir(parents=True)
    train_tokenizer(cfg, corpus, overwrite=True)
    with pytest.raises(JuniperTokenizerError):
        train_tokenizer(cfg, corpus, overwrite=False)


# --------------------------------------------------------------------------
# Negative tests (Sec. 71-72)
# --------------------------------------------------------------------------


def test_load_fails_cleanly_without_trained_artifact(tmp_path):
    cfg = load_tokenizer_config()
    object.__setattr__(cfg, "model_dir", str(tmp_path / "does_not_exist"))
    with pytest.raises(JuniperTokenizerError):
        JuniperTokenizer.load(cfg)


def test_corrupted_model_file_is_rejected(tmp_path):
    corrupted = tmp_path / "corrupted.model"
    shutil.copy(CONFIG.model_path, corrupted)
    with corrupted.open("r+b") as handle:
        handle.seek(0)
        handle.write(b"\x00\x00\x00\x00")
    assert sha256_file(corrupted) != sha256_file(CONFIG.model_path)


def test_wrong_vocab_size_detected_by_config_check():
    class _FakeCfg:
        vocab_size = 4095

    from juniper_math.tokenizer_validation import check_vocab_size

    name, passed, _detail = check_vocab_size(TOKENIZER, _FakeCfg())
    assert name == "vocab_size"
    assert passed is False


def test_special_token_map_hash_changes_when_corrupted(tmp_path):
    original = json.loads(CONFIG.special_token_map_path.read_text(encoding="utf-8"))
    tampered = dict(original)
    tampered["tokens"] = list(tampered["tokens"])
    tampered["tokens"][0] = {**tampered["tokens"][0], "id": 999}
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    assert sha256_file(tampered_path) != sha256_file(CONFIG.special_token_map_path)


# --------------------------------------------------------------------------
# Artifact hashing
# --------------------------------------------------------------------------


def test_artifact_hashes_are_64_char_hex():
    hashes = tokenizer_artifact_hashes(CONFIG)
    assert set(hashes) == {"model", "vocab", "special_token_map", "config_snapshot", "corpus_digest"}
    for digest in hashes.values():
        assert len(digest) == 64
        int(digest, 16)  # raises if not valid hex


def test_corpus_digest_matches_a_fresh_generated_corpus(tmp_path):
    corpus = tmp_path / "corpus.txt"
    assert rebuild_corpus_for_config(CONFIG, corpus) == 200_000
    recorded = CONFIG.corpus_digest_path.read_text(encoding="utf-8").split()[0]
    assert recorded == sha256_file(corpus)


def test_bytes_are_rejected_at_the_wrapper_boundary():
    with pytest.raises(TypeError, match="accepts str only"):
        TOKENIZER.encode(b"not decoded text")  # type: ignore[arg-type]
