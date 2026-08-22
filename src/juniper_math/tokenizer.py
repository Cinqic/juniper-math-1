"""Juniper Math 1 canonical tokenizer: training, wrapper, and validation.

Wraps SentencePiece so the rest of the project (and Phase 3+) never touches
SentencePiece-specific quirks directly (Sec. 45). Everything that defines
tokenizer *identity* — vocab size, BPE settings, byte fallback,
normalization, special tokens, digit policy, seed — lives in
config/tokenizer.yaml, not scattered Python constants (Sec. 46).
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sentencepiece as spm
import yaml

from juniper_math.errors import JuniperConfigError
from juniper_math.hashing import sha256_file
from juniper_math.paths import CONFIG_DIR, REPO_ROOT
from juniper_math.tokenizer_corpus import DEFAULT_TOTAL_LINES, MASTER_SEED, write_corpus

TOKENIZER_CONFIG_PATH = CONFIG_DIR / "tokenizer.yaml"

# Vocabulary pieces containing two-or-more consecutive decimal digits are the
# uncontrolled multi-digit tokens Sec. 21 forbids. SentencePiece's internal
# whitespace marker "▁" is not a digit and never confuses this pattern.
_MULTI_DIGIT_PIECE = re.compile(r"\d{2,}")
# SentencePiece byte-fallback pieces look like "<0x1A>" — their hex digits are
# not decimal digits and must never be scanned by _MULTI_DIGIT_PIECE.
_BYTE_FALLBACK_PIECE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


class JuniperTokenizerError(RuntimeError):
    pass


@dataclass(frozen=True)
class SpecialTokenSpec:
    token: str
    purpose: str
    user_visible: bool
    generated_by_model: bool
    runtime_only: bool


@dataclass(frozen=True)
class TokenizerTrainingConfig:
    tokenizer_version: str
    vocab_size: int
    seed: int
    corpus_total_lines: int
    model_type: str
    character_coverage: float
    split_digits: bool
    byte_fallback: bool
    normalization_rule_name: str
    add_dummy_prefix: bool
    remove_extra_whitespace: bool
    unk_id: int
    bos_id: int
    eos_id: int
    pad_id: int
    digit_symbols: list[str]
    special_tokens: list[SpecialTokenSpec]
    model_dir: str
    model_file: str
    vocab_file: str
    special_token_map_file: str
    config_snapshot_file: str
    raw: dict[str, Any]

    @property
    def model_path(self) -> Path:
        return REPO_ROOT / self.model_dir / self.model_file

    @property
    def vocab_path(self) -> Path:
        return REPO_ROOT / self.model_dir / self.vocab_file

    @property
    def special_token_map_path(self) -> Path:
        return REPO_ROOT / self.model_dir / self.special_token_map_file

    @property
    def config_snapshot_path(self) -> Path:
        return REPO_ROOT / self.model_dir / self.config_snapshot_file

    @property
    def user_defined_symbols(self) -> list[str]:
        return [s.token for s in self.special_tokens] + list(self.digit_symbols)


def load_tokenizer_config(path: Path | None = None) -> TokenizerTrainingConfig:
    source = path or TOKENIZER_CONFIG_PATH
    if not source.is_file():
        raise JuniperConfigError(f"Tokenizer config not found at {source}.")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise JuniperConfigError(f"{source}: invalid YAML ({exc}).") from exc

    try:
        training = raw["training"]
        specials = [
            SpecialTokenSpec(
                token=s["token"],
                purpose=s["purpose"],
                user_visible=s["user_visible"],
                generated_by_model=s["generated_by_model"],
                runtime_only=s["runtime_only"],
            )
            for s in raw["special_tokens"]
        ]
        output = raw["output"]
        return TokenizerTrainingConfig(
            tokenizer_version=raw["tokenizer_version"],
            vocab_size=raw["vocab_size"],
            seed=training["seed"],
            corpus_total_lines=training["corpus_total_lines"],
            model_type=training["model_type"],
            character_coverage=training["character_coverage"],
            split_digits=training["split_digits"],
            byte_fallback=training["byte_fallback"],
            normalization_rule_name=training["normalization_rule_name"],
            add_dummy_prefix=training["add_dummy_prefix"],
            remove_extra_whitespace=training["remove_extra_whitespace"],
            unk_id=training["unk_id"],
            bos_id=training["bos_id"],
            eos_id=training["eos_id"],
            pad_id=training["pad_id"],
            digit_symbols=list(raw["digit_symbols"]),
            special_tokens=specials,
            model_dir=output["model_dir"],
            model_file=output["model_file"],
            vocab_file=output["vocab_file"],
            special_token_map_file=output["special_token_map_file"],
            config_snapshot_file=output["config_snapshot_file"],
            raw=raw,
        )
    except (KeyError, TypeError) as exc:
        raise JuniperConfigError(f"{source}: missing or malformed field ({exc}).") from exc


def train_tokenizer(
    config: TokenizerTrainingConfig,
    corpus_path: Path,
    overwrite: bool = False,
) -> None:
    """Train a SentencePiece BPE model per ``config`` and write artifacts.

    Refuses to overwrite an existing frozen artifact unless ``overwrite`` is
    explicitly set (Sec. 50) — an accidental rerun of this function must
    never silently mutate a tokenizer that has already been approved.
    """
    if config.model_path.exists() and not overwrite:
        raise JuniperTokenizerError(
            f"Refusing to overwrite existing tokenizer artifact at {config.model_path}. "
            f"Pass overwrite=True (CLI: --overwrite) to deliberately retrain."
        )

    config.model_path.parent.mkdir(parents=True, exist_ok=True)

    # SentencePiece embeds its TrainerSpec — including the literal `input`
    # and `model_prefix` paths it was called with — inside the serialized
    # .model file. Training directly at config.model_path would make the
    # artifact depend on the repository's absolute clone location, so two
    # honest rebuilds from an identical corpus/config/seed on two different
    # machines (or two different checkouts) would produce byte-DIFFERENT
    # models despite being semantically identical. Training instead against
    # a fixed, path-invariant staging location makes the embedded metadata
    # (and therefore the artifact hash) depend only on the corpus bytes and
    # training parameters, which is what Sec. 51/75 deterministic
    # reconstruction actually requires.
    staging_dir = Path(tempfile.gettempdir()) / "juniper_tokenizer_build"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    staging_corpus = staging_dir / "corpus.txt"
    shutil.copyfile(corpus_path, staging_corpus)
    staging_prefix = staging_dir / config.model_path.stem

    spm.SentencePieceTrainer.Train(
        input=str(staging_corpus),
        model_prefix=str(staging_prefix),
        vocab_size=config.vocab_size,
        model_type=config.model_type,
        character_coverage=config.character_coverage,
        split_digits=config.split_digits,
        byte_fallback=config.byte_fallback,
        normalization_rule_name=config.normalization_rule_name,
        add_dummy_prefix=config.add_dummy_prefix,
        remove_extra_whitespaces=config.remove_extra_whitespace,
        unk_id=config.unk_id,
        bos_id=config.bos_id,
        eos_id=config.eos_id,
        pad_id=config.pad_id,
        user_defined_symbols=config.user_defined_symbols,
        num_threads=1,  # deterministic reconstruction (Sec. 51) over training speed
        shuffle_input_sentence=False,
        input_sentence_size=0,
        train_extremely_large_corpus=False,
    )

    shutil.copyfile(staging_prefix.with_suffix(".model"), config.model_path)
    shutil.copyfile(staging_prefix.with_suffix(".vocab"), config.vocab_path)
    shutil.rmtree(staging_dir)

    sp = spm.SentencePieceProcessor(model_file=str(config.model_path))
    special_map = _build_special_token_map(config, sp)
    config.special_token_map_path.write_text(
        json.dumps(special_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    config.config_snapshot_path.write_text(
        json.dumps(config.raw, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )


def _build_special_token_map(config: TokenizerTrainingConfig, sp: spm.SentencePieceProcessor) -> dict:
    core = {
        "<unk>": config.unk_id,
        "<s>": config.bos_id,
        "</s>": config.eos_id,
        "<pad>": config.pad_id,
    }
    entries = []
    for token, tid in core.items():
        entries.append(
            {
                "token": token,
                "id": tid,
                "category": "core",
                "purpose": {
                    "<unk>": "unknown-token fallback",
                    "<s>": "beginning-of-sequence marker",
                    "</s>": "end-of-sequence marker",
                    "<pad>": "padding for batched sequences",
                }[token],
                "user_visible": False,
                "generated_by_model": False,
                "runtime_only": True,
            }
        )
    for spec in config.special_tokens:
        entries.append(
            {
                "token": spec.token,
                "id": sp.piece_to_id(spec.token),
                "category": "control",
                "purpose": spec.purpose,
                "user_visible": spec.user_visible,
                "generated_by_model": spec.generated_by_model,
                "runtime_only": spec.runtime_only,
            }
        )
    for digit in config.digit_symbols:
        entries.append(
            {
                "token": digit,
                "id": sp.piece_to_id(digit),
                "category": "digit",
                "purpose": "atomic decimal digit",
                "user_visible": True,
                "generated_by_model": True,
                "runtime_only": False,
            }
        )
    return {"tokenizer_version": config.tokenizer_version, "vocab_size": sp.vocab_size(), "tokens": entries}


class JuniperTokenizer:
    """Project-owned tokenizer wrapper. Phase 3+ code should use this, not SentencePiece directly."""

    def __init__(self, sp: spm.SentencePieceProcessor, special_token_map: dict) -> None:
        self._sp = sp
        self._special_token_map = special_token_map

    @classmethod
    def load(cls, config: TokenizerTrainingConfig | None = None) -> JuniperTokenizer:
        cfg = config or load_tokenizer_config()
        if not cfg.model_path.is_file():
            raise JuniperTokenizerError(
                f"No trained tokenizer at {cfg.model_path}. Run `juniper-math tokenizer train` first."
            )
        sp = spm.SentencePieceProcessor(model_file=str(cfg.model_path))
        special_map = json.loads(cfg.special_token_map_path.read_text(encoding="utf-8"))
        return cls(sp, special_map)

    def encode(self, text: str) -> list[int]:
        return list(self._sp.encode(text, out_type=int))

    def encode_pieces(self, text: str) -> list[str]:
        return list(self._sp.encode(text, out_type=str))

    def decode(self, ids: list[int]) -> str:
        return str(self._sp.decode(ids))

    def token_to_id(self, token: str) -> int:
        return int(self._sp.piece_to_id(token))

    def id_to_token(self, token_id: int) -> str:
        return str(self._sp.id_to_piece(token_id))

    @property
    def vocab_size(self) -> int:
        return int(self._sp.vocab_size())

    @property
    def special_tokens(self) -> list[dict]:
        return list(self._special_token_map["tokens"])

    def all_pieces(self) -> list[str]:
        return [self._sp.id_to_piece(i) for i in range(self._sp.vocab_size())]


# --------------------------------------------------------------------------
# Vocabulary auditing (Sec. 59-61)
# --------------------------------------------------------------------------


def audit_vocabulary(tok: JuniperTokenizer) -> dict[str, Any]:
    pieces = tok.all_pieces()
    special_strings = {s["token"] for s in tok.special_tokens}
    digit_strings = {s["token"] for s in tok.special_tokens if s["category"] == "digit"}

    stats: dict[str, Any] = {
        "vocab_size": len(pieces),
        "special_tokens": len(special_strings),
        "digits": len(digit_strings),
        "whitespace_prefixed": sum(1 for p in pieces if p.startswith("▁")),
        "byte_fallback_pieces": sum(1 for p in pieces if re.fullmatch(r"<0x[0-9A-F]{2}>", p)),
        "pieces_containing_digit": sum(1 for p in pieces if any(c.isdigit() for c in p)),
        "unusually_long_pieces": [p for p in pieces if len(p) > 20],
    }

    violations = []
    for piece in pieces:
        if piece in special_strings or _BYTE_FALLBACK_PIECE.match(piece):
            continue
        # Strip the SentencePiece whitespace marker before scanning for
        # digit runs — it is not part of the piece's textual content.
        content = piece.replace("▁", "")
        if _MULTI_DIGIT_PIECE.search(content):
            violations.append(piece)
    stats["unauthorized_multi_digit_pieces"] = violations
    return stats


# --------------------------------------------------------------------------
# Hashing / identity
# --------------------------------------------------------------------------


def tokenizer_artifact_hashes(config: TokenizerTrainingConfig) -> dict[str, str]:
    return {
        "model": sha256_file(config.model_path),
        "vocab": sha256_file(config.vocab_path),
        "special_token_map": sha256_file(config.special_token_map_path),
        "config_snapshot": sha256_file(config.config_snapshot_path),
    }


def rebuild_corpus_for_config(config: TokenizerTrainingConfig, corpus_path: Path) -> int:
    return write_corpus(
        corpus_path,
        seed=config.seed or MASTER_SEED,
        total_lines=config.corpus_total_lines or DEFAULT_TOTAL_LINES,
    )
