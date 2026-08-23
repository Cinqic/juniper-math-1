"""Deterministic cleaning/normalization pipeline (Sec. 15)."""

from __future__ import annotations

import re
import unicodedata
from typing import Literal, get_args

from juniper_math.dataset.config import NormalizationConfig
from juniper_math.errors import JuniperConfigError

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

_UnicodeForm = Literal["NFC", "NFD", "NFKC", "NFKD"]
_VALID_UNICODE_FORMS = frozenset(get_args(_UnicodeForm))


def normalize_text(text: str, config: NormalizationConfig) -> str:
    if config.unicode_form not in _VALID_UNICODE_FORMS:
        raise JuniperConfigError(
            f"normalize_text: unicode_form must be one of {sorted(_VALID_UNICODE_FORMS)}, "
            f"got {config.unicode_form!r}"
        )
    unicode_form: _UnicodeForm = config.unicode_form  # type: ignore[assignment]
    normalized = unicodedata.normalize(unicode_form, text)
    if config.reject_control_characters and _CONTROL_CHAR_RE.search(normalized):
        raise JuniperConfigError(f"normalize_text: control character(s) present in {normalized!r}")
    if config.collapse_repeated_whitespace:
        normalized = _WHITESPACE_RE.sub(" ", normalized)
        normalized = _MULTI_NEWLINE_RE.sub("\n\n", normalized)
    if config.strip_surrounding_whitespace:
        normalized = normalized.strip()
    if not normalized:
        raise JuniperConfigError("normalize_text: text is empty after normalization")
    return normalized
