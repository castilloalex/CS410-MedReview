# AI-assisted code
# Model: Fable 5.2 (medium effort)
"""Sentence splitting (regex only; no model downloads)."""
import re

from ..text import clean_text

_ABBREV = r"(?<!\bDr\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bMrs\.)(?<!\bvs\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bapprox\.)"
# Split after . ! ? (plus closing quotes/brackets) when followed by whitespace and
# a capital letter, digit or opening quote; also split on line breaks.
_SPLIT = re.compile(r"(?<=[.!?])" + _ABBREV + r"[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])|[\r\n]+")


def split_sentences(text, min_chars: int = 3) -> list[str]:
    """Split a review into sentences. Safe on raw or already-cleaned text."""
    if not isinstance(text, str) or not text.strip():
        return []
    parts = []
    for line in re.split(r"[\r\n]+", text):
        line = clean_text(line)
        parts.extend(p.strip() for p in _SPLIT.split(line))
    return [p for p in parts if len(p) >= min_chars and re.search(r"[A-Za-z]", p)]
