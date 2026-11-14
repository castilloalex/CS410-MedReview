"""Shared, deterministic text normalization (no model downloads required)."""
import html
import re
import unicodedata

TOKENIZER_VERSION = "unicode-word-apostrophe-v1"
TOKEN = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)
TAG = re.compile(r"</?[A-Za-z][^>]*>")


def clean_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    for _ in range(3):
        decoded = html.unescape(value)
        if decoded == value:
            break
        value = decoded
    value = TAG.sub(" ", value)
    value = "".join(c if not unicodedata.category(c).startswith("C") or c.isspace() else " " for c in value)
    value = " ".join(value.split())
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        value = value[1:-1].strip()
    return value


def tokenize(text: str) -> list[str]:
    # Keep stopwords, numbers and negation; preserve readable text separately.
    return TOKEN.findall(unicodedata.normalize("NFKC", text).casefold().replace("’", "'"))


def duplicate_key(text: str) -> str:
    return text.casefold().replace("’", "'")
