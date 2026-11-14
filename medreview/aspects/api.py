# AI-assisted code
# Model: Fable 5.2 (medium effort)
"""Convenience helpers for the app."""
from .keywords import KeywordAspectClassifier
from .sentences import split_sentences

_KW = None


def default_classifier():
    global _KW
    if _KW is None:
        _KW = KeywordAspectClassifier()
    return _KW


def classify_review(review: str, classifier=None, condition: str | None = None) -> list[tuple[str, str]]:
    """Return [(sentence, aspect), ...]; aspect is effectiveness, side_effects or neither."""
    classifier = classifier or default_classifier()
    sents = split_sentences(review)
    return list(zip(sents, classifier.predict(sents, condition))) if sents else []
