"""Aspect classification and side-effect extraction for drug reviews.

    from medreview.aspects import classify_review, extract_side_effects, SideEffectTable
"""
from .sentences import split_sentences
from .keywords import KeywordAspectClassifier, LABELS, EFFECTIVENESS, SIDE_EFFECTS, NEITHER
from .logreg import LogRegAspectClassifier
from .side_effects import (
    extract_side_effects, find_side_effect_mentions, SideEffectTable, build_side_effect_table,
)
from .api import classify_review
from .data import load_reviews

__all__ = [
    "split_sentences", "KeywordAspectClassifier", "LogRegAspectClassifier", "LABELS",
    "EFFECTIVENESS", "SIDE_EFFECTS", "NEITHER", "extract_side_effects", "find_side_effect_mentions",
    "SideEffectTable", "build_side_effect_table", "classify_review", "load_reviews",
]
