# AI-assisted code
# Model: Opus 5.5 (max effort)
"""Concern-focused summaries: what one drug's reviews say about a user's question.

    from medreview.summary import summarize_question, load_sentiment
"""
from .query import STOPWORDS, Drug, DrugCatalog, DrugMatch, ParsedQuery, find_concerns, parse_query
from .retrieval import METHODS, DrugView, retrieve
from .analysis import (SENTIMENTS, Concern, DrugComparison, DrugProfile, DrugSummary, ReviewAnalysis,
                       SentimentCounts, SideEffectRow, analyze_reviews, build_profile, compare_drugs, summarize)
from .pipeline import fetch_hits, load_sentiment, summarize_question

__all__ = [
    "STOPWORDS", "Drug", "DrugCatalog", "DrugMatch", "ParsedQuery", "find_concerns", "parse_query",
    "METHODS", "DrugView", "retrieve", "SENTIMENTS", "Concern", "DrugComparison", "DrugProfile", "DrugSummary",
    "ReviewAnalysis", "SentimentCounts", "SideEffectRow", "analyze_reviews", "build_profile", "compare_drugs",
    "summarize", "fetch_hits", "load_sentiment", "summarize_question",
]
