"""VADER baseline: a sentiment lexicon with rules, no training."""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .labels import NEGATIVE, NEUTRAL, POSITIVE


class VaderSentimentClassifier:
    """Label a text by VADER's compound score (-1..1)."""

    def __init__(self, pos_threshold: float = 0.05, neg_threshold: float = -0.05):
        self.analyzer = SentimentIntensityAnalyzer()
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

    def score(self, text: str) -> float:
        return self.analyzer.polarity_scores(text or "")["compound"]

    def label(self, compound: float) -> str:
        if compound >= self.pos_threshold:
            return POSITIVE
        if compound <= self.neg_threshold:
            return NEGATIVE
        return NEUTRAL

    def predict(self, texts) -> list[str]:
        return [self.label(self.score(t)) for t in texts]

    def predict_one(self, text: str) -> str:
        return self.label(self.score(text))
