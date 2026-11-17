"""Convenience helpers for the app."""
from pathlib import Path

from .model import LogRegSentimentClassifier
from .vader import VaderSentimentClassifier

MODEL = Path("data/sentiment/sentiment_logreg.joblib")
_CACHE = {}


def default_classifier(method: str = "logreg", model_path=MODEL):
    """Cached classifier. logreg needs `python -m medreview.sentiment train` first."""
    key = (method, str(model_path))
    if key not in _CACHE:
        if method == "logreg":
            if not Path(model_path).exists():
                raise FileNotFoundError(f"{model_path} not found; run `python -m medreview.sentiment train`.")
            _CACHE[key] = LogRegSentimentClassifier.load(model_path)
        elif method == "vader":
            _CACHE[key] = VaderSentimentClassifier()
        else:
            raise ValueError(f"unknown method {method!r}; use 'logreg' or 'vader'")
    return _CACHE[key]


def predict_sentiment(text: str, method: str = "logreg", classifier=None) -> str:
    """Return 'negative', 'neutral' or 'positive' for one review."""
    return (classifier or default_classifier(method)).predict_one(text)
