"""Logistic-regression sentiment classifier (TF-IDF word 1-2 grams)."""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def make_pipeline(C: float = 1.0, min_df: int = 2) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)),
        ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced")),
    ])


class LogRegSentimentClassifier:
    def __init__(self, C: float = 1.0, min_df: int = 2):
        self.pipeline = make_pipeline(C=C, min_df=min_df)

    def fit(self, texts, labels):
        self.pipeline.fit(list(texts), list(labels))
        return self

    def predict(self, texts) -> list[str]:
        return list(self.pipeline.predict(list(texts)))

    def predict_one(self, text: str) -> str:
        return self.predict([text])[0]

    def predict_proba(self, texts):
        return self.pipeline.predict_proba(list(texts))

    @property
    def classes_(self):
        return list(self.pipeline.classes_)

    def save(self, path):
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path):
        obj = cls()
        obj.pipeline = joblib.load(path)
        return obj
