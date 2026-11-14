# AI-assisted code
# Model: Fable 5.2 (medium effort)
"""Logistic-regression aspect classifier (TF-IDF word 1-2grams + char n-grams)."""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion


def make_pipeline(C: float = 4.0, min_df: int = 1) -> Pipeline:
    feats = FeatureUnion([
        ("word", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)),
        ("char", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5),
                                 min_df=min_df, sublinear_tf=True)),
    ])
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    return Pipeline([("feats", feats), ("clf", clf)])


class LogRegAspectClassifier:
    def __init__(self, C: float = 4.0, min_df: int = 1):
        self.pipeline = make_pipeline(C=C, min_df=min_df)

    def fit(self, sentences, labels):
        self.pipeline.fit(list(sentences), list(labels))
        return self

    def predict(self, sentences, conditions=None) -> list[str]:
        """`conditions` is accepted for interface parity with the keyword classifier and ignored."""
        return list(self.pipeline.predict(list(sentences)))

    def predict_one(self, sentence: str, condition=None) -> str:
        return self.predict([sentence])[0]

    def predict_proba(self, sentences):
        return self.pipeline.predict_proba(list(sentences))

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
