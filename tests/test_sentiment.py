# AI-assisted code
# Model: Opus 5.5 (medium effort)
import tempfile
import unittest
from pathlib import Path

from medreview.sentiment import (LogRegSentimentClassifier, VaderSentimentClassifier, predict_sentiment,
                                 rating_to_label)


class SentimentTests(unittest.TestCase):
    def test_rating_to_label(self):
        expected = {1: "negative", 4: "negative", 5: "neutral", 6: "neutral", 7: "positive", 10: "positive"}
        for rating, label in expected.items():
            self.assertEqual(rating_to_label(rating), label, rating)
        with self.assertRaises(ValueError):
            rating_to_label(0)

    def test_vader(self):
        vader = VaderSentimentClassifier()
        self.assertEqual(vader.predict_one("This medicine is wonderful, it saved my life!"), "positive")
        self.assertEqual(vader.predict_one("Horrible. It made me sick and miserable."), "negative")
        self.assertEqual(vader.predict_one("I took it in the morning."), "neutral")
        self.assertEqual(vader.predict_one(None), "neutral")

    def test_logreg_fit_predict_save(self):
        texts = ["great drug works well", "love it helped a lot", "terrible awful side effects",
                 "worst ever made me sick", "it was okay nothing special", "mixed results so so"] * 5
        labels = ["positive", "positive", "negative", "negative", "neutral", "neutral"] * 5
        model = LogRegSentimentClassifier(min_df=1).fit(texts, labels)
        self.assertEqual(model.predict_one("great drug helped a lot"), "positive")
        self.assertEqual(model.predict_one("awful made me sick"), "negative")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.joblib"
            model.save(path)
            loaded = LogRegSentimentClassifier.load(path)
            self.assertEqual(loaded.predict(texts[:6]), model.predict(texts[:6]))
            self.assertEqual(predict_sentiment("terrible awful", classifier=loaded), "negative")


if __name__ == "__main__":
    unittest.main()
