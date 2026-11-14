# AI-assisted code
# Model: Fable 5.2 (medium effort)
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from medreview.aspects import (KeywordAspectClassifier, LogRegAspectClassifier, build_side_effect_table,
                               classify_review, extract_side_effects, find_side_effect_mentions, split_sentences)


class AspectTests(unittest.TestCase):
    def setUp(self):
        self.kw = KeywordAspectClassifier()

    def test_split_sentences(self):
        text = "I've taken it for 2 weeks. It works great!\r\nNo complaints. Dr. Lee agreed."
        self.assertEqual(split_sentences(text),
                         ["I've taken it for 2 weeks.", "It works great!", "No complaints.", "Dr. Lee agreed."])
        self.assertEqual(split_sentences(None), [])

    def test_keyword_labels(self):
        cases = {
            "This medication really helped my pain.": "effectiveness",
            "It made me so dizzy and nauseous.": "side_effects",
            "I have been on it for three months.": "neither",
            "The only side effect was dry mouth.": "side_effects",
            "It has even taken away my anxiety and depression.": "effectiveness",
            "I work retail at a large store.": "neither",
            "I had no nausea at all.": "side_effects",  # still about side effects
        }
        for sentence, label in cases.items():
            self.assertEqual(self.kw.predict_one(sentence), label, sentence)
        self.assertEqual(self.kw.predict_one("My anxiety is gone.", "Anxiety"), "effectiveness")

    def test_mentions_skip_negated_and_treated(self):
        self.assertEqual(find_side_effect_mentions("I had no nausea at all."), [])
        self.assertEqual(find_side_effect_mentions("It gave me terrible headaches."), ["headache"])
        self.assertEqual(find_side_effect_mentions("It gave me anxiety.", condition="Anxiety"), [])
        self.assertEqual(find_side_effect_mentions("It helps with my nausea, anxiety and insomnia."), [])
        self.assertEqual(find_side_effect_mentions("It gave me my first migraine."), ["headache"])

    def test_extract_and_table(self):
        review = "It worked well for my pain. But it made me very drowsy and I gained 10 pounds."
        self.assertEqual(extract_side_effects(review), ["drowsiness", "weight gain"])
        df = pd.DataFrame({"drug_name": ["A"] * 4 + ["B"], "condition": ["Pain"] * 5,
                           "review": [review, review, review, "Works fine.", "It gave me a headache."]})
        table = build_side_effect_table(df)
        top = table.top("a", min_reviews=1)
        self.assertEqual(set(top["side_effect"]), {"drowsiness", "weight gain"})
        self.assertEqual(top["pct"].iloc[0], 0.75)
        self.assertEqual(table.top_list("B", min_reviews=1), [("headache", 1.0)])
        self.assertEqual(table.top_list("unknown drug"), [])
        with tempfile.TemporaryDirectory() as d:
            table.save(Path(d) / "t.csv")
            self.assertEqual(len(type(table).load(Path(d) / "t.csv").df), len(table.df))

    def test_logreg_roundtrip(self):
        X = ["It works great", "helped my pain a lot", "made me dizzy", "caused nausea",
             "I took it for a week", "my doctor prescribed it"] * 3
        y = ["effectiveness", "effectiveness", "side_effects", "side_effects", "neither", "neither"] * 3
        model = LogRegAspectClassifier().fit(X, y)
        with tempfile.TemporaryDirectory() as d:
            model.save(Path(d) / "m.joblib")
            loaded = LogRegAspectClassifier.load(Path(d) / "m.joblib")
        self.assertEqual(loaded.predict(["it made me dizzy"]), ["side_effects"])
        self.assertEqual(classify_review("It works great. It made me dizzy.", loaded)[1][1], "side_effects")


if __name__ == "__main__":
    unittest.main()
