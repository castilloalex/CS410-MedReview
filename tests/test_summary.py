# AI-assisted code
# Model: Opus 5.5 (max effort)
import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from medreview.aspects import build_side_effect_table, extract_side_effects
from medreview.index import SearchIndex, build_index
from medreview.rm3 import search_rm3
from medreview.summary import (DrugCatalog, DrugView, analyze_reviews, build_profile, compare_drugs, parse_query,
                               retrieve, summarize, summarize_question)
from medreview.summary import render
from medreview.tfidf import search_tfidf

REVIEWS = [  # (drug, condition, review, rating, useful_count)
    ("Trazodone", "Insomnia", "It helped me sleep through the night. But I felt groggy all morning.", 8, 10),
    ("Trazodone", "Insomnia", "Works great for sleep. No grogginess at all.", 10, 30),
    ("Trazodone", "Insomnia", "Did nothing for my insomnia. It gave me a headache.", 2, 5),
    ("Trazodone", "Insomnia", "I take 50 mg at bedtime. I had no side effects.", 9, 1),
    ("Trazodone", "Insomnia", "Helps me sleep but I am very drowsy the next day and gained 10 pounds.", 7, 2),
    ("Sertraline", "Anxiety", "It really helped my anxiety. It made me so nauseous for two weeks.", 7, 4),
    ("Sertraline", "Depression", "Sertraline did not help my depression at all. Terrible nausea and headaches.", 1, 8),
    ("Sertraline", "Anxiety", "My anxiety is gone. No nausea.", 10, 3),
    ("RID", "Head Lice", "It got rid of the lice in one treatment, under 5 minutes & <10 dollars.", 9, 0),
]


class StubSentiment:
    """Positive when the text says it helped or worked."""
    def predict(self, texts):
        words = ("helped", "works", "helps", "gone")
        return ["positive" if any(w in t.lower() for w in words) else "negative" for t in texts]


def write_dataset(root: Path):
    processed = root / "processed"
    processed.mkdir(parents=True)
    rows = []
    for i, (drug, condition, review, rating, useful) in enumerate(REVIEWS, 1):
        rows.append({"review_id": i, "split": "train" if i % 3 else "test", "drug_name": drug,
                     "drug_key": drug.casefold(), "condition": condition, "review": review, "rating": rating,
                     "date": "2016-05-02", "useful_count": useful, "source_file": "x.tsv", "source_row": i + 1})
    for split in ("train", "test"):
        with (processed / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                if row["split"] == split:
                    f.write(json.dumps(row) + "\n")
    build_index(processed, root / "index" / "reviews.sqlite")
    df = pd.DataFrame(rows)
    (root / "aspects").mkdir()
    table = build_side_effect_table(df)
    table.save(root / "aspects" / "side_effects_by_drug.csv")
    return df, table


class SummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.df, cls.table = write_dataset(cls.root)
        cls.index = SearchIndex(cls.root / "index" / "reviews.sqlite")
        cls.catalog = DrugCatalog.from_index(cls.index, min_reviews=0)

    @classmethod
    def tearDownClass(cls):
        cls.index.close()
        cls.tmp.cleanup()

    def test_parse_query(self):
        q = parse_query("Does trazodone help with sleep and does it make you feel groggy?", self.catalog)
        self.assertEqual(q.drug, "Trazodone")
        self.assertTrue(q.drug_found)
        self.assertEqual(q.keywords, ["help", "sleep", "groggy"])
        self.assertEqual(q.side_effects, ["drowsiness"])
        q = parse_query("Does it help with insomnia, or cause nausea?", self.catalog)
        self.assertIsNone(q.drug)
        self.assertEqual((q.treated, q.side_effects), (["insomnia"], ["nausea"]))
        q = parse_query("does it cause nausea", self.catalog, drug="sertraline")
        self.assertEqual((q.drug, q.drug_found, q.keywords), ("Sertraline", False, ["nausea"]))
        q = parse_query("Is sertraline better than trazodone?", self.catalog, drug="Trazodone")
        self.assertEqual((q.drug, q.other_drugs, q.keywords), ("Trazodone", ["Sertraline"], ["sertraline", "better"]))

    def test_catalog_matching(self):
        catalog = DrugCatalog({"Plan B": 637, "Plan B One-Step": 433, "RID": 1, "Zoloft": 857, "Lexapro": 996,
                               "My Way": 19}, common_terms={"plan", "b", "one", "step", "rid", "my", "way"})
        names = lambda text: [m.drug.name for m in catalog.find(text)]
        self.assertEqual(names("does plan-b one step cause cramps"), ["Plan B One-Step"])
        self.assertEqual(names("does plan b cause cramps"), ["Plan B"])
        self.assertEqual(names("how do I get rid of acne my way"), [])          # everyday words
        self.assertEqual(names("Is RID safe? It worked My Way"), ["RID", "My Way"])
        self.assertEqual(names("Is Zoloft better than lexapro?"), ["Zoloft", "Lexapro"])
        self.assertEqual(catalog.get("zoloft").reviews, 857)

    def test_drug_view_matches_full_index(self):
        for query, drug in [("sleep groggy", "Trazodone"), ("nausea", "Sertraline"), ("help my anxiety", "Sertraline")]:
            view = DrugView(self.index, drug)
            self.assertEqual(search_tfidf(self.index, query, drug=drug), search_tfidf(view, query, drug=drug))
            self.assertEqual(search_rm3(self.index, query, drug=drug, feedback_docs=2, expansion_terms=5),
                             search_rm3(view, query, drug=drug, feedback_docs=2, expansion_terms=5))
            for method in ("bm25", "tfidf", "rm3"):
                hits = retrieve(self.index, query, drug=drug, method=method, limit=3)
                self.assertTrue(hits and all(h["drug_name"] == drug for h in hits), (method, query))
        self.assertEqual(retrieve(self.index, "  ", drug="Trazodone"), [])
        with self.assertRaises(ValueError):
            retrieve(self.index, "sleep", method="lsi")

    def test_analyze_reviews(self):
        docs = [dict(r) for r in self.index.connection.execute("SELECT * FROM documents ORDER BY doc_id")]
        analyzed = analyze_reviews(docs, StubSentiment())
        by_id = {r.doc_id: r for r in analyzed}
        for r in analyzed:  # same rule as the side-effect table
            self.assertEqual(r.side_effects, extract_side_effects(r.review, r.condition), r.review)
        self.assertEqual(by_id[1].side_effects, ["drowsiness"])
        self.assertEqual((by_id[2].side_effects, by_id[2].denied), ([], ["drowsiness"]))
        self.assertEqual(by_id[5].side_effects, ["drowsiness", "weight gain"])
        self.assertTrue(by_id[4].no_side_effects)
        self.assertFalse(by_id[3].no_side_effects)
        self.assertEqual([by_id[i].effectiveness_sentiment for i in (1, 2, 3, 4, 5)],
                         ["positive", "positive", "negative", None, None])
        self.assertEqual(by_id[2].sentences[by_id[2].denied_in["drowsiness"]][0], "No grogginess at all.")
        self.assertIsNone(analyze_reviews(docs[:1])[0].effectiveness_sentiment)

    def test_summary(self):
        parsed = parse_query("Does trazodone help with sleep and does it make you feel groggy?", self.catalog)
        profile = build_profile(self.index, "trazodone", StubSentiment())
        self.assertEqual((profile.drug, profile.n), ("Trazodone", 5))
        hits = retrieve(self.index, parsed.search_text, drug=parsed.drug)
        s = summarize(profile, hits, parsed, self.table)
        self.assertTrue(s.searched)
        self.assertEqual([r.doc_id for r in s.reviews], [h["doc_id"] for h in hits])
        self.assertEqual(sorted(r.doc_id for r in s.reviews), [1, 2, 5])
        self.assertEqual((s.effectiveness.positive, s.effectiveness.negative), (2, 0))
        self.assertEqual((s.baseline.positive, s.baseline.negative, s.baseline.total), (2, 1, 3))
        concern, = s.concerns
        self.assertEqual((concern.name, concern.reported, concern.denied, concern.reviews), ("drowsiness", 2, 1, 5))
        self.assertEqual((concern.matched_reported, concern.matched_denied), (2, 1))
        self.assertAlmostEqual(concern.lift, 0.4 / (2 / 9), places=2)
        self.assertEqual(concern.denied_examples[0][1], "No grogginess at all.")
        top = s.side_effects[0]
        self.assertEqual((top.name, top.reported, top.asked, top.drug_reported), ("drowsiness", 2, True, 2))
        self.assertAlmostEqual(top.drug_share, 0.4)
        self.assertEqual(self.table.top_list("Trazodone", k=3, min_reviews=1)[0], ("drowsiness", 0.4))
        # No search words: the most useful reviews stand in for search results.
        s = summarize(profile, None, parse_query("trazodone", self.catalog), self.table, limit=2)
        self.assertFalse(s.searched)
        self.assertEqual([r.doc_id for r in s.reviews], [2, 1])
        # Condition filter keeps only that condition's reviews.
        sertraline = build_profile(self.index, "Sertraline", StubSentiment())
        self.assertEqual(sertraline.for_condition("anxiety").n, 2)
        self.assertEqual(sertraline.conditions(), [("Anxiety", 2), ("Depression", 1)])

    def test_summarize_question_and_compare(self):
        parsed, s = summarize_question(self.index, "Does sertraline cause nausea?", self.table, StubSentiment(),
                                       catalog=self.catalog, method="rm3")
        self.assertEqual((s.drug, s.method), ("Sertraline", "rm3"))
        self.assertEqual(s.concerns[0].reported, 2)       # "nauseous" and "nausea"; "No nausea" is denied
        self.assertEqual(s.concerns[0].denied, 1)
        parsed, s = summarize_question(self.index, "nausea", self.table, StubSentiment(), catalog=self.catalog,
                                       drug="Sertraline", condition="Anxiety")
        self.assertEqual((s.profile.n, [r.doc_id for r in s.reviews]), (2, [8]))
        parsed, rows = summarize_question(self.index, "nausea headaches", self.table, StubSentiment(),
                                          catalog=self.catalog)
        self.assertIsNone(parsed.drug)
        self.assertEqual([(d.drug, d.matched) for d in rows], [("Sertraline", 2)])
        self.assertEqual(set(rows[0].concern_shares), {"nausea", "headache"})
        self.assertAlmostEqual(rows[0].concern_shares["nausea"], 2 / 3, places=3)    # the table rounds shares
        self.assertAlmostEqual(rows[0].concern_shares["headache"], 1 / 3, places=3)
        hits = retrieve(self.index, "sleep nausea", limit=10)
        rows = compare_drugs(analyze_reviews(hits), parse_query("sleep nausea", self.catalog), self.table)
        self.assertEqual([d.drug for d in rows], ["Trazodone", "Sertraline"])

    def test_render_escapes_and_marks(self):
        rid = analyze_reviews([dict(self.index.document(9))])[0]
        out = render.review_html(rid, render.term_pattern(["lice"]))
        self.assertIn("&amp; &lt;10 dollars", out)
        self.assertIn("<b>lice</b>", out)
        self.assertEqual(render.mark('<script>alert("x")</script> sleep', render.term_pattern(["sleep"])),
                         "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; <b>sleep</b>")
        pattern = render.term_pattern(["sleep"], ["drowsiness"])
        self.assertEqual(render.mark("Groggy & sleepy, slept", pattern), "<b>Groggy</b> &amp; <b>sleepy</b>, slept")
        self.assertIsNone(render.term_pattern())
        counts = analyze_reviews([dict(self.index.document(i)) for i in (1, 2, 3)], StubSentiment())
        from medreview.summary import SentimentCounts
        bar = render.sentiment_bar(SentimentCounts.of(counts), "Effectiveness")
        self.assertIn('aria-label="Effectiveness: 67% positive, 0% mixed or neutral, 33% negative"', bar)
        self.assertEqual(render.sentiment_bar(SentimentCounts(), "x"), "")


class AppTests(unittest.TestCase):
    """Runs app.py headlessly on a small dataset (skipped without Streamlit).

    Each case is a fresh single run with its state set up front, which also works with
    the test harness of older Streamlit versions."""

    def run_app(self, **state):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=60)
        for key, value in state.items():
            at.session_state[key] = value
        at.run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        return [h.value for h in at.subheader]

    def test_app_runs(self):
        try:
            import streamlit.testing.v1  # noqa: F401
        except ImportError:
            self.skipTest("streamlit is not installed")
        with tempfile.TemporaryDirectory() as d:
            write_dataset(Path(d))
            old = os.environ.get("MEDREVIEW_DATA")
            os.environ["MEDREVIEW_DATA"] = d
            try:
                headers = self.run_app(question="Does Trazodone help with sleep and does it make you feel groggy?")
                self.assertIn("Does it work?", headers)
                self.assertIn("Does it cause drowsiness?", headers)
                headers = self.run_app(question="nausea headaches")
                self.assertIn("Drugs whose reviews best match your question", headers)
                headers = self.run_app(question="nausea", drug_choice="Sertraline", method="rm3")
                self.assertIn("Does it cause nausea?", headers)
                headers = self.run_app(question="", drug_choice="Trazodone", method="tfidf")
                self.assertIn("The 5 most useful reviews", headers)
            finally:
                if old is None:
                    os.environ.pop("MEDREVIEW_DATA", None)
                else:
                    os.environ["MEDREVIEW_DATA"] = old


if __name__ == "__main__":
    unittest.main()
