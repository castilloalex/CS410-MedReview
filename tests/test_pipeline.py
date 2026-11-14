import csv
import io
import json
import math
import sqlite3
import tempfile
import unittest
import zipfile
from pathlib import Path

from medreview.index import SearchIndex, build_index
from medreview.prepare import MEMBERS, prepare
from medreview.text import clean_text, tokenize


class PipelineTests(unittest.TestCase):
    def test_text_preserves_clinically_useful_content(self):
        text = clean_text('"<p>I don&#039;t feel groggy!</p>\nNo nausea at 25mg &amp; sleep is good."')
        self.assertEqual(text, "I don't feel groggy! No nausea at 25mg & sleep is good.")
        self.assertEqual(tokenize("DON’T have nausea at 25mg"), ["don't", "have", "nausea", "at", "25mg"])
        self.assertEqual(clean_text("dose < 5 mg"), "dose < 5 mg")
        self.assertEqual(clean_text(clean_text(text)), text)

    def test_clean_build_and_search(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            archive_path = root / "source.zip"
            # Review 2 leaks from test; review 4 duplicates train; 5 is malformed;
            # review 7 leaks across drug names. Missing conditions remain searchable.
            rows = {
                "test": [[1, "Trazodone", "Insomnia", '"sleep sleep groggy"', 8, "January 1, 2017", 2]],
                "train": [[2, "Trazodone", "Insomnia", "SLEEP SLEEP GROGGY", 8, "January 1, 2017", 2],
                          [3, "Trazodone", "2</span> users found this comment useful.", "sleep nausea", 4, "January 2, 2017", 0],
                          [4, "Trazodone", "", "sleep nausea", 4, "January 2, 2017", 0],
                          [5, "Other", "", "bad", 12, "January 2, 2017", 0],
                          [6, "Other", "", "groggy", 2, "invalid", 0],
                          [7, "Other", "", "sleep sleep groggy", 2, "January 2, 2017", 0]]}
            with zipfile.ZipFile(archive_path, "w") as archive:
                for split, member in MEMBERS.items():
                    buffer = io.StringIO(newline="")
                    writer = csv.writer(buffer, delimiter="\t")
                    writer.writerow(["", "drugName", "condition", "review", "rating", "date", "usefulCount"])
                    writer.writerows(rows[split])
                    archive.writestr(member, buffer.getvalue())
            processed = root / "processed"
            report = prepare(archive_path, processed)
            self.assertEqual(report["counts"]["input"], 7)
            self.assertEqual(report["counts"]["retained"], 3)
            self.assertEqual(report["counts"]["excluded_train_test_text_overlap"], 2)
            self.assertEqual(report["counts"]["excluded_duplicate_drug_review"], 1)
            train = [json.loads(line) for line in (processed / "train.jsonl").read_text().splitlines()]
            self.assertIsNone(train[0]["condition"])
            self.assertIsNone(train[1]["date"])
            self.assertEqual(train[0]["date"], "2017-01-02")
            self.assertEqual(report, prepare(archive_path, processed))
            path = root / "reviews.sqlite"
            metadata = build_index(processed, path)
            self.assertEqual(metadata["tokens"], 6)
            self.assertEqual(metadata["postings"], 5)
            with SearchIndex(path) as index:
                self.assertEqual(index.term_statistics("sleep")["df"], 2)
                self.assertEqual(index.term_statistics("sleep")["cf"], 3)
                self.assertEqual(index.document_terms(1), {"sleep": 2, "groggy": 1})
                self.assertEqual(index.postings("sleep"), [{"doc_id": 1, "tf": 2}, {"doc_id": 3, "tf": 1}])
                hits = index.search("sleep", drug="TRAZODONE")
                self.assertEqual([hit["doc_id"] for hit in hits], [1, 3])
                expected = math.log1p(1.5 / 2.5) * 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 3 / 2))
                self.assertAlmostEqual(hits[0]["score"], expected)
                self.assertEqual([r["doc_id"] for r in index.search("groggy", drug="Other")], [6])
                self.assertEqual(index.search(""), [])
                self.assertEqual(index.search("notinvocabulary"), [])
                self.assertEqual(index.search("sleep", drug="nonexistent"), [])
                self.assertIsNone(index.document(100))
                with self.assertRaises(ValueError):
                    index.search("sleep", limit=0)
            # Train-only index supports development without indexing held-out text.
            build_index(processed, path, splits=("train",))
            with SearchIndex(path) as index:
                self.assertEqual(index.metadata["documents"], 2)
                self.assertIsNone(index.document(1))
            with self.assertRaises(ValueError):
                build_index(processed, path, splits=("train", "train"))

    def test_missing_index_does_not_create_database(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "missing.sqlite"
            with self.assertRaises(sqlite3.OperationalError):
                SearchIndex(path)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
