"""Portable disk-backed inverted index, with a small BM25 smoke-test search API."""
import heapq
import json
import math
import os
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from .prepare import sha256
from .text import TOKENIZER_VERSION, clean_text, tokenize

SCHEMA = """
CREATE TABLE documents (
    doc_id INTEGER PRIMARY KEY, split TEXT NOT NULL, drug_name TEXT NOT NULL,
    drug_key TEXT NOT NULL, condition TEXT, review TEXT NOT NULL,
    rating INTEGER NOT NULL, date TEXT, useful_count INTEGER NOT NULL,
    length INTEGER NOT NULL, source_file TEXT NOT NULL, source_row INTEGER NOT NULL
);
CREATE TABLE terms (term_id INTEGER PRIMARY KEY, term TEXT NOT NULL UNIQUE,
    df INTEGER NOT NULL, cf INTEGER NOT NULL);
CREATE TABLE postings (term_id INTEGER NOT NULL, doc_id INTEGER NOT NULL,
    tf INTEGER NOT NULL, PRIMARY KEY(term_id, doc_id)) WITHOUT ROWID;
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def build_index(processed_dir, index_path, splits=("train", "test")):
    if not splits or len(set(splits)) != len(splits) or set(splits) - {"train", "test"}:
        raise ValueError("Choose distinct train/test splits")
    processed_dir, index_path = Path(processed_dir), Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_suffix(".building")
    temporary.unlink(missing_ok=True)
    vocabulary = {}
    df, cf = Counter(), Counter()
    total_tokens = n = 0
    connection = sqlite3.connect(temporary)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.executescript(SCHEMA)
        for split in splits:
            with (processed_dir / f"{split}.jsonl").open(encoding="utf-8") as source:
                for line in source:
                    row = json.loads(line)
                    if row["split"] != split:
                        raise ValueError("Record split disagrees with input file")
                    frequencies = Counter(tokenize(row["review"]))
                    length = sum(frequencies.values())
                    connection.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
                        row["review_id"], split, row["drug_name"], row["drug_key"], row["condition"],
                        row["review"], row["rating"], row["date"], row["useful_count"], length,
                        row["source_file"], row["source_row"]))
                    postings = []
                    for term, frequency in frequencies.items():
                        if term not in vocabulary:
                            vocabulary[term] = len(vocabulary) + 1
                        term_id = vocabulary[term]
                        df[term_id] += 1
                        cf[term_id] += frequency
                        postings.append((term_id, row["review_id"], frequency))
                    connection.executemany("INSERT INTO postings VALUES (?,?,?)", postings)
                    n += 1
                    total_tokens += length
                    if n % 25000 == 0:
                        connection.commit()
                        print(f"Indexed {n:,} reviews", flush=True)
        connection.executemany("INSERT INTO terms VALUES (?,?,?,?)", ((i, term, df[i], cf[i]) for term, i in vocabulary.items()))
        connection.execute("CREATE INDEX postings_by_doc ON postings(doc_id)")
        connection.execute("CREATE INDEX documents_by_drug ON documents(drug_key)")
        connection.execute("CREATE INDEX documents_by_split ON documents(split)")
        metadata = {"schema_version": 1, "tokenizer": TOKENIZER_VERSION, "indexed_field": "review",
                    "splits": list(splits), "documents": n, "terms": len(vocabulary),
                    "postings": sum(df.values()), "tokens": total_tokens,
                    "average_document_length": total_tokens / n if n else 0,
                    "input_sha256": {f"{s}.jsonl": sha256(processed_dir / f"{s}.jsonl") for s in splits}}
        connection.executemany("INSERT INTO metadata VALUES (?,?)", ((key, json.dumps(value)) for key, value in metadata.items()))
        connection.commit()
        connection.execute("ANALYZE")
        connection.commit()
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Index integrity check failed")
    finally:
        connection.close()
    os.replace(temporary, index_path)
    index_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


class SearchIndex:
    def __init__(self, path):
        self.connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
        self.connection.row_factory = sqlite3.Row
        self.metadata = {r[0]: json.loads(r[1]) for r in self.connection.execute("SELECT key,value FROM metadata")}
        if self.metadata["tokenizer"] != TOKENIZER_VERSION:
            self.close()
            raise ValueError("Index tokenizer does not match this code; rebuild the index")

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def document(self, doc_id):
        row = self.connection.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def term_statistics(self, term):
        row = self.connection.execute("SELECT * FROM terms WHERE term=?", (term,)).fetchone()
        return dict(row) if row else None

    def postings(self, term):
        return [dict(row) for row in self.connection.execute(
            "SELECT p.doc_id,p.tf FROM postings p JOIN terms t USING(term_id) WHERE t.term=? ORDER BY p.doc_id", (term,))]

    def document_terms(self, doc_id):
        return dict(self.connection.execute(
            "SELECT t.term,p.tf FROM postings p JOIN terms t USING(term_id) WHERE p.doc_id=?", (doc_id,)))

    def search(self, query, drug=None, limit=10, k1=1.2, b=0.75):
        if limit < 1 or not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("Require limit >= 1, finite k1 > 0, and 0 <= b <= 1")
        scores = defaultdict(float)
        n, average = self.metadata["documents"], self.metadata["average_document_length"]
        if not n or not average:
            return []
        for term in sorted(set(tokenize(query))):
            info = self.term_statistics(term)
            if not info:
                continue
            idf = math.log1p((n - info["df"] + 0.5) / (info["df"] + 0.5))
            sql = "SELECT p.doc_id,p.tf,d.length FROM postings p JOIN documents d USING(doc_id) WHERE p.term_id=?"
            args = [info["term_id"]]
            if drug is not None:
                sql += " AND d.drug_key=?"
                args.append(clean_text(drug).casefold())
            for row in self.connection.execute(sql, args):
                tf = row["tf"]
                scores[row["doc_id"]] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * row["length"] / average))
        best = heapq.nsmallest(limit, scores, key=lambda doc_id: (-scores[doc_id], doc_id))
        return [dict(self.document(doc_id), score=scores[doc_id]) for doc_id in best]
