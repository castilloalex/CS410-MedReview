"""Validate generated artifacts, including full-corpus count and leakage invariants.

Run: python -m medreview.validate
"""
import json
from pathlib import Path

from .index import SearchIndex
from .prepare import sha256
from .text import duplicate_key, tokenize


def validate(data_dir=Path("data")):
    data_dir = Path(data_dir)
    processed = data_dir / "processed"
    cleaning = json.loads((processed / "cleaning_report.json").read_text(encoding="utf-8"))
    checks = {}
    split_texts = {}
    all_ids = set()
    token_count = 0
    split_counts = {}
    for split in ("test", "train"):
        path = processed / f"{split}.jsonl"
        checks[f"{split}_hash_matches"] = sha256(path) == cleaning["outputs"][path.name]
        texts = set()
        count = 0
        with path.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                if row["review_id"] in all_ids or row["split"] != split:
                    raise AssertionError("Duplicate ID or invalid split")
                all_ids.add(row["review_id"])
                texts.add(duplicate_key(row["review"]))
                token_count += len(tokenize(row["review"]))
                count += 1
        split_texts[split] = texts
        split_counts[split] = count
        checks[f"{split}_count_matches"] = count == cleaning["splits"][split]["retained"]
    overlap = len(split_texts["train"] & split_texts["test"])
    checks["no_exact_text_train_test_leakage"] = overlap == 0
    excluded = sum(v for k, v in cleaning["counts"].items() if k.startswith("excluded_"))
    checks["all_input_rows_accounted_for"] = cleaning["counts"]["input"] == len(all_ids) + excluded
    with SearchIndex(data_dir / "index/reviews.sqlite") as index:
        connection = index.connection
        checks["sqlite_integrity"] = connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        checks["indexed_ids_match_corpus"] = all_ids == {r[0] for r in connection.execute("SELECT doc_id FROM documents")}
        checks["index_input_hashes_match"] = index.metadata["input_sha256"] == cleaning["outputs"]
        checks["document_count_matches"] = len(all_ids) == index.metadata["documents"]
        checks["tokens_match"] = token_count == index.metadata["tokens"] == connection.execute("SELECT SUM(tf) FROM postings").fetchone()[0]
        checks["term_statistics_match"] = connection.execute("""
            SELECT COUNT(*) FROM terms t LEFT JOIN
            (SELECT term_id, COUNT(*) df, SUM(tf) cf FROM postings GROUP BY term_id) p
            USING(term_id) WHERE p.df IS NULL OR t.df != p.df OR t.cf != p.cf
        """).fetchone()[0] == 0
        checks["document_lengths_match"] = connection.execute("""
            SELECT COUNT(*) FROM documents d LEFT JOIN
            (SELECT doc_id, SUM(tf) length FROM postings GROUP BY doc_id) p USING(doc_id)
            WHERE d.length != COALESCE(p.length, 0)
        """).fetchone()[0] == 0
        hits = index.search("sleep groggy", drug="Trazodone", limit=5)
        checks["drug_filtered_smoke_search"] = len(hits) == 5 and all(h["drug_key"] == "trazodone" for h in hits)
        for hit in hits:
            checks["sample_term_vectors_match"] = checks.get("sample_term_vectors_match", True) and sum(index.document_terms(hit["doc_id"]).values()) == hit["length"]
        report = {"cleaning": cleaning, "index": index.metadata, "checks": checks,
                  "train_test_text_overlap": overlap, "split_counts": split_counts,
                  "smoke_query": {"query": "sleep groggy", "drug": "Trazodone",
                                  "top_results": [{k: h[k] for k in ("doc_id", "drug_name", "score")} for h in hits]}}
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise AssertionError(f"Validation failed: {failures}")
    return report


if __name__ == "__main__":
    result = validate()
    destination = Path("reports/part1_results.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
