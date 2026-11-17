# AI-assisted code
# Model: Opus 5.5 (max effort)
"""Checks for the summaries. Run from the MedReview folder after `python -m medreview all`.

    python -m medreview.summary.evaluate                 # all three checks, about 10 minutes
    python -m medreview.summary.evaluate --part table    # one check

retrieval  DrugView gives search_tfidf / search_rm3 exactly the results they return on
           the full index, for the 25 judged queries, and how much faster.
table      Drug-wide side-effect counts from the summaries equal side_effects_by_drug.csv
           for every drug.
sentiment  Test split only: reading just the effectiveness sentences vs the whole review
           (logistic regression model from `python -m medreview.sentiment train`).

Writes reports/summary_checks.json.
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

from ..aspects import SideEffectTable
from ..index import SearchIndex
from ..rm3 import search_rm3
from ..tfidf import search_tfidf
from .analysis import analyze_reviews
from .pipeline import SENTIMENT_MODEL
from .retrieval import DrugView


def check_retrieval(index, limit=10):
    from ..pool import QUERIES

    out = {"queries": len(QUERIES), "limit": limit}
    for name, function in (("tfidf", search_tfidf), ("rm3", search_rm3)):
        full_time = view_time = 0.0
        identical = 0
        for _, query, drug in QUERIES:
            t = time.perf_counter()
            expected = function(index, query, drug=drug, limit=limit)
            full_time += time.perf_counter() - t
            t = time.perf_counter()
            got = function(DrugView(index, drug), query, drug=drug, limit=limit)
            view_time += time.perf_counter() - t
            identical += got == expected
        out[name] = {"identical": identical, "seconds_full_index": round(full_time, 2),
                     "seconds_drug_view": round(view_time, 2)}
        print(f"{name}: {identical}/{len(QUERIES)} identical; {full_time:.1f}s on the full index, "
              f"{view_time:.2f}s with DrugView")
    return out


def check_table(index, table_path):
    table = SideEffectTable.load(table_path)
    expected = {(d, s): int(n) for d, s, n in zip(table.df["drug"], table.df["side_effect"], table.df["n_reviews"])}
    got = Counter()
    rows = index.connection.execute("SELECT doc_id, drug_name, condition, review, rating FROM documents")
    batch = []
    for row in rows:
        batch.append(dict(row))
        if len(batch) == 5000:
            got.update((r.drug_name, name) for r in analyze_reviews(batch) for name in r.side_effects)
            batch = []
    got.update((r.drug_name, name) for r in analyze_reviews(batch) for name in r.side_effects)
    mismatches = [k for k in set(expected) | set(got) if expected.get(k, 0) != got.get(k, 0)]
    print(f"table: {len(expected) - len([k for k in mismatches if k in expected])}/{len(expected)} drug and "
          f"side-effect pairs match; {len(mismatches)} mismatches")
    return {"pairs": len(expected), "mismatches": len(mismatches), "examples": [list(k) for k in mismatches[:10]]}


def check_sentiment(index, model_path):
    from sklearn.metrics import accuracy_score, f1_score

    from ..sentiment import LABELS, rating_to_label
    from ..sentiment.api import default_classifier

    model = default_classifier("logreg", model_path)
    docs = [dict(r) for r in index.connection.execute(
        "SELECT doc_id, drug_name, condition, review, rating FROM documents WHERE split='test'")]
    reviews = analyze_reviews(docs, model)
    whole = model.predict([r.review for r in reviews])
    gold = [rating_to_label(r.rating) for r in reviews]
    has_eff = [i for i, r in enumerate(reviews) if r.effectiveness_sentiment]
    eff = {i: reviews[i].effectiveness_sentiment for i in has_eff}

    def share(items, label):
        return round(sum(x == label for x in items) / len(items), 3) if items else None

    with_se = [i for i in has_eff if reviews[i].side_effects]
    low = [i for i in with_se if gold[i] == "negative"]
    out = {
        "test_reviews": len(reviews),
        "with_effectiveness_sentence": len(has_eff),
        "whole_review_vs_rating": {"accuracy": round(accuracy_score(gold, whole), 3),
                                   "macro_f1": round(f1_score(gold, whole, labels=LABELS, average="macro"), 3)},
        "effectiveness_only_vs_rating": {
            "accuracy": round(accuracy_score([gold[i] for i in has_eff], [eff[i] for i in has_eff]), 3),
            "macro_f1": round(f1_score([gold[i] for i in has_eff], [eff[i] for i in has_eff], labels=LABELS,
                                       average="macro"), 3)},
        "reporting_side_effects": {
            "reviews": len(with_se),
            "positive_share_whole_review": share([whole[i] for i in with_se], "positive"),
            "positive_share_effectiveness_only": share([eff[i] for i in with_se], "positive"),
            "rated_1_to_4": len(low),
            "rated_1_to_4_but_effectiveness_positive": sum(eff[i] == "positive" for i in low),
        },
        "no_side_effects_claims": sum(r.no_side_effects for r in reviews),
    }
    s = out["reporting_side_effects"]
    print(f"sentiment: {len(has_eff):,} of {len(reviews):,} test reviews have an effectiveness sentence")
    print(f"  vs rating labels: whole review {out['whole_review_vs_rating']}, "
          f"effectiveness only {out['effectiveness_only_vs_rating']}")
    print(f"  {s['reviews']:,} report a side effect: positive {s['positive_share_whole_review']:.1%} reading the whole "
          f"review, {s['positive_share_effectiveness_only']:.1%} reading only effectiveness sentences")
    print(f"  {s['rated_1_to_4_but_effectiveness_positive']:,} of {s['rated_1_to_4']:,} of them rated 1-4 still "
          "read positive about effectiveness")
    return out


def main():
    ap = argparse.ArgumentParser(prog="python -m medreview.summary.evaluate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["all", "retrieval", "table", "sentiment"], default="all")
    ap.add_argument("--index", type=Path, default=Path("data/index/reviews.sqlite"))
    ap.add_argument("--table", type=Path, default=Path("data/aspects/side_effects_by_drug.csv"))
    ap.add_argument("--model", type=Path, default=SENTIMENT_MODEL)
    ap.add_argument("--out", type=Path, default=Path("reports/summary_checks.json"))
    a = ap.parse_args()
    results = json.loads(a.out.read_text(encoding="utf-8")) if a.out.exists() else {}
    with SearchIndex(a.index) as index:
        if a.part in ("all", "retrieval"):
            results["retrieval"] = check_retrieval(index)
        if a.part in ("all", "table"):
            results["table"] = check_table(index, a.table)
        if a.part in ("all", "sentiment"):
            if a.model.exists():
                results["sentiment"] = check_sentiment(index, a.model)
            else:
                print(f"sentiment: skipped, {a.model} not found (run `python -m medreview.sentiment train`)")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
