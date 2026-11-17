# AI-assisted code
# Model: Opus 5.5 (max effort)
"""Command line for the summaries. Run from the MedReview folder.

    python -m medreview.summary "Does trazodone help with sleep and does it make you feel groggy?"
    python -m medreview.summary "nausea" --drug Sertraline --method rm3 --limit 30
    python -m medreview.summary "sleep without grogginess"      # no drug: compares drugs

Needs the index from `python -m medreview all`. Uses the logistic regression sentiment
model when `python -m medreview.sentiment train` has been run, VADER otherwise.
"""
import argparse
import textwrap
from pathlib import Path

from ..aspects import SideEffectTable
from ..index import SearchIndex
from .pipeline import SENTIMENT_MODEL, load_sentiment, summarize_question
from .retrieval import METHODS


def _pct(x):
    return f"{x:.0%}"


def _sentiment_line(counts):
    return ", ".join(f"{_pct(counts.share(k))} {k}" for k in ("positive", "neutral", "negative"))


def print_summary(parsed, s, reviews_to_show=5):
    p = s.profile
    scope = f" for {p.condition}" if p.condition else ""
    rating = f", average rating {p.average_rating:.1f}/10" if p.average_rating is not None else ""
    print(f"{s.drug}: {p.n:,} reviews{scope}{rating}")
    if s.searched:
        print(f"Searched for: {parsed.search_text}  ({METHODS[s.method]}, {len(s.reviews)} reviews)")
    else:
        print(f"No search words; using the {len(s.reviews)} most useful reviews.")
    if parsed.side_effects:
        print(f"Side effects asked about: {', '.join(parsed.side_effects)}")
    if parsed.treated:
        print(f"Asks whether it helps with: {', '.join(parsed.treated)}")
    e = s.effectiveness
    print(f"\nEffectiveness, in the {e.total} of these reviews that say whether it worked:")
    print(f"  {_sentiment_line(e)}" if e.total else "  none of these reviews discuss it")
    if s.baseline.total:
        print(f"  all {p.n:,} reviews: {_sentiment_line(s.baseline)} (of {s.baseline.total:,})")
    for c in s.concerns:
        lift = f", {c.lift:.1f}x the rate across all drugs" if c.lift else ""
        print(f"\nConcern: {c.name}")
        print(f"  reported in {c.reported:,} of {c.reviews:,} reviews ({c.share:.1%}){lift}; "
              f"{c.denied:,} say they did not have it")
        print(f"  in these {len(s.reviews)} reviews: {c.matched_reported} report it, {c.matched_denied} say they did not")
        if c.treated:
            print(f"  note: many {s.drug} reviews are for this problem, so those mentions are not counted")
        for review, sentence in c.reported_examples[:2]:
            print(textwrap.fill(f'"{sentence}"', 100, initial_indent="    ", subsequent_indent="     "))
    seen = [r for r in s.side_effects if r.reported or r.asked][:8]
    if seen:
        print(f"\nSide effects reported in these reviews ({s.no_side_effects} say they had none):")
        for r in seen:
            lift = f", {r.lift:.1f}x all drugs" if r.lift else ""
            print(f"  {r.name:20s} {r.reported:4d} ({_pct(r.share)}); all {s.drug} reviews {r.drug_share:.1%}{lift}")
    if s.common_side_effects:
        common = ", ".join(f"{name} {share:.0%}" for name, share in s.common_side_effects)
        print(f"\nMost reported for {s.drug} overall: {common}")
    print(f"\nTop {min(reviews_to_show, len(s.reviews))} reviews:")
    for rank, r in enumerate(s.reviews[:reviews_to_show], 1):
        score = f", score {r.score:.2f}" if r.score is not None else ""
        cond = f", {r.condition}" if r.condition else ""
        print(f"  {rank}. [{r.rating}/10{cond}{score}; effectiveness {r.effectiveness_sentiment or '-'}]")
        for sentence, aspect in r.sentences:
            tag = {"effectiveness": "E", "side_effects": "S"}.get(aspect, " ")
            print(textwrap.fill(sentence, 100, initial_indent=f"     {tag} ", subsequent_indent="       "))


def print_comparison(parsed, rows):
    if not parsed.keywords:
        print("Type a question or some search words.")
        return
    print(f"No drug named. Drugs whose reviews best match: {parsed.search_text}\n")
    asked = parsed.side_effects
    header = f"{'drug':32s} {'matches':>7s} {'effective+':>10s} {'rating':>6s}" + "".join(f" {a[:14]:>14s}" for a in asked)
    print(header)
    for d in rows:
        pos = _pct(d.effectiveness.share("positive")) if d.effectiveness.total else "-"
        rating = f"{d.average_rating:.1f}" if d.average_rating is not None else "-"
        concern = "".join(f" {(_pct(v) if v is not None else '-'):>14s}" for v in d.concern_shares.values())
        print(f"{d.drug[:32]:32s} {d.matched:7d} {pos:>10s} {rating:>6s}{concern}")
    if asked:
        print("\nSide-effect columns: share of all the drug's reviews that report it.")


def main():
    ap = argparse.ArgumentParser(prog="python -m medreview.summary", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--drug", help="summarize this drug instead of the one named in the question")
    ap.add_argument("--condition", help="only reviews for this condition, e.g. Insomnia")
    ap.add_argument("--method", choices=sorted(METHODS), default="bm25")
    ap.add_argument("--limit", type=int, default=50, help="reviews to summarize")
    ap.add_argument("--sentiment", choices=["auto", "logreg", "vader"], default="auto")
    ap.add_argument("--show", type=int, default=5, help="reviews to print")
    ap.add_argument("--index", type=Path, default=Path("data/index/reviews.sqlite"))
    ap.add_argument("--table", type=Path, default=Path("data/aspects/side_effects_by_drug.csv"))
    ap.add_argument("--model", type=Path, default=SENTIMENT_MODEL)
    a = ap.parse_args()
    if a.limit < 1:
        raise SystemExit("--limit must be at least 1")
    if not a.index.exists():
        raise SystemExit(f"{a.index} not found; run `python -m medreview all` first.")
    sentiment, used = load_sentiment(a.sentiment, a.model)
    table = SideEffectTable.load(a.table) if a.table.exists() else None
    with SearchIndex(a.index) as index:
        parsed, result = summarize_question(index, a.question, table, sentiment, drug=a.drug,
                                            condition=a.condition, method=a.method, limit=a.limit)
    print(f"(effectiveness sentiment: {'logistic regression' if used == 'logreg' else 'VADER'})\n")
    if isinstance(result, list):
        print_comparison(parsed, result)
    else:
        print_summary(parsed, result, a.show)


if __name__ == "__main__":
    main()
