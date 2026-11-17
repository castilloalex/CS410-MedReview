# AI-assisted code
# Model: Opus 5.5 (max effort)
"""One call from a question to a summary; used by the command line.

    from medreview.index import SearchIndex
    from medreview.aspects import SideEffectTable
    from medreview.summary import load_sentiment, summarize_question

    with SearchIndex("data/index/reviews.sqlite") as index:
        sentiment, _ = load_sentiment()
        table = SideEffectTable.load("data/aspects/side_effects_by_drug.csv")
        parsed, result = summarize_question(index, "Does trazodone make you groggy?", table, sentiment)

`result` is a DrugSummary when the question names a drug (or `drug` is given), and
a list of DrugComparison rows when it does not.
"""
from pathlib import Path

from .analysis import analyze_reviews, build_profile, compare_drugs, summarize
from .query import DrugCatalog, parse_query
from .retrieval import DrugView, retrieve

SENTIMENT_MODEL = Path("data/sentiment/sentiment_logreg.joblib")


def load_sentiment(method: str = "auto", model_path=SENTIMENT_MODEL):
    """(classifier, method used). "auto" picks logistic regression when its model file
    exists (`python -m medreview.sentiment train`) and VADER otherwise."""
    from ..sentiment.api import default_classifier

    if method == "auto":
        method = "logreg" if Path(model_path).exists() else "vader"
    return default_classifier(method, model_path), method


def fetch_hits(index, parsed, drug: str, method: str = "bm25", limit: int = 50, depth: int | None = None):
    """Search results for a parsed question within one drug, or None with no search words.

    `depth` ranks more than `limit` reviews, for filtering by condition afterwards."""
    if not parsed.keywords:
        return None
    view = DrugView(index, drug) if method != "bm25" else None
    return retrieve(index, parsed.search_text, drug=drug, method=method, limit=max(1, depth or limit), view=view)


def summarize_question(index, question: str, table=None, sentiment=None, drug: str | None = None,
                       condition: str | None = None, method: str = "bm25", limit: int = 50,
                       catalog: DrugCatalog | None = None, compare_limit: int = 200):
    """(ParsedQuery, DrugSummary or list of DrugComparison)."""
    catalog = catalog or DrugCatalog.from_index(index)
    parsed = parse_query(question, catalog, drug=drug)
    if parsed.drug is None:
        hits = retrieve(index, parsed.search_text, None, "bm25", compare_limit) if parsed.keywords else []
        return parsed, compare_drugs(analyze_reviews(hits, sentiment), parsed, table, catalog)
    full = build_profile(index, parsed.drug, sentiment)
    hits = fetch_hits(index, parsed, full.drug, method, limit, depth=full.n if condition else None)
    return parsed, summarize(full.for_condition(condition), hits, parsed, table, method=method, limit=limit)
