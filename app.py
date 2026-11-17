# Model: Opus 5.5 (max effort) For integrating components to GUIs
"""MedReview: search medication reviews by the concern you have.

    streamlit run app.py

Needs data/index/reviews.sqlite (`python -m medreview all`). Uses the logistic
regression sentiment model when data/sentiment/sentiment_logreg.joblib exists
(`python -m medreview.sentiment train`) and VADER otherwise. Set MEDREVIEW_DATA to
use a different data folder.
"""
import html
import os
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from medreview.aspects import SideEffectTable
from medreview.index import SearchIndex
from medreview.summary import (METHODS, DrugCatalog, DrugView, analyze_reviews, build_profile, compare_drugs,
                               load_sentiment, parse_query, retrieve, summarize)
from medreview.summary import render

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("MEDREVIEW_DATA", ROOT / "data"))
INDEX = DATA / "index" / "reviews.sqlite"
TABLE = DATA / "aspects" / "side_effects_by_drug.csv"
MODEL = DATA / "sentiment" / "sentiment_logreg.joblib"
REPORTS = ROOT / "reports"

AUTO = "Find it in my question"
COMPARE_LIMIT = 200
SHOWN_REVIEWS = 10
EXAMPLES = [
    "Does trazodone help with sleep and does it make you feel groggy?",
    "Does sertraline help anxiety, and does it cause nausea?",
    "Will Nexplanon make me gain weight?",
    "Does metformin cause diarrhea?",
    "Does gabapentin help nerve pain without making you dizzy?",
    "Which sleep medicine works without grogginess?",
]
SENTIMENT_METHODS = {"logreg": "Logistic regression", "vader": "VADER"}


def mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


# ---------------------------------------------------------------- cached data

@st.cache_resource(show_spinner=False)
def load_catalog(index_path: str, stamp: float) -> DrugCatalog:
    with SearchIndex(index_path) as index:
        return DrugCatalog.from_index(index)


@st.cache_data(show_spinner=False)
def index_metadata(index_path: str, stamp: float) -> dict:
    with SearchIndex(index_path) as index:
        return dict(index.metadata)


@st.cache_resource(show_spinner=False)
def load_table(path: str, stamp: float):
    return SideEffectTable.load(path) if Path(path).exists() else None


@st.cache_resource(show_spinner=False)
def load_classifier(method: str, model_path: str, stamp: float):
    return load_sentiment(method, model_path)[0]


@st.cache_resource(show_spinner=False, max_entries=24)
def load_profile(index_path: str, stamp: float, drug: str, method: str, model_path: str, model_stamp: float):
    classifier = load_classifier(method, model_path, model_stamp)
    with SearchIndex(index_path) as index:
        return build_profile(index, drug, classifier)


@st.cache_data(show_spinner=False, max_entries=256)
def search_drug(index_path: str, stamp: float, query: str, drug: str, method: str, depth: int) -> list[dict]:
    with SearchIndex(index_path) as index:
        view = DrugView(index, drug) if method != "bm25" else None
        hits = retrieve(index, query, drug=drug, method=method, limit=depth, view=view)
    return [{"doc_id": h["doc_id"], "score": h["score"]} for h in hits]


@st.cache_resource(show_spinner=False, max_entries=32)
def search_all_drugs(index_path: str, stamp: float, query: str, sentiment_method: str, model_path: str,
                     model_stamp: float):
    classifier = load_classifier(sentiment_method, model_path, model_stamp)
    with SearchIndex(index_path) as index:
        hits = retrieve(index, query, None, "bm25", COMPARE_LIMIT)
    return analyze_reviews(hits, classifier)


@st.cache_data(show_spinner=False)
def evaluation_results(reports_dir: str) -> dict:
    """The team's evaluation numbers, read from reports/ (each part is optional)."""
    reports, out = Path(reports_dir), {}
    try:
        from medreview.evaluation import compute_ndcg, compute_precision, load_qrels_and_results
        qrels, runs = load_qrels_and_results(reports / "relevance_judgments_pool.xlsx")
        out["retrieval"] = pd.DataFrame(
            [{"Method": m, "Precision@10": float(compute_precision(r, qrels)), "nDCG@10": float(compute_ndcg(r, qrels))}
             for m, r in runs.items()])
    except Exception:
        pass
    try:
        df = pd.read_csv(reports / "sentiment_eval.csv")
        rows = []
        for method, g in df.groupby("method", sort=False):
            rows.append({"Method": {"vader": "VADER", "logreg": "Logistic regression"}.get(method, method),
                         "Accuracy": float(g.loc[g["class"] == "accuracy", "f1"].iloc[0]),
                         "Macro-F1": float(g.loc[g["class"] == "macro_avg", "f1"].iloc[0])})
        out["sentiment"] = pd.DataFrame(rows)
    except Exception:
        pass
    try:
        df = pd.read_csv(reports / "aspect_eval.csv")
        df = df[df["class"] == "macro_avg"]
        out["aspects"] = pd.DataFrame({"Method": df["method"], "Precision": df["precision"],
                                       "Recall": df["recall"], "Macro-F1": df["f1"]})
    except Exception:
        pass
    return out


# ---------------------------------------------------------------- callbacks

def use_example():
    choice = st.session_state.get("example")
    if choice:
        st.session_state["question"] = choice
        st.session_state["drug_choice"] = AUTO
    st.session_state["example"] = None


def open_drug():
    choice = st.session_state.get("open_drug")
    if choice:
        st.session_state["drug_choice"] = choice
    st.session_state["open_drug"] = None


# ---------------------------------------------------------------- formatting

def esc(text) -> str:
    return html.escape(str(text))


def nice_date(iso: str | None) -> str | None:
    try:
        d = date.fromisoformat(iso)
    except (TypeError, ValueError):
        return None
    return f"{d:%B} {d.day}, {d.year}"


def review_meta(rank: int, review, method_name: str | None) -> str:
    parts = [f"<b>{rank}.</b> Rated {review.rating}/10"]
    if review.condition:
        parts[0] += f" for {esc(review.condition)}"
    when = nice_date(review.date)
    if when:
        parts[0] += f", {when}"
    if review.useful_count:
        parts.append(f"{review.useful_count:,} found it useful")
    if review.effectiveness_sentiment:
        k = review.effectiveness_sentiment
        parts.append(f'<i class="mr-key mr-{k}"></i>Effectiveness reads {render.SENTIMENT_NAMES[k].lower()}')
    if review.score is not None and method_name:
        parts.append(f"{method_name} score {review.score:.2f}")
    return '<p class="mr-meta">' + ". ".join(parts) + ".</p>"


def show_reviews(reviews, pattern, method_name):
    for rank, review in enumerate(reviews, 1):
        with st.container(border=True):
            st.markdown(review_meta(rank, review, method_name) + render.review_html(review, pattern),
                        unsafe_allow_html=True)


# ---------------------------------------------------------------- sections

def effectiveness_section(s):
    st.subheader("Does it work?")
    if s.searched:
        e, scope = s.effectiveness, f"{len(s.reviews)} reviews found"
    else:
        e, scope = s.baseline, f"{s.profile.n:,} {esc(s.drug)} reviews{for_condition(s)}"
    if not e.total:
        st.markdown(f"None of the {scope} has a sentence about whether the drug worked.")
        return
    st.markdown(render.sentiment_bar(e, f"Effectiveness in the {scope}") + render.sentiment_legend(),
                unsafe_allow_html=True)
    st.caption(f"{e.positive:,} positive, {e.neutral:,} mixed or neutral and {e.negative:,} negative, from the "
               f"{e.total:,} of the {scope} that say whether it worked. Only sentences about effectiveness are "
               "scored, so side effects do not pull this down.")
    b = s.baseline
    if b.total and s.searched:
        st.markdown(f"**All {s.profile.n:,} {esc(s.drug)} reviews{for_condition(s)}**")
        st.markdown(render.sentiment_bar(b, f"Effectiveness in all {s.drug} reviews"), unsafe_allow_html=True)
        st.caption(f"{b.share('positive'):.0%} positive, from the {b.total:,} reviews that say whether it worked.")


def for_condition(s) -> str:
    return f" for {esc(s.profile.condition)}" if s.profile.condition else ""


def concern_section(s, pattern):
    for c in s.concerns:
        st.subheader(f"Does it cause {c.name}?")
        drug = esc(s.drug)
        figure = f"{c.share:.0%}" if c.share >= 0.095 else f"{c.share:.1%}"
        st.markdown(f'<div class="mr-figure">{figure}</div><div class="mr-figure-label">of the {c.reviews:,} '
                    f'{drug} reviews{for_condition(s)} report {esc(c.name)}</div>', unsafe_allow_html=True)
        notes = []
        if c.lift:
            notes.append(f"That is {c.lift:.1f} times the rate across all drugs.")
        notes.append(f"{c.denied:,} reviews say they did not have it.")
        if s.searched:
            notes.append(f"In the {len(s.reviews)} reviews found: {c.matched_reported} report it, "
                         f"{c.matched_denied} say they did not.")
        st.markdown(" ".join(notes))
        if c.treated:
            st.caption(f"Many {drug} reviews are for this problem, so mentions of it as the problem being "
                       "treated are not counted as a side effect.")
        quotes = "".join(render.quote_html(r, sentence, pattern) for r, sentence in c.reported_examples[:2])
        quotes += "".join(render.quote_html(r, sentence, pattern) for r, sentence in c.denied_examples[:1])
        if quotes:
            st.markdown(quotes, unsafe_allow_html=True)


def common_section(s):
    st.subheader(f"Most reported for {s.drug}")
    if not s.common_side_effects:
        st.markdown("The side-effect table has nothing for this drug.")
        return
    df = pd.DataFrame([{"Side effect": name, "Share of reviews": share * 100} for name, share in s.common_side_effects])
    st.dataframe(df, hide_index=True, column_config={
        "Share of reviews": st.column_config.ProgressColumn("Share of all its reviews", format="%.1f%%",
                                                            min_value=0, max_value=100)})
    st.caption("Across all of the drug's reviews. Problems the drug is often taken for are left out.")


def side_effects_section(s, scope_text):
    rows = s.side_effects[:15]
    st.subheader(f"Side effects in the {scope_text}")
    if not any(r.reported for r in rows):
        st.markdown(f"None of the {scope_text} report a listed side effect.")
    if rows:
        df = pd.DataFrame([{
            "Side effect": r.name + ("  (you asked)" if r.asked else ""),
            "Reviews": r.reported,
            "Here": r.share * 100,
            "All reviews": r.drug_share * 100,
            "vs all drugs": r.lift,
        } for r in rows])
        st.dataframe(df, hide_index=True, height=35 * (len(df) + 1) + 3, column_config={
            "Reviews": st.column_config.NumberColumn(f"In these {len(s.reviews)}",
                                                     help="How many of these reviews report it"),
            "Here": st.column_config.ProgressColumn(f"Share of these {len(s.reviews)}", format="%.0f%%",
                                                    min_value=0, max_value=100),
            "All reviews": st.column_config.ProgressColumn(f"Share of all {s.profile.n:,}", format="%.1f%%",
                                                           min_value=0, max_value=100),
            "vs all drugs": st.column_config.NumberColumn(
                "vs all drugs", format="%.1fx", help="The drug's share divided by the share across all drugs"),
        })
    notes = []
    if s.no_side_effects:
        notes.append(f"{s.no_side_effects} of these reviews say they had no side effects.")
    if s.common_side_effects:
        notes.append(f"The list also includes {s.drug}'s most reported side effects across all its reviews.")
    if s.searched and any(r.asked for r in rows):
        notes.append("Reviews found by searching for a side effect mention it more often than reviews in general, "
                     "so compare with the share of all reviews.")
    if notes:
        st.caption(" ".join(notes))


def drug_view(parsed, method, limit, sentiment_method, condition_slot):
    stamp = mtime(INDEX)
    info = load_catalog(str(INDEX), stamp).get(parsed.drug)
    total = info.reviews if info else 0
    with st.spinner(f"Reading all {total:,} {parsed.drug} reviews..."):
        full = load_profile(str(INDEX), stamp, parsed.drug, sentiment_method, str(MODEL), mtime(MODEL))
    conditions = full.conditions()
    counts = dict(conditions)
    condition = condition_slot.selectbox(
        "Condition", [None] + [c for c, _ in conditions], key=f"condition::{parsed.drug}",
        format_func=lambda c: "All conditions" if c is None else f"{c} ({counts[c]:,})",
        help="Only use reviews written by people taking the drug for this condition.")
    profile = full.for_condition(condition)
    hits = None
    if parsed.keywords:
        depth = full.n if condition else limit
        hits = search_drug(str(INDEX), stamp, parsed.search_text, parsed.drug, method, max(1, depth))
    s = summarize(profile, hits, parsed, load_table(str(TABLE), mtime(TABLE)), method=method, limit=limit)
    pattern = render.term_pattern(parsed.keywords, parsed.side_effects)
    if s.searched and not s.reviews:
        st.warning(f"No {parsed.drug} reviews{for_condition(s)} contain {', '.join(parsed.keywords)}. "
                   "Try other words, or clear the question to see the most useful reviews.")
        return
    scope_text = "reviews found" if s.searched else "most useful reviews"
    left, right = st.columns([3, 2], gap="large")
    with left:
        effectiveness_section(s)
    with right:
        if s.concerns:
            concern_section(s, pattern)
        else:
            common_section(s)
    st.divider()
    side_effects_section(s, scope_text)
    st.divider()
    method_name = METHODS[method] if s.searched else None
    if s.searched:
        st.subheader(f"The {len(s.reviews)} reviews found, best match first")
    else:
        st.subheader(f"The {len(s.reviews)} most useful reviews")
        st.caption("No search words in the question, so reviews are ordered by how many readers found them useful.")
    st.markdown(render.aspect_legend(), unsafe_allow_html=True)
    show_reviews(s.reviews[:SHOWN_REVIEWS], pattern, method_name)
    if len(s.reviews) > SHOWN_REVIEWS:
        with st.expander(f"Show the other {len(s.reviews) - SHOWN_REVIEWS} reviews"):
            for rank, review in enumerate(s.reviews[SHOWN_REVIEWS:], SHOWN_REVIEWS + 1):
                st.markdown(review_meta(rank, review, method_name) + render.review_html(review, pattern),
                            unsafe_allow_html=True)


def compare_view(parsed, sentiment_method, catalog):
    if not parsed.keywords:
        st.info("Ask about a medication and a concern, or pick a drug in the sidebar.")
        return
    stamp = mtime(INDEX)
    with st.spinner("Searching all drugs..."):
        analyzed = search_all_drugs(str(INDEX), stamp, parsed.search_text, sentiment_method, str(MODEL), mtime(MODEL))
    rows = compare_drugs(analyzed, parsed, load_table(str(TABLE), mtime(TABLE)), catalog)
    if not rows:
        st.warning(f"No reviews contain {', '.join(parsed.keywords)}.")
        return
    st.subheader("Drugs whose reviews best match your question")
    st.caption(f"Your question names no drug, so these are the drugs behind the {len(analyzed)} best-matching "
               "reviews across all drugs (BM25). Pick one to see its full summary.")
    data = []
    for d in rows:
        row = {"Drug": d.drug, "Reviews found": d.matched,
               "Works (positive)": d.effectiveness.share("positive") * 100 if d.effectiveness.total else None,
               "Average rating": d.average_rating}
        for name, share in d.concern_shares.items():
            row[name] = share * 100 if share is not None else None
        data.append(row)
    config = {
        "Works (positive)": st.column_config.ProgressColumn(
            "Effectiveness: positive", format="%.0f%%", min_value=0, max_value=100,
            help="Share of the reviews found here whose effectiveness sentences read positive"),
        "Average rating": st.column_config.NumberColumn("Average rating", format="%.1f",
                                                        help="Of the reviews found here, out of 10"),
    }
    for name in parsed.side_effects:
        config[name] = st.column_config.ProgressColumn(
            f"Report {name}", format="%.1f%%", min_value=0, max_value=100,
            help="Share of all the drug's reviews that report it")
    st.dataframe(pd.DataFrame(data), hide_index=True, column_config=config)
    st.pills("Open a drug's summary", [d.drug for d in rows[:8]], key="open_drug", on_change=open_drug)


def how_it_works():
    with st.expander("How this works"):
        st.markdown(
            "1. **Search.** The question becomes search words: question words and the drug's name are dropped. "
            "BM25, TF-IDF or RM3 ranks the drug's reviews.\n"
            "2. **Sentences.** A keyword classifier labels each sentence as about effectiveness, about side "
            "effects, or neither.\n"
            "3. **Does it work?** A sentiment classifier trained on the reviews' 1-10 ratings reads only the "
            "effectiveness sentences of each review.\n"
            "4. **Side effects.** Sentences about side effects are matched against a list of about 40 side "
            "effects. \"No nausea\" counts as saying they did not have it. A drug's own treated problem, such as "
            "anxiety in an anxiety review, is not counted.\n"
            "5. **Context.** Shares for the whole drug come from all of its reviews, and \"vs all drugs\" "
            "divides that share by the rate across every drug.")
        names = {"retrieval": "Ranking, 25 judged queries", "sentiment": "Review sentiment, 53,753 test reviews",
                 "aspects": "Sentence aspects, 120 hand-labeled sentences"}
        cols = st.columns(len(results)) if results else []
        for col, (key, df) in zip(cols, results.items()):
            with col:
                st.markdown(f"**{names[key]}**")
                st.dataframe(df, hide_index=True, column_config={
                    c: st.column_config.NumberColumn(format="%.3f") for c in df.columns if c != "Method"})
        st.caption("Limits: the effectiveness reading is approximate. The sentiment model was trained on whole "
                   "reviews, and some sentences labeled as effectiveness are about another treatment. Brand and "
                   "generic names are separate drugs (Zoloft and Sertraline). Counts are mentions in reviews, not "
                   "clinical rates. Negation is a short-window rule, so \"I don't like the groggy feeling\" reads "
                   "as not having it.")


# ---------------------------------------------------------------- page

st.set_page_config(page_title="MedReview: search reviews by concern", page_icon="💊", layout="wide")
st.markdown(render.CSS, unsafe_allow_html=True)

if not INDEX.exists():
    st.title("MedReview")
    st.error(f"The search index is missing ({INDEX}). From the project folder, run:")
    st.code("python -m medreview all\npython -m medreview.sentiment train   # optional, better sentiment\n"
            "streamlit run app.py", language="bash")
    st.stop()

catalog = load_catalog(str(INDEX), mtime(INDEX))
meta = index_metadata(str(INDEX), mtime(INDEX))
st.session_state.setdefault("question", EXAMPLES[0])
st.session_state.setdefault("drug_choice", AUTO)

results = evaluation_results(str(REPORTS))
ranking_help = "How the drug's reviews are ranked."
if "retrieval" in results:
    df = results["retrieval"]
    ranking_help += " nDCG@10 on the team's 25 judged queries: " + ", ".join(
        f"{m} {v:.3f}" for m, v in zip(df["Method"], df["nDCG@10"])) + "."
sentiment_help = "Scores the effectiveness sentences of each review."
if "sentiment" in results:
    df = results["sentiment"]
    sentiment_help += " Macro-F1 on the test split: " + ", ".join(
        f"{m} {v:.3f}" for m, v in zip(df["Method"], df["Macro-F1"])) + "."

with st.sidebar:
    counts = {name: catalog.get(name).reviews for name in catalog.names()}
    drug_choice = st.selectbox(
        "Drug", [AUTO] + list(counts), key="drug_choice",
        format_func=lambda d: d if d == AUTO else f"{d} ({counts[d]:,})",
        help="Brand and generic names are listed separately, as on Drugs.com.")
    condition_slot = st.empty()
    method = st.radio("Ranking", list(METHODS), format_func=METHODS.get, horizontal=True, help=ranking_help,
                      key="method")
    limit = st.slider("Reviews to summarize", 10, 200, 50, step=10, key="limit",
                      help="How many of the best-matching reviews the summary is based on.")
    if MODEL.exists():
        sentiment_method = st.radio("Sentiment model", list(SENTIMENT_METHODS), format_func=SENTIMENT_METHODS.get,
                                    help=sentiment_help, key="sentiment_method")
    else:
        sentiment_method = "vader"
        st.caption("Sentiment: VADER. Run `python -m medreview.sentiment train` for the more accurate model.")
    st.caption(f"{meta['documents']:,} Drugs.com reviews of {len(catalog):,} drug names "
               "(UCI Drug Review Dataset).")

st.title("MedReview")
st.markdown("Ask about a medication and the concern you have. The reviews that match are summarized "
            "two ways: whether the drug worked, and which side effects people report.")
question = st.text_input("Your question", key="question", placeholder="Does sertraline cause nausea?")
st.pills("Examples", EXAMPLES, key="example", on_change=use_example, label_visibility="collapsed")

override = None if drug_choice == AUTO else drug_choice
parsed = parse_query(question, catalog, drug=override)

notes = []
if parsed.drug:
    where = "named in your question" if parsed.drug_found and not override else "chosen in the sidebar"
    notes.append(f"**{parsed.drug}**, {where}")
if parsed.keywords:
    notes.append("searching for " + ", ".join(f"*{esc(k)}*" for k in parsed.keywords))
if parsed.side_effects:
    notes.append("side effects you asked about: " + ", ".join(parsed.side_effects))
if parsed.treated:
    notes.append("asks whether it helps with " + ", ".join(parsed.treated))
if notes:
    line = "; ".join(notes)
    st.markdown(line[0].upper() + line[1:] + ".")
if override and parsed.other_drugs:
    st.caption(f"Your question also names {', '.join(parsed.other_drugs)}. Choose \"{AUTO}\" in the sidebar "
               "to use the drug in your question.")
elif len(parsed.other_drugs) > 0:
    st.caption(f"Your question also names {', '.join(parsed.other_drugs)}; pick it in the sidebar to see it instead.")

if parsed.drug:
    drug_view(parsed, method, limit, sentiment_method, condition_slot)
else:
    compare_view(parsed, sentiment_method, catalog)

st.divider()
how_it_works()
st.caption("Reviews describe personal experiences and are not medical advice. Data: Kallumadi and Gräßer (2018), "
           "Drug Review Dataset (Drugs.com), UCI Machine Learning Repository.")
