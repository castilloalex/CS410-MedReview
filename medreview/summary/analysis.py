# AI-assisted code
# Model: Opus 5.5 (max effort)
"""Per-review analysis and per-drug summaries.

For each review:
  - `classify_review` labels every sentence effectiveness / side_effects / neither
    (the keyword classifier by default);
  - the sentiment classifier labels only the review's effectiveness sentences, so the
    result says whether the drug worked, separately from its side effects;
  - side-effect sentences are searched for the lexicon's side effects. A mention is
    "reported" unless it is negated ("no nausea") or names the problem being treated;
    a negated mention is "denied". Reported side effects follow the same rule as
    `extract_side_effects`, so drug-wide counts match the side-effect table.

A summary sets the reviews a query retrieved against all of the drug's reviews.
"""
import re
from collections import Counter
from dataclasses import dataclass, field, replace

from ..aspects import EFFECTIVENESS, SIDE_EFFECTS, classify_review
from ..aspects.mentions import symptom_mentions
from ..text import clean_text
from .query import ParsedQuery

SENTIMENTS = ("positive", "neutral", "negative")

# "No side effects", "didn't have any side effects", "zero side effects". Counted only
# for reviews that report no specific side effect.
NO_SIDE_EFFECTS = re.compile(
    r"(?:\bno|\bzero|\bwithout(?: any)?|\bnot (?:any|many)|(?:n't|\bnot|\bnever) (?:have|had|get|got|experience|"
    r"experienced|notice|noticed)(?: any)?)(?: (?:real|major|bad|noticeable|significant|serious|negative))?"
    r"\s+side[- ]?effects?\b", re.I)


@dataclass
class ReviewAnalysis:
    doc_id: int
    drug_name: str
    condition: str | None
    rating: int
    date: str | None
    useful_count: int
    review: str
    sentences: list                       # [(sentence, aspect), ...] from classify_review
    side_effects: list                    # side effects the reviewer reports, in order
    denied: list                          # side effects the reviewer says they did not have
    reported_in: dict                     # side effect -> index of the first sentence reporting it
    denied_in: dict                       # side effect -> index of the first sentence denying it
    no_side_effects: bool                 # says "no side effects" and reports none
    effectiveness_sentiment: str | None = None   # None when no sentence is about effectiveness
    score: float | None = None            # retrieval score, for reviews a search returned

    @property
    def effectiveness_sentences(self) -> list[str]:
        return [s for s, label in self.sentences if label == EFFECTIVENESS]

    @property
    def effectiveness_text(self) -> str:
        return " ".join(self.effectiveness_sentences)


def analyze_reviews(docs, sentiment=None, aspect_classifier=None) -> list[ReviewAnalysis]:
    """Analyze review dicts (index documents or search hits).

    `sentiment` is any classifier with `predict(texts)`, such as
    `medreview.sentiment.default_classifier("logreg")`. Without one,
    `effectiveness_sentiment` stays None.
    """
    out = []
    for doc in docs:
        condition = doc.get("condition") if isinstance(doc.get("condition"), str) else None
        sentences = classify_review(doc["review"], aspect_classifier, condition=condition)
        reported, denied, reported_in, denied_in = [], [], {}, {}
        for i, (sentence, label) in enumerate(sentences):
            if label != SIDE_EFFECTS:
                continue
            for mention in symptom_mentions(sentence, condition):
                if mention.treated:
                    continue
                names, where = (denied, denied_in) if mention.negated else (reported, reported_in)
                if mention.name not in names:
                    names.append(mention.name)
                    where[mention.name] = i
        denied = [name for name in denied if name not in reported]
        denied_in = {name: denied_in[name] for name in denied}
        out.append(ReviewAnalysis(
            doc_id=doc["doc_id"], drug_name=doc["drug_name"], condition=condition, rating=int(doc["rating"]),
            date=doc.get("date"), useful_count=int(doc.get("useful_count") or 0), review=doc["review"],
            sentences=sentences, side_effects=reported, denied=denied, reported_in=reported_in,
            denied_in=denied_in, no_side_effects=not reported and bool(NO_SIDE_EFFECTS.search(doc["review"])),
            score=doc.get("score")))
    if sentiment is not None:
        todo = [r for r in out if r.effectiveness_sentences]
        if todo:
            for review, label in zip(todo, sentiment.predict([r.effectiveness_text for r in todo])):
                review.effectiveness_sentiment = str(label)
    return out


@dataclass
class SentimentCounts:
    positive: int = 0
    neutral: int = 0
    negative: int = 0

    @classmethod
    def of(cls, reviews) -> "SentimentCounts":
        c = Counter(r.effectiveness_sentiment for r in reviews if r.effectiveness_sentiment)
        return cls(c["positive"], c["neutral"], c["negative"])

    @property
    def total(self) -> int:
        return self.positive + self.neutral + self.negative

    def share(self, label: str) -> float:
        return getattr(self, label) / self.total if self.total else 0.0


def _average(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


@dataclass
class DrugProfile:
    """All reviews of one drug, analyzed (optionally only those for one condition)."""
    drug: str
    reviews: list
    condition: str | None = None

    def __post_init__(self):
        self.by_id = {r.doc_id: r for r in self.reviews}
        self.reported = Counter(name for r in self.reviews for name in r.side_effects)
        self.denied = Counter(name for r in self.reviews for name in r.denied)

    @property
    def n(self) -> int:
        return len(self.reviews)

    @property
    def average_rating(self) -> float | None:
        return _average(r.rating for r in self.reviews)

    @property
    def effectiveness(self) -> SentimentCounts:
        return SentimentCounts.of(self.reviews)

    def conditions(self) -> list[tuple[str, int]]:
        """[(condition, reviews), ...], most common first."""
        return Counter(r.condition for r in self.reviews if r.condition).most_common()

    def for_condition(self, condition: str | None) -> "DrugProfile":
        if not condition:
            return self
        key = condition.casefold()
        return DrugProfile(self.drug, [r for r in self.reviews if (r.condition or "").casefold() == key], condition)

    def most_useful(self, limit: int) -> list[ReviewAnalysis]:
        return sorted(self.reviews, key=lambda r: (-r.useful_count, r.doc_id))[:limit]


def build_profile(index, drug: str, sentiment=None, aspect_classifier=None) -> DrugProfile:
    """Analyze every review of `drug` in a SearchIndex."""
    key = clean_text(drug).casefold()
    docs = [dict(row) for row in index.connection.execute(
        "SELECT * FROM documents WHERE drug_key=? ORDER BY doc_id", (key,))]
    name = docs[0]["drug_name"] if docs else clean_text(drug)
    return DrugProfile(name, analyze_reviews(docs, sentiment, aspect_classifier))


@dataclass
class SideEffectRow:
    name: str
    reported: int               # matched reviews that report it
    share: float                # reported / matched reviews
    drug_reported: int          # all of the drug's reviews (in the profile) that report it
    drug_share: float           # drug_reported / reviews in the profile
    lift: float | None          # side-effect table: the drug's rate / the rate across all drugs
    asked: bool                 # the question asks about it


@dataclass
class Concern:
    """A side effect the question asks about, over all of the drug's reviews."""
    name: str
    reported: int               # reviews of the drug that report it
    denied: int                 # reviews that say they did not have it
    reviews: int                # reviews of the drug (in the profile)
    lift: float | None          # from the side-effect table, when it has the pair
    treated: bool               # the table marks it as a problem this drug is often taken for
    matched_reported: int       # same counts within the matched reviews
    matched_denied: int
    reported_examples: list = field(default_factory=list)   # [(ReviewAnalysis, sentence), ...]
    denied_examples: list = field(default_factory=list)

    @property
    def share(self) -> float:
        return self.reported / self.reviews if self.reviews else 0.0


@dataclass
class DrugSummary:
    drug: str
    query: ParsedQuery
    method: str
    searched: bool              # False when there were no search words (most useful reviews shown)
    reviews: list               # matched reviews, best first
    profile: DrugProfile
    effectiveness: SentimentCounts
    side_effects: list          # SideEffectRow for side effects in the matched reviews, asked about,
                                # or among the drug's most common; most reported first
    concerns: list              # Concern for each side effect asked about
    common_side_effects: list   # SideEffectTable.top_list(drug, k): [(name, share of reviews), ...]
    no_side_effects: int        # matched reviews saying they had no side effects

    @property
    def baseline(self) -> SentimentCounts:
        return self.profile.effectiveness

    @property
    def average_rating(self) -> float | None:
        return _average(r.rating for r in self.reviews)


def _table_rows(table, drug):
    """{side effect: row} for a drug in the side-effect table, all rows included."""
    if table is None:
        return {}
    top = table.top(drug, k=10_000, min_reviews=1, min_pct=0.0, include_treated=True)
    return {row.side_effect: row for row in top.itertuples(index=False)}


def _examples(reviews, name, where_attr, preferred_ids, k):
    pool = [r for r in reviews if name in getattr(r, where_attr)]
    pool.sort(key=lambda r: (r.doc_id not in preferred_ids, -r.useful_count, r.doc_id))
    return [(r, r.sentences[getattr(r, where_attr)[name]][0]) for r in pool[:k]]


def summarize(profile: DrugProfile, hits, query: ParsedQuery, table=None, method: str = "bm25",
              limit: int | None = None, top_k: int = 8, examples: int = 3) -> DrugSummary:
    """Summarize the reviews a search returned for one drug.

    `hits` are search results (dicts with doc_id and score) or None. With no search
    words, the drug's most useful reviews stand in for search results. Hits outside the
    profile (another condition) are skipped. `table` is a SideEffectTable or None.
    """
    searched = bool(query.keywords) and hits is not None
    if searched:
        matched = []
        for hit in hits:
            review = profile.by_id.get(hit["doc_id"])
            if review is not None:
                matched.append(replace(review, score=hit.get("score")))
        if limit is not None:
            matched = matched[:limit]
    else:
        matched = profile.most_useful(limit or 50)
    rows = _table_rows(table, profile.drug)
    common = table.top_list(profile.drug, k=top_k) if table is not None else []
    counts = Counter(name for r in matched for name in r.side_effects)
    names = set(counts) | set(query.side_effects) | {name for name, _ in common}
    names = sorted(names, key=lambda n: (-counts[n], -profile.reported[n], n))
    side_effects = [SideEffectRow(
        name=name, reported=counts[name], share=counts[name] / len(matched) if matched else 0.0,
        drug_reported=profile.reported[name], drug_share=profile.reported[name] / profile.n if profile.n else 0.0,
        lift=float(rows[name].lift) if name in rows else None, asked=name in query.side_effects) for name in names]
    matched_ids = {r.doc_id for r in matched}
    concerns = []
    for name in query.side_effects:
        row = rows.get(name)
        concerns.append(Concern(
            name=name, reported=profile.reported[name], denied=profile.denied[name], reviews=profile.n,
            lift=float(row.lift) if row is not None else None, treated=bool(row.treated) if row is not None else False,
            matched_reported=sum(name in r.side_effects for r in matched),
            matched_denied=sum(name in r.denied for r in matched),
            reported_examples=_examples(profile.reviews, name, "reported_in", matched_ids, examples),
            denied_examples=_examples(profile.reviews, name, "denied_in", matched_ids, examples)))
    return DrugSummary(
        drug=profile.drug, query=query, method=method, searched=searched, reviews=matched, profile=profile,
        effectiveness=SentimentCounts.of(matched), side_effects=side_effects, concerns=concerns,
        common_side_effects=common, no_side_effects=sum(r.no_side_effects for r in matched))


@dataclass
class DrugComparison:
    """One drug's share of a search across all drugs."""
    drug: str
    matched: int                    # reviews of this drug among the results
    effectiveness: SentimentCounts  # over those reviews
    average_rating: float | None
    side_effects: list              # most reported side effects among those reviews
    concern_shares: dict            # asked side effect -> share of all the drug's reviews (table), or None
    total_reviews: int | None = None


def compare_drugs(reviews, query: ParsedQuery, table=None, catalog=None, top: int = 10) -> list[DrugComparison]:
    """Group analyzed search results from all drugs by drug, most results first."""
    groups = {}
    for review in reviews:
        groups.setdefault(review.drug_name, []).append(review)
    out = []
    for drug, items in groups.items():
        rows = _table_rows(table, drug)
        seen = Counter(name for r in items for name in r.side_effects)
        info = catalog.get(drug) if catalog is not None else None
        out.append(DrugComparison(
            drug=drug, matched=len(items), effectiveness=SentimentCounts.of(items),
            average_rating=_average(r.rating for r in items),
            side_effects=[name for name, _ in seen.most_common(3)],
            concern_shares={name: (float(rows[name].pct) if name in rows else (0.0 if rows else None))
                            for name in query.side_effects},
            total_reviews=info.reviews if info else None))
    out.sort(key=lambda d: (-d.matched, d.drug.casefold()))
    return out[:top]
