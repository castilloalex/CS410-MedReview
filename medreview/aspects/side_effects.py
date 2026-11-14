# AI-assisted code
# Model: Fable 5.2 (medium effort)
"""Extract each drug's common side effects from review text."""
from collections import Counter, defaultdict
from dataclasses import dataclass

import pandas as pd

from .keywords import KeywordAspectClassifier, SIDE_EFFECTS
from .mentions import symptom_mentions, treats
from .sentences import split_sentences


def find_side_effect_mentions(sentence: str, condition: str | None = None) -> list[str]:
    """Canonical side effects mentioned in one sentence, skipping negated mentions
    ("no nausea") and terms that name the problem being treated."""
    return [m.name for m in symptom_mentions(sentence, condition) if not m.negated and not m.treated]


def extract_side_effects(review: str, condition: str | None = None, classifier=None) -> list[str]:
    """Distinct side effects mentioned in one review.

    Only sentences the aspect classifier labels as side effects are searched.
    """
    classifier = classifier or _default_clf()
    out = []
    for sent in split_sentences(review):
        if classifier.predict_one(sent, condition) != SIDE_EFFECTS:
            continue
        for name in find_side_effect_mentions(sent, condition):
            if name not in out:
                out.append(name)
    return out


_CLF = None


def _default_clf():
    global _CLF
    if _CLF is None:
        _CLF = KeywordAspectClassifier()
    return _CLF


@dataclass
class SideEffectTable:
    """Per-drug side-effect frequencies.

    df columns:
      drug, side_effect
      n_reviews     reviews of the drug that mention the side effect
      drug_reviews  total reviews of the drug
      pct           n_reviews / drug_reviews
      lift          pct divided by the rate across all drugs (>1 = unusually common here)
      treated       True if the term matches a condition the drug is commonly reviewed
                    for (e.g. "anxiety" for an anxiety drug); hidden by default in top()
    """
    df: pd.DataFrame

    def top(self, drug: str, k: int = 5, min_reviews: int = 3, min_pct: float = 0.02,
            include_treated: bool = False) -> pd.DataFrame:
        d = self.df[self.df["drug"].str.lower() == drug.lower()]
        d = d[(d["n_reviews"] >= min_reviews) & (d["pct"] >= min_pct)]
        if not include_treated:
            d = d[~d["treated"]]
        return d.sort_values(["pct", "lift"], ascending=False).head(k).reset_index(drop=True)

    def top_list(self, drug: str, k: int = 5, **kw) -> list[tuple[str, float]]:
        """[(side_effect, share of the drug's reviews), ...], most common first."""
        t = self.top(drug, k, **kw)
        return [(s, round(float(p), 3)) for s, p in zip(t["side_effect"], t["pct"])]

    def drugs(self) -> list[str]:
        return sorted(self.df["drug"].unique())

    def save(self, path):
        self.df.to_csv(path, index=False)

    @classmethod
    def load(cls, path):
        return cls(pd.read_csv(path))


def build_side_effect_table(reviews: pd.DataFrame, drug_col="drug_name", review_col="review",
                            condition_col="condition", classifier=None,
                            condition_share: float = 0.10) -> SideEffectTable:
    """Aggregate side-effect mentions over a reviews DataFrame.

    A side effect is flagged `treated` for a drug when it matches a condition that
    accounts for at least `condition_share` of that drug's reviews.
    """
    classifier = classifier or _default_clf()
    counts: dict[str, Counter] = defaultdict(Counter)
    conds: dict[str, Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    overall: Counter = Counter()
    has_cond = condition_col in reviews.columns
    cols = [drug_col, review_col] + ([condition_col] if has_cond else [])
    for row in reviews[cols].itertuples(index=False):
        drug, text = row[0], row[1]
        cond = row[2] if has_cond and isinstance(row[2], str) else None
        if not isinstance(drug, str):
            continue
        totals[drug] += 1
        if cond:
            conds[drug][cond] += 1
        for se in extract_side_effects(text, cond, classifier):
            counts[drug][se] += 1
            overall[se] += 1
    n_all = max(sum(totals.values()), 1)
    rows = []
    for drug, c in counts.items():
        main = [k for k, n in conds[drug].items() if n / totals[drug] >= condition_share]
        for se, n in c.items():
            pct = n / totals[drug]
            rows.append((drug, se, n, totals[drug], round(pct, 4), round(pct / (overall[se] / n_all), 2),
                         any(treats(se, k) for k in main)))
    df = pd.DataFrame(rows, columns=["drug", "side_effect", "n_reviews", "drug_reviews", "pct", "lift", "treated"])
    df = df.sort_values(["drug", "pct"], ascending=[True, False]).reset_index(drop=True)
    return SideEffectTable(df)
