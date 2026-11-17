# AI-assisted code
# Model: Opus 5.5 (max effort)
"""Read a free-text question: which drug, which search words, which concerns.

    catalog = DrugCatalog.from_index(index)
    q = parse_query("Does trazodone help with sleep and does it make you feel groggy?", catalog)
    q.drug          # "Trazodone"
    q.keywords      # ["help", "sleep", "groggy"]
    q.side_effects  # ["drowsiness"]  side effects the question asks about
"""
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import NamedTuple

from ..aspects.mentions import symptom_mentions
from ..text import clean_text, tokenize

# Question words, fillers and words for "a drug". They say nothing about the concern, so
# they are left out of the search words (TF-IDF has no length normalization and RM3
# expands on them). "pill" stays: it means birth control in many reviews.
STOPWORDS = frozenset("""
a about after again all also am an and any anybody anyone are as at be been before being both but by can
could did do does doing during each else ever for from get gets getting go goes going got had has have
having he her here hers him his how i i'd i'm i've if in into is it it's its just like make makes making
me more most much my myself no nor not now of off on once only or other our out over own people really
same she should so some someone something such take takes taking than that the their them then there
these they this those through to too under until up us use used uses using very was we were what when
where whether which while who whom why will with without would you you're your yours feel feels feeling
cause causes medicine medicines medication medications meds drug drugs
""".split())


@dataclass(frozen=True)
class Drug:
    name: str             # as written in the data, e.g. "Trazodone"
    reviews: int          # reviews of this drug in the index
    tokens: tuple         # tokenize(name)
    ambiguous: bool       # few reviews and only everyday words ("RID", "My Way")


class DrugMatch(NamedTuple):
    drug: Drug
    start: int            # token positions in tokenize(text)
    end: int


def _exact(name: str, text: str) -> bool:
    return re.search(r"(?<![^\W_])" + re.escape(name) + r"(?![^\W_])", text) is not None


class DrugCatalog:
    """The drug names in the index, for finding them in a question.

    `counts` maps drug name to its number of reviews. A name is `ambiguous` when it
    has fewer than `min_reviews` reviews and every word in it is in `common_terms`
    (frequent words in the reviews): "RID" in "get rid of acne", "My Way". Ambiguous
    names are only recognized when written with their exact capitalization.
    """

    def __init__(self, counts: dict, common_terms=(), min_reviews: int = 50):
        common = set(common_terms)
        self._by_key = {}
        self._by_first = defaultdict(list)
        for name, n in counts.items():
            tokens = tuple(tokenize(name))
            if not tokens:
                continue
            drug = Drug(name, int(n), tokens, int(n) < min_reviews and all(t in common for t in tokens))
            self._by_key[clean_text(name).casefold()] = drug
            self._by_first[tokens[0]].append(drug)

    @classmethod
    def from_index(cls, index, common_share: float = 0.004, min_reviews: int = 50):
        """Build from a SearchIndex. Words in at least `common_share` of reviews count as common."""
        connection = index.connection
        counts = {row[0]: row[1] for row in connection.execute(
            "SELECT MIN(drug_name), COUNT(*) FROM documents GROUP BY drug_key")}
        threshold = max(1, int(index.metadata["documents"] * common_share))
        common = {row[0] for row in connection.execute("SELECT term FROM terms WHERE df >= ?", (threshold,))}
        return cls(counts, common, min_reviews)

    def __len__(self):
        return len(self._by_key)

    def __contains__(self, name):
        return self.get(name) is not None

    def get(self, name: str) -> Drug | None:
        """Look up a drug by name, ignoring case."""
        return self._by_key.get(clean_text(name or "").casefold())

    def names(self) -> list[str]:
        return sorted((d.name for d in self._by_key.values()), key=str.casefold)

    def find(self, text: str) -> list[DrugMatch]:
        """Drugs named in `text`, in the order they appear.

        Matches whole words, ignoring case and punctuation ("plan-b" finds "Plan B").
        A match inside a longer one is dropped ("Plan B" inside "Plan B One-Step").
        """
        text = clean_text(text or "")
        tokens = tokenize(text)
        found = []
        for i, token in enumerate(tokens):
            for drug in self._by_first.get(token, ()):
                end = i + len(drug.tokens)
                if tuple(tokens[i:end]) == drug.tokens and (not drug.ambiguous or _exact(drug.name, text)):
                    found.append(DrugMatch(drug, i, end))
        kept = [m for m in found if not any(
            o.start <= m.start and m.end <= o.end and (o.end - o.start) > (m.end - m.start) for o in found)]
        kept.sort(key=lambda m: (m.start, -(m.end - m.start), -m.drug.reviews))
        seen, out = set(), []
        for match in kept:
            if match.drug.name not in seen:
                seen.add(match.drug.name)
                out.append(match)
        return out


@dataclass
class ParsedQuery:
    text: str
    keywords: list[str]                                      # search words, in question order
    drug: str | None = None                                  # the drug the question is about
    drug_found: bool = False                                 # True if `drug` was found in the text
    other_drugs: list[str] = field(default_factory=list)     # other drugs named in the text
    side_effects: list[str] = field(default_factory=list)    # side effects asked about, e.g. "drowsiness"
    treated: list[str] = field(default_factory=list)         # lexicon terms used as the problem treated
                                                              # ("help with my anxiety")

    @property
    def search_text(self) -> str:
        return " ".join(self.keywords)


def find_concerns(text: str) -> tuple[list[str], list[str]]:
    """(side effects asked about, problems the drug should treat), as lexicon names.

    Uses the aspect module's mention rules: "help with insomnia" names the problem
    treated; "make you feel groggy" and "without nausea" are side-effect concerns.
    """
    side_effects, treated = [], []
    for mention in symptom_mentions(clean_text(text or "")):
        target = treated if mention.treated else side_effects
        if mention.name not in target:
            target.append(mention.name)
    return side_effects, treated


def parse_query(text: str, catalog: DrugCatalog | None = None, drug: str | None = None) -> ParsedQuery:
    """Split a question into drug, search words and concerns.

    `drug` overrides detection (for a drug picked in the app). The drug's own words
    are removed from the search words, since results are already limited to it.
    """
    text = clean_text(text or "")
    tokens = tokenize(text)
    matches = catalog.find(text) if catalog else []
    chosen = None
    if drug:
        chosen = catalog.get(drug) if catalog else None
        name = chosen.name if chosen else clean_text(drug)
        drug_tokens = chosen.tokens if chosen else tuple(tokenize(name))
    elif matches:
        chosen = matches[0].drug
        name, drug_tokens = chosen.name, chosen.tokens
    else:
        name, drug_tokens = None, ()
    drop = set()
    if drug_tokens:
        n = len(drug_tokens)
        for i in range(len(tokens) - n + 1):
            if tuple(tokens[i:i + n]) == drug_tokens:
                drop.update(range(i, i + n))
    keywords = []
    for i, token in enumerate(tokens):
        if i in drop or token in STOPWORDS or (len(token) == 1 and token.isalpha()) or token in keywords:
            continue
        keywords.append(token)
    side_effects, treated = find_concerns(text)
    found = name is not None and any(m.drug.name == name for m in matches)
    others = [m.drug.name for m in matches if m.drug.name != name]
    return ParsedQuery(text, keywords, name, found, others, side_effects, treated)
