# AI-assisted code
# Model: Opus 5.5 (max effort)
"""Run the team's retrieval methods (BM25, TF-IDF, RM3) for one drug, fast.

`search_tfidf` and `search_rm3` score every review in the corpus that contains a
query word and drop the other drugs afterwards. RM3 also expands the query with
feedback words such as "i" and "the", which appear in most reviews, so a single
drug-filtered RM3 search takes about 14 seconds on the full index.

`DrugView` passes those functions, unchanged, only the selected drug's postings
and keeps that drug's reviews in memory. Corpus statistics (number of reviews,
document frequencies, average length) still come from the full index, so scores
and rankings are exactly what the functions return on the full index. The tests
check this.
"""
from ..rm3 import search_rm3
from ..text import clean_text
from ..tfidf import search_tfidf

METHODS = {"bm25": "BM25", "tfidf": "TF-IDF", "rm3": "RM3"}


class DrugView:
    """The parts of the SearchIndex interface that search_tfidf and search_rm3 use,
    restricted to one drug's reviews."""

    def __init__(self, index, drug: str):
        self.index = index
        self.metadata = index.metadata
        self.drug_key = clean_text(drug).casefold()
        rows = index.connection.execute("SELECT * FROM documents WHERE drug_key=? ORDER BY doc_id", (self.drug_key,))
        self._documents = {row["doc_id"]: dict(row) for row in rows}

    def __len__(self):
        return len(self._documents)

    def documents(self) -> list[dict]:
        """All of the drug's reviews (copies), in review-ID order."""
        return [dict(d) for d in self._documents.values()]

    def document(self, doc_id):
        doc = self._documents.get(doc_id)
        return dict(doc) if doc is not None else self.index.document(doc_id)

    def term_statistics(self, term):
        return self.index.term_statistics(term)

    def postings(self, term):
        return [dict(row) for row in self.index.connection.execute(
            "SELECT p.doc_id, p.tf FROM terms t JOIN postings p USING(term_id) JOIN documents d USING(doc_id)"
            " WHERE t.term=? AND d.drug_key=? ORDER BY p.doc_id", (term, self.drug_key))]

    def document_terms(self, doc_id):
        return self.index.document_terms(doc_id)

    def search(self, query, drug=None, limit=10, **kwargs):
        return self.index.search(query, drug=drug, limit=limit, **kwargs)


def retrieve(index, query: str, drug: str | None = None, method: str = "bm25", limit: int = 50,
             view: DrugView | None = None) -> list[dict]:
    """The top `limit` reviews for `query`: document dicts from the index plus "score".

    `method` is "bm25" (SearchIndex.search), "tfidf" (search_tfidf) or "rm3"
    (search_rm3). With a drug, TF-IDF and RM3 run on a DrugView (pass `view` to reuse
    one). Without a drug they search the whole corpus, which is slow for RM3.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; use one of {sorted(METHODS)}")
    if not (query or "").strip():
        return []
    if method == "bm25":
        return index.search(query, drug=drug, limit=limit)
    source = view if view is not None else (DrugView(index, drug) if drug is not None else index)
    function = search_tfidf if method == "tfidf" else search_rm3
    return function(source, query, drug=drug, limit=limit)
