import math
from collections import defaultdict

from .text import tokenize

def search_tfidf(index, query, drug=None, limit=10):
    """Search reviews using TF-IDF and return the highest-scoring results."""

    if limit < 1:
        raise ValueError("Limit must be a positive integer.")

    query_terms = set(tokenize(query))

    if not query_terms:
        return []

    scores = defaultdict(float)

    n = index.metadata["documents"]
    if n == 0:
        return []

    for term in query_terms:
        stats = index.term_statistics(term)
        if not stats:
            continue

        df = stats["df"]
        idf = math.log((n+1) / (df +1)) + 1  # Adding 1 to avoid division by zero and smooth IDF

        for posting in index.postings(term):
            doc_id = posting["doc_id"]
            tf = posting["tf"]
            scores[doc_id] += tf * idf
    results = []

    for doc_id, score in scores.items():
        document = index.document(doc_id)
        if drug is not None:
            if document["drug_key"] != drug.casefold():
                continue
        results.append({**document, "score": score})
    
    results.sort(key=lambda result: (-result["score"], result["doc_id"]))
    return results[:limit]