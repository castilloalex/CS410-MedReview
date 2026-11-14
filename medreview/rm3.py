from collections import Counter
from .text import tokenize
import math
from .text import clean_text

def search_rm3(index, query, drug=None, limit=10, feedback_docs=10, expansion_terms=10, original_weight=0.5):
    """Search reviews using BM25 followed by RM3 query expansion and return the highest-scoring results."""

    if limit < 1 or feedback_docs < 1 or expansion_terms < 1:
        raise ValueError("Limit, feedback_docs, and expansion_terms must be positive integers.")

    if not 0 <= original_weight <= 1:
        raise ValueError("Original weight must be between 0 and 1.")

    query_terms = tokenize(query)

    if not query_terms:
        return []

    #Step 1: Run the initial BM25 search.
    initial_results = index.search(query, drug=drug, limit=feedback_docs)

    if not initial_results:
        return []

    #Step 2: Collect weighted terms from the top-ranked reviews.
    feedback = Counter()
    total_score = sum(result["score"] for result in initial_results)

    for result in initial_results:
        terms = Counter(tokenize(result["review"]))
        weight = result["score"] / total_score
        document_length = result["length"]

        if document_length == 0:
            continue

        for term, freq in terms.items():
            term_probability = freq / document_length
            feedback[term] += weight * term_probability

    #Step 3: Select the most useful expansion terms.
    selected_terms = sorted(feedback, key=feedback.get, reverse=True)[:expansion_terms]
    feedback_total = sum(feedback[term] for term in selected_terms)
    if feedback_total == 0:
        return initial_results[:limit]

    feedback_model = {term: feedback[term] / feedback_total for term in selected_terms}

    #Step 4: Combine the original query with the feedback model.
    query_counts = Counter(query_terms)
    query_total = sum(query_counts.values())
    query_model = {term: count / query_total for term, count in query_counts.items()}
    
    combined_model = Counter()

    for term, prob in query_model.items():
        combined_model[term] += original_weight * prob

    for term, prob in feedback_model.items():
        combined_model[term] += (1 - original_weight) * prob

    #Step 5: Rank reviews using the expanded query.
    scores = Counter()

    n = index.metadata["documents"]
    average_length = index.metadata["average_document_length"]

    if n == 0 or average_length == 0:
        return []

    k1 = 1.2
    b = 0.75

    for term, weight in combined_model.items():
        stats = index.term_statistics(term)
        if not stats:
            continue

        df = stats["df"]
        idf = math.log1p((n - df + 0.5) / (df + 0.5))

        for posting in index.postings(term):
            doc_id = posting["doc_id"]
            document = index.document(doc_id)

            if drug is not None:
                if document["drug_key"] != clean_text(drug).casefold():
                    continue


            tf = posting["tf"]
            length = document["length"]

            denominator = tf + k1 * (1 - b + b * length / average_length)

            bm25_score = idf * tf * (k1 + 1) / denominator
            scores[doc_id] += weight * bm25_score
            
    #Step 6: Return the top-ranked reviews.
    ranked_ids = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:limit]
    results = []
    for doc_id in ranked_ids:
        document = index.document(doc_id)
        results.append({**document, "score": scores[doc_id]})
    return results