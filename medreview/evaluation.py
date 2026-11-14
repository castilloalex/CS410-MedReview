import numpy as np
import pandas as pd
from pathlib import Path

def compute_ndcg(results, qrels, k=10):
    def dcg(relevances):
        # return sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevances[:k]))
        dcg_simple = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances[:k]))
        return dcg_simple

    ndcg_scores = []
    for qid, query_results in results.items():
        if qid not in qrels:
            # print(f"Query {qid} not found in qrels")
            continue
        relevances_current = [qrels[qid].get(docid, 0) for docid, _ in query_results]
        idcg = dcg(sorted(qrels[qid].values(), reverse=True)[:k])
        if idcg == 0:
            print(f"IDCG is 0 for query {qid}")
            continue
        ndcg_scores.append(dcg(relevances_current) / idcg)

    if not ndcg_scores:
        print("No valid NDCG scores computed")
        return 0.0
    return np.mean(ndcg_scores)

def compute_precision(results, qrels, k=10):
    precision_scores = []
    for qid, query_results in results.items():
        if qid not in qrels:
            # print(f"Query {qid} not found in qrels")
            continue
        relevances_current = [qrels[qid].get(docid, 0) for docid, _ in query_results]
        relevant_count = sum(1 for rel in relevances_current[:k] if rel > 0)

        precision_scores.append(relevant_count / k)

    if not precision_scores:
        print("No valid Precision scores computed")
        return 0.0
    return np.mean(precision_scores)


def load_qrels_and_results(file_path):
    df = pd.read_excel(file_path)

    # Store relevance judgments
    qrels = {}

    for _, row in df.iterrows():
        qid = str(row["Query ID"])
        docid = str(int(row["Review ID"]))
        relevance = int(row["Relevance"])

        if qid not in qrels:
            qrels[qid] = {}

        qrels[qid][docid] = relevance

    # Build ranked results for each retrieval method
    methods = {
        "TF-IDF": "TF-IDF Rank",
        "BM25": "BM25 Rank",
        "RM3": "RM3 Rank"
    }

    all_results = {}

    for method, rank_column in methods.items():
        results = {qid: [] for qid in qrels}

        ranked_df = df.dropna(subset=[rank_column])

        for qid, group in ranked_df.groupby("Query ID"):
            group = group.sort_values(rank_column)

            results[str(qid)] = [
                (str(int(row["Review ID"])), row[rank_column])
                for _, row in group.iterrows()
            ]

        all_results[method] = results

    return qrels, all_results


def main():
    file_path = (
        Path(__file__).resolve().parent.parent
        / "reports"
        / "relevance_judgments_pool.xlsx"
    )

    qrels, all_results = load_qrels_and_results(file_path)

    print(f"Number of queries: {len(qrels)}")
    print(f"Number of relevance judgments: {sum(len(v) for v in qrels.values())}")

    top_k = 10

    print("\n" + "=" * 55)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 55)

    for method, results in all_results.items():
        ndcg = compute_ndcg(results, qrels, k=top_k)
        precision = compute_precision(results, qrels, k=top_k)

        print(f"\n{method}")
        print("-" * 30)
        print(f"Precision@{top_k}: {precision:.4f}")
        print(f"nDCG@{top_k}:       {ndcg:.4f}")


if __name__ == "__main__":
    main()
