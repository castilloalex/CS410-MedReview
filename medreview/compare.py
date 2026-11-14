from .index import SearchIndex
from .tfidf import search_tfidf
from .rm3 import search_rm3

def compare_search(query, drug=None, limit=10):
    index = SearchIndex("data/index/reviews.sqlite")

    try:
        methods = {"TF-IDF": search_tfidf,
                   "BM25": lambda idx, q, drug=None, limit=10: idx.search(q, drug=drug, limit=limit),
                    "RM3": search_rm3}
        for name, search_method in methods.items():
            print(f"\n{'='*60}")
            print(name)
            print('='*60)

            results = search_method(index, query, drug=drug, limit=limit)
            for rank, result in enumerate(results, start=1):
                print(f"\nRank: {rank} | " 
                      f"ID: {result['doc_id']} | "
                      f"Score: {result['score']:.4f}")
                print(result["review"])
    finally:
        index.close()

if __name__ == "__main__":
    compare_search("sleep groggy", drug="Trazodone", limit=10)