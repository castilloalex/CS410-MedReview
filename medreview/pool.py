
import csv
from pathlib import Path

from .index import SearchIndex
from .tfidf import search_tfidf
from .rm3 import search_rm3


QUERIES = [
    ("Q01", "sleep groggy", "Trazodone"),
    ("Q02", "fall asleep stay asleep", "Zolpidem"),
    ("Q03", "sleepy during the day", "Mirtazapine"),
    ("Q04", "helps anxiety", "Buspirone"),
    ("Q05", "anxiety nausea", "Sertraline"),
    ("Q06", "improves depression mood", "Fluoxetine"),
    ("Q07", "insomnia trouble sleeping", "Bupropion"),
    ("Q08", "sweating night sweats", "Venlafaxine"),
    ("Q09", "nerve pain relief", "Gabapentin"),
    ("Q10", "weight gain swelling", "Pregabalin"),
    ("Q11", "prevents migraines", "Topiramate"),
    ("Q12", "brain fog memory problems", "Topiramate"),
    ("Q13", "lower blood sugar", "Metformin"),
    ("Q14", "diarrhea stomach upset", "Metformin"),
    ("Q15", "blood pressure control", "Lisinopril"),
    ("Q16", "dry cough", "Lisinopril"),
    ("Q17", "muscle pain aches", "Atorvastatin"),
    ("Q18", "fatigue energy improvement", "Levothyroxine"),
    ("Q19", "focus concentration", "Adderall"),
    ("Q20", "appetite loss weight loss", "Adderall"),
    ("Q21", "heartburn acid reflux relief", "Omeprazole"),
    ("Q22", "headache", "Omeprazole"),
    ("Q23", "allergy symptom relief", "Cetirizine"),
    ("Q24", "drowsy tired", "Cetirizine"),
    ("Q25", "weight gain appetite", "Quetiapine"),
]

# Relevance judgments already completed for Q01.
Q01_RELEVANT = {
    231359, 231269, 231585, 231845, 231859,
    231271, 231507, 231657, 231663, 231790,
    231920, 231600, 232022, 231770, 231991,
    231279, 232052, 231680, 231330,
}


def collect_results(index, query_id, query, drug):
    methods = {
        "TF-IDF": search_tfidf,
        "BM25": lambda idx, q, drug=None, limit=10:
            idx.search(q, drug=drug, limit=limit),
        "RM3": search_rm3,
    }

    rank_columns = {
        "TF-IDF": "TF-IDF Rank",
        "BM25": "BM25 Rank",
        "RM3": "RM3 Rank",
    }

    pooled = {}

    for method_name, search_method in methods.items():
        results = search_method(
            index,
            query,
            drug=drug,
            limit=10
        )

        for rank, result in enumerate(results, start=1):
            doc_id = result["doc_id"]

            if doc_id not in pooled:
                pooled[doc_id] = {
                    "Query ID": query_id,
                    "Query text": query,
                    "Drug filter": drug,
                    "Review ID": doc_id,
                    "Review text": result["review"],
                    "Condition": result.get("condition", ""),
                    "Found by": [],
                    "TF-IDF Rank": "",
                    "BM25 Rank": "",
                    "RM3 Rank": "",
                    "Relevance": "",
                    "Judge": "",
                    "Notes": "",
                }

            pooled[doc_id]["Found by"].append(method_name)
            pooled[doc_id][rank_columns[method_name]] = rank

            if query_id == "Q01" and doc_id in Q01_RELEVANT:
                pooled[doc_id]["Relevance"] = 1
                pooled[doc_id]["Judge"] = "Amy"

    rows = list(pooled.values())

    for row in rows:
        row["Found by"] = ", ".join(row["Found by"])

    return rows


def main():
    index = SearchIndex("data/index/reviews.sqlite")

    output_path = Path("reports/relevance_judgments_pool.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []

    try:
        for query_id, query, drug in QUERIES:
            rows = collect_results(index, query_id, query, drug)
            all_rows.extend(rows)

            print(
                f"{query_id}: {len(rows)} unique reviews "
                f"from the three retrieval methods"
            )

    finally:
        index.close()

    fieldnames = [
        "Query ID",
        "Query text",
        "Drug filter",
        "Review ID",
        "Review text",
        "Condition",
        "Found by",
        "TF-IDF Rank",
        "BM25 Rank",
        "RM3 Rank",
        "Relevance",
        "Judge",
        "Notes",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nTotal unique query-review pairs: {len(all_rows)}")
    print(f"Spreadsheet saved to: {output_path}")


if __name__ == "__main__":
    main()
