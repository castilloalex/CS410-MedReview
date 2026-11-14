# MedReview — Part 1

Clean the UCI Drugs.com reviews and build a searchable index for the team's retrieval, sentiment, and summarization work.

Requires **Python 3.10+**. Part 1 (cleaning, index, BM25 search) needs no third-party packages. The aspect work and the retrieval evaluation need the packages in `requirements.txt` (`pip install -r requirements.txt`). Run commands from the `MedReview` folder.

## Commands

### 1. Prepare everything

```powershell
python -m medreview all
```

Downloads the dataset if needed, cleans the reviews, and builds the search index. Running it again replaces the cleaned files and index.

For example, cleaning changes `don&#039;t` to `don't` and removes duplicate reviews. The index records which reviews contain each word.

### 2. Search reviews

```powershell
python -m medreview search "sleep groggy" --drug "Trazodone" --limit 5
```

Finds up to five Trazodone reviews about sleep or grogginess, ranked by BM25 relevance. Results include the review text, user rating, and search score. It does not change files.

- `"sleep groggy"`: words to search for.
- `--drug`: filter by drug name, ignoring case. Omit it to search all drugs.
- `--limit`: maximum number of results.

For example, a result may describe better sleep but feeling groggy the next day. The search score measures relevance, not drug quality.

### 3. Test the code

```powershell
python -m unittest discover -s tests -v
```

Finds tests in `tests` and prints detailed results. Uses temporary sample data to check cleaning, indexing, and search; it does not change the real dataset.

For example, a test checks that a training review is removed when its text also appears in the test set. Successful output ends with `OK`.

### 4. Validate the full dataset and index

```powershell
python -m medreview.validate
```

Checks the generated files: review IDs, counts, word frequencies, database integrity, and train/test text overlap.

For example, every cleaned review must appear in the index. If all checks pass, it writes `reports/part1_results.json`. This checks correctness, not search quality.

## Run individual steps

| Command | Purpose and example use |
| --- | --- |
| `python -m medreview prepare` | Download if needed and clean data. Use after changing cleaning rules. Does not update the index. |
| `python -m medreview build` | Build the index from existing cleaned files. Use after cleaning or changing tokenization. |
| `python -m medreview build --splits train --index data/index/train.sqlite` | Build a separate training-only index, for experiments that exclude test reviews. |

To search the training-only index:

```powershell
python -m medreview search "sleep" --drug "Trazodone" --index data/index/train.sqlite
```

## Output files

| Location | Contents |
| --- | --- |
| `data/raw/uci_drug_reviews.zip` | Original dataset |
| `data/processed/train.jsonl`, `test.jsonl` | Cleaned reviews; one JSON record per line |
| `data/processed/cleaning_report.json` | Cleaning counts and file hashes |
| `data/index/reviews.sqlite` | Search index |
| `data/index/reviews.json` | Index statistics |
| `reports/part1_results.json` | Full-data validation results |
| `reports/part1_summary.md` | Project progress summary |

Generated data and indexes are gitignored. Teammates can recreate them with `python -m medreview all`.

## Cleaning rules and results

- Remove HTML tags, decode HTML characters, and normalize whitespace.
- Preserve readable text, punctuation, negation, and dosage information.
- Replace missing or malformed conditions with `null`; standardize dates.
- Remove invalid numeric records, missing drug names, and reviews with no searchable tokens.
- Remove duplicate drug/review pairs. Keep test records and remove matching training text, even across drug names, to prevent exact-text leakage.
- Preserve original review IDs and train/test assignments. Near-duplicates are not detected.

The completed run retained **182,475 of 215,063 reviews**: 128,722 training and 53,753 test reviews. It removed 32,514 training reviews whose text appeared in the test set. The index contains 54,794 distinct terms.

## Team integration

```python
from medreview.index import SearchIndex

with SearchIndex("data/index/reviews.sqlite") as index:
    hits = index.search("sleep groggy", drug="Trazodone", limit=5)
    for hit in hits:
        print(hit["doc_id"], hit["review"])
```

JSONL `review_id` and search `doc_id` identify the same review. Records include drug name, condition, review text, rating, date, split, helpfulness count, and source location.

For retrieval experiments, use `term_statistics(term)` for document/collection frequencies, `postings(term)` for matching review IDs and word counts, and `document_terms(doc_id)` for a review's word counts. Use `medreview.text.tokenize()` to keep query tokenization consistent with the index.

The default index searches both splits and scores review text only. Drug filtering matches names exactly, ignoring case; it does not resolve brand/generic aliases. BM25 uses `k1=1.2`, `b=0.75`, and global corpus statistics.

Train sentiment models and fit text features using **train.jsonl only**; reserve **test.jsonl** for evaluation. TF-IDF/RM3 experiments, relevance evaluation, classifiers, summaries, and the UI remain later work.

## Other parts of the project

### Retrieval methods and evaluation

| File | Role |
| --- | --- |
| `medreview/tfidf.py` | TF-IDF search over the index (`search_tfidf`). |
| `medreview/rm3.py` | BM25 followed by RM3 query expansion (`search_rm3`). |
| `medreview/compare.py` | `python -m medreview.compare` prints TF-IDF, BM25, and RM3 results side by side. |
| `medreview/pool.py` | `python -m medreview.pool` pools the top 10 results of each method for 25 queries into `reports/relevance_judgments_pool.csv` for relevance judging. |
| `medreview/evaluation.py` | `python -m medreview.evaluation` reports Precision@10 and nDCG@10 from the judged `reports/relevance_judgments_pool.xlsx`. |

### Aspect classification and side-effect extraction

Labels review sentences as effectiveness, side effects, or neither, and extracts each drug's commonly reported side effects. See [medreview/aspects/README.md](medreview/aspects/README.md).

## Data source

Kallumadi, S. and Gräßer, F. (2018). *Drug Review Dataset (Drugs.com)*, UCI Machine Learning Repository. [Dataset and citation](https://doi.org/10.24432/C5SK5S).

Reviews describe personal experiences and are not medical advice.
