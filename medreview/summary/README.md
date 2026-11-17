# Summaries: concern-focused search, per-drug summaries, and the app

Answers questions like "Does trazodone help with sleep and does it make you feel
groggy?" with the matching reviews and a summary that keeps the two aspects apart:
whether the drug worked, and which side effects people report.

Needs the index from `python -m medreview all` and the packages in `requirements.txt`.
The logistic regression sentiment model (`python -m medreview.sentiment train`) is
used when it exists; otherwise VADER. Run commands from the `MedReview` folder.

## Commands

| Command | What it does |
| --- | --- |
| `streamlit run app.py` | Opens the app in the browser. |
| `python -m medreview.summary "Does trazodone help with sleep and does it make you feel groggy?"` | Prints the same summary in the terminal. Options: `--drug`, `--condition`, `--method bm25/tfidf/rm3`, `--limit`, `--sentiment auto/logreg/vader`. |
| `python -m medreview.summary "sleep without grogginess"` | No drug named: compares the drugs behind the best-matching reviews. |
| `python -m medreview.summary.evaluate` | Runs the checks below and writes `reports/summary_checks.json`. Takes about 10 minutes. |
| `python -m unittest tests.test_summary -v` | Runs the tests, including the app on a small dataset. No downloaded data needed. |

## Use from code

```python
from medreview.index import SearchIndex
from medreview.aspects import SideEffectTable
from medreview.summary import load_sentiment, summarize_question

with SearchIndex("data/index/reviews.sqlite") as index:
    sentiment, method = load_sentiment()            # logreg if trained, else VADER
    table = SideEffectTable.load("data/aspects/side_effects_by_drug.csv")
    parsed, s = summarize_question(index, "Does trazodone make you groggy?", table, sentiment)

parsed.drug, parsed.keywords, parsed.side_effects   # "Trazodone", ["groggy"], ["drowsiness"]
s.effectiveness.share("positive")                    # effectiveness sentiment in the matched reviews
s.concerns[0].reported, s.concerns[0].denied         # Trazodone reviews reporting / denying drowsiness
s.side_effects[0].name, s.side_effects[0].share      # most reported side effect in the matched reviews
s.reviews[0].sentences                               # [(sentence, aspect), ...] for the top review
```

The pieces can be used separately: `parse_query`, `retrieve` (BM25, TF-IDF or RM3),
`build_profile` (every review of a drug, analyzed), `summarize`, and `compare_drugs`.

## How a summary is made

1. **Question.** `parse_query` finds the drug name as whole words, preferring the longest
   name ("Plan B One-Step" over "Plan B"), drops question words and the drug's own words,
   and keeps the rest as search words.
   Side effects in the question are found with the aspect lexicon, so "groggy" is a
   question about drowsiness while "help with insomnia" names the problem treated.
2. **Search.** The search words go to BM25 (`SearchIndex.search`), `search_tfidf`, or
   `search_rm3`, limited to the drug. TF-IDF and RM3 get a `DrugView`, which hands them
   only that drug's postings; their code is unchanged and so are their results.
3. **Reviews.** `classify_review` labels each sentence. The sentiment model reads only
   the effectiveness sentences of each review. Side-effect sentences are matched against
   the lexicon: a mention is reported, or denied when negated ("no grogginess"); the
   problem being treated is skipped. Reported side effects use the same rule as
   `extract_side_effects`, so the drug's counts equal `side_effects_by_drug.csv`.
4. **Summary.** For the matched reviews: effectiveness sentiment, side effects reported,
   and reviews saying they had none. For the whole drug: the same sentiment over all its
   reviews, how many report or deny each side effect asked about, `lift` from the
   side-effect table, and `SideEffectTable.top_list` for its most reported side effects.
5. **No drug named.** BM25 searches all reviews, and the top 200 are grouped by drug.

| File | Role |
| --- | --- |
| `query.py` | `DrugCatalog` (drug names in the index) and `parse_query`. |
| `retrieval.py` | `DrugView` and `retrieve` for the three ranking methods. |
| `analysis.py` | Per-review analysis, drug profiles, `summarize`, `compare_drugs`. |
| `pipeline.py` | `load_sentiment`, `summarize_question`. |
| `render.py` | HTML for the app; review text is always escaped. |
| `__main__.py`, `evaluate.py` | Command line and checks. |
| `../../app.py` | The Streamlit app. |

## Known limits

- Matched reviews were found by searching for the concern, so they mention it more often
  than reviews in general. The app shows the drug-wide share next to it for that reason.
- A drug name made only of everyday words with fewer than 50 reviews ("RID", "My Way")
  is found only when written with its exact capitalization.
- Negation is the lexicon's three-word rule: "don't like the groggy feeling" counts as
  not having it.
- The effectiveness reading is approximate. The sentiment model was trained on whole
  reviews and here reads a few sentences, and some sentences the keyword classifier labels
  as effectiveness are about another treatment. See `reports/summaries_summary.md`.
- Brand and generic names are separate drugs. Counts are mentions, not clinical rates.
