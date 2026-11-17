## Summaries and the Streamlit app

This part answers a question such as "Does trazodone help with sleep and does it make you
feel groggy?" with the matching reviews and a per-drug summary that reports effectiveness
and side effects separately. The code is in `medreview/summary` and `app.py`; usage is in
`medreview/summary/README.md`. The checks below are saved in `reports/summary_checks.json`
by `python -m medreview.summary.evaluate`.

## What was built

- **Question parsing.** Finds the drug named in the question among the 3,540 drug names
  in the index and keeps the other content words as search words ("help sleep groggy").
  Side effects in the question are found with the aspect lexicon, so "groggy" becomes a
  question about drowsiness, while "help with insomnia" names the problem being treated.
- **Retrieval for one drug.** BM25, TF-IDF, or RM3, using the Part 2 functions unchanged.
  TF-IDF and RM3 score every review containing a query word before filtering by drug, so a
  drug-filtered RM3 query took about 14 seconds. They now receive a `DrugView` that holds
  only the drug's postings but the full index's statistics, so the results are unchanged.
- **Review analysis.** `classify_review` labels each sentence. The logistic regression
  sentiment model reads only the effectiveness sentences. Side-effect sentences are
  matched against the lexicon, and each side effect is counted as reported, or as denied
  when negated ("no grogginess").
- **Summaries.** For the matched reviews and for all of the drug's reviews: effectiveness
  sentiment, side effects reported and denied, each side effect's rate compared with all
  drugs (`lift` from the side-effect table), and `SideEffectTable.top_list`. With no drug
  named, the top 200 BM25 results across all drugs are grouped by drug.
- **App.** A Streamlit page with the question box, sidebar settings (drug, condition,
  ranking method, number of reviews, sentiment model), the summary, and the ranked reviews
  with sentences highlighted by aspect. `python -m medreview.summary` prints the same
  summary in the terminal.

## Checks

| Check | Result |
| --- | --- |
| TF-IDF through `DrugView` vs `search_tfidf` on the full index (25 judged queries, top 10) | 25 of 25 identical; 6.7 s on the full index, 0.36 s with `DrugView` |
| RM3 through `DrugView` vs `search_rm3` on the full index | 25 of 25 identical; 345 s on the full index, 0.56 s with `DrugView` |
| Drug-wide side-effect counts vs `data/aspects/side_effects_by_drug.csv` | 24,922 of 24,922 drug and side-effect pairs equal |
| Tests (`tests/test_summary.py`, including the app on a small dataset) | 8 new tests; all 19 project tests pass on Python 3.10 with Streamlit 1.40 and 1.65 |

## Effectiveness-only sentiment

The summaries score only the sentences about effectiveness, so that side effects do not
pull the result down. On the test split (53,753 reviews; the model was trained on the
train split), 38,458 reviews have at least one effectiveness sentence.

| Text the model reads | Accuracy vs rating label | Macro-F1 |
| --- | ---: | ---: |
| Whole review (all test reviews) | 0.781 | 0.633 |
| Effectiveness sentences only (38,458 reviews) | 0.724 | 0.530 |

Agreement with the rating is lower, as expected, since the rating also reflects side
effects. Among the 20,747 of these reviews that report a side effect, 59.3% read positive
from the whole review and 65.4% from the effectiveness sentences alone.

The effectiveness-only reading is noisy, though. Of the reviews rated 1-4 that report a
side effect, 1,783 of 5,328 read positive about effectiveness. In 20 random examples of
these (checked by Claude, not yet by a teammate), 3 clearly say the drug worked and 2 say
it partly or briefly worked. The other 15 credit another treatment, say the drug did not
work, or are sentences that the keyword classifier labeled as effectiveness but that are
about something else. The effectiveness bar should therefore be read as approximate. A
sentiment model trained on effectiveness sentences would be the next improvement.

## Example

"Does trazodone help with sleep and does it make you feel groggy?" with BM25 and 50 reviews:

- Drug: Trazodone (786 reviews). Search words: help, sleep, groggy. Side effect asked
  about: drowsiness.
- Effectiveness, in the 39 matched reviews that discuss it: 67% positive, 13% mixed or
  neutral, 21% negative. Across all Trazodone reviews: 65%, 11%, and 24%.
- Drowsiness: reported in 146 of 786 reviews (18.6%), 5.0 times the rate across all drugs;
  33 reviews say they did not have it. In the 50 matched reviews, 33 report it and 13 say
  they did not.

## Known limits

- The matched reviews were found by searching for the concern, so they mention it more
  often than reviews in general. The app shows the drug-wide share next to it.
- Negation is the lexicon's three-word rule: "don't like the groggy feeling" reads as not
  having it.
- A drug name made only of everyday words with fewer than 50 reviews ("RID", "My Way") is
  found only with its exact capitalization. Brand and generic names are separate drugs.
- Counts are mentions in reviews, not clinical rates.
- The first question about a drug analyzes all of its reviews: about 1 second for
  Trazodone and 6 seconds for Levonorgestrel (4,065 reviews) in testing, mostly in the
  keyword classifier. Later questions about the same drug reuse the result.
