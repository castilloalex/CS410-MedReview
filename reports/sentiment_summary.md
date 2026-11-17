## Sentiment classification

Each review is labeled `negative` (rating 1-4), `neutral` (5-6), or `positive` (7-10). As the proposal planned, we compare VADER and logistic regression on the test split using accuracy and macro-F1. The code is in `medreview/sentiment`; usage is described in `medreview/sentiment/README.md`.

## Methods

- **VADER.** A sentiment lexicon with rules, using the recommended compound-score cutoffs of +/-0.05. No training.
- **Logistic regression.** TF-IDF word 1-2 grams (min_df 2, sublinear tf) with balanced class weights. Trained on the 128,722 training reviews. `C` was chosen from {0.25, 1, 4, 16} by macro-F1 on a 10% validation split grouped by review text; `C = 1` was best (validation macro-F1 0.633).

## Results (test split, 53,753 reviews)

| Method | Accuracy | Macro-F1 | F1 negative | F1 neutral | F1 positive |
| --- | ---: | ---: | ---: | ---: | ---: |
| VADER | 0.577 | 0.413 | 0.504 | 0.052 | 0.684 |
| Logistic regression | **0.780** | **0.632** | 0.732 | 0.290 | 0.875 |

The test split is 66% positive, 25% negative, and 9% neutral. Per-class precision and recall are in `reports/sentiment_eval.csv`, and confusion matrices are in `reports/sentiment_confusion.csv`.

## Observations

- Logistic regression is better on every class. VADER labels 37% of positive reviews as negative (13,087 of 35,429). Drug reviews usually describe the symptoms or side effects the reviewer had, and VADER scores those words as negative even when the drug helped.
- Neutral is the hardest class for both methods. A 5-6 rating usually means mixed experience rather than no opinion, so the text contains both positive and negative language.
- Many reviews appear under several drug names with the same text. Splitting the validation set at random let copies leak across and raised validation macro-F1 to 0.84, so validation is grouped by text. The cleaned test split does not share text with training.
- For an app that shows only positive and negative reviews, the 3-class model can still be used: show `positive` and `negative` predictions and hide `neutral`.
