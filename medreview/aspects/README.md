# Aspects: sentence classification and side-effect extraction

Labels each review sentence as `effectiveness`, `side_effects`, or `neither`, and
extracts each drug's most commonly reported side effects.

Needs `pandas`, `scikit-learn`, and `joblib` (`pip install -r requirements.txt`).
Run commands from the `MedReview` folder, after `python -m medreview prepare`.

## Use from the app

```python
from medreview.aspects import classify_review, extract_side_effects, SideEffectTable

classify_review("It helped me sleep. But I felt groggy all morning.", condition="Insomnia")
# [("It helped me sleep.", "effectiveness"), ("But I felt groggy all morning.", "side_effects")]

extract_side_effects(review_text, condition="Insomnia")      # e.g. ["drowsiness"]

table = SideEffectTable.load("data/aspects/side_effects_by_drug.csv")
table.top_list("Trazodone", k=5)   # [("drowsiness", 0.186), ...] share of the drug's reviews
table.top("Trazodone", k=5)        # DataFrame: n_reviews, drug_reviews, pct, lift, treated
```

`condition` is optional everywhere; passing the review's condition stops the
treated problem (e.g. "anxiety" for an anxiety drug) from counting as a side effect.
Drug names match exactly, ignoring case, as in the search index.

To use the logistic regression model instead of keywords:

```python
from medreview.aspects import LogRegAspectClassifier
clf = LogRegAspectClassifier.load("data/aspects/aspect_logreg.joblib")
classify_review(review_text, clf)
```

## Commands

| Command | What it does |
| --- | --- |
| `python -m medreview.aspects sample` | Writes 120 test-split sentences to `data/labels/aspect_labels.csv` for hand labeling. Refuses to overwrite without `--force`. |
| `python -m medreview.aspects evaluate` | Compares the methods on the hand labels. Writes `reports/aspect_eval.csv`, `reports/aspect_predictions.csv`, and the model `data/aspects/aspect_logreg.joblib`. Takes about a minute. |
| `python -m medreview.aspects side-effects` | Builds `data/aspects/side_effects_by_drug.csv` from all cleaned reviews. Takes a few minutes. |
| `python -m medreview.aspects classify "text"` | Prints the label of each sentence. |
| `python -m unittest tests.test_aspects -v` | Runs the tests. |

## Hand labeling

Open `data/labels/aspect_labels.csv` and fill in the `label` column with
`effectiveness`, `side_effects`, or `neither` (`e`, `s`, `n` also work). Half the
sentences were chosen so each keyword-predicted class appears and half at random;
the keyword prediction is not in the file, so labeling is blind. Suggested rule:
label what the sentence is mainly about. "No side effects" is `side_effects`;
"it did nothing" is `effectiveness`; dosage, history, cost, and general opinion
are `neither`.

## Evaluation design

`evaluate` reports per-class precision, recall, and F1 for three methods:

- `keyword`: keyword matching, no training.
- `logreg-cv`: logistic regression with stratified 5-fold cross-validation on the
  hand labels (trained on about 96 sentences per fold).
- `logreg-weak`: logistic regression trained on about 118,000 training-split
  sentences that the keyword classifier labeled automatically, tested on the hand
  labels. Hand-labeled sentences come from the test split, so there is no overlap.

## How it works

| File | Role |
| --- | --- |
| `sentences.py` | Regex sentence splitter. |
| `lexicon.py` | Editable keyword lists: effectiveness cues, side-effect cues, about 40 side effects with their variants, negations, and condition hints. |
| `mentions.py` | Finds side-effect terms and marks each as negated ("no nausea") or treated ("helps my nausea", or matching the review's condition). |
| `keywords.py` | Keyword classifier. Counts cue hits per aspect; a side-effect term counts double; a treated term counts toward effectiveness; no hits means `neither`; ties go to side effects. |
| `logreg.py` | TF-IDF (word 1-2 grams and character 3-5 grams) into logistic regression. |
| `side_effects.py` | In sentences labeled `side_effects`, collects terms that are not negated or treated, then counts reviews per drug. |

Side-effect table columns: `pct` is the share of the drug's reviews mentioning the
side effect, and `lift` is that share divided by the all-drug rate. `treated` marks
terms matching a condition that makes up at least 10% of the drug's reviews; `top()`
hides them unless `include_treated=True`.

## Known limits

- The side-effect list is fixed, so unlisted effects are missed.
- Symptoms the reviewer had before the drug can be counted when the sentence gives
  no treating cue and the condition field does not match.
- Brand and generic names are separate drugs.
- Counts are mentions in reviews, not clinical incidence.
