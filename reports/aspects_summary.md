## Aspect classification and side-effect extraction

This part labels each review sentence as `effectiveness`, `side_effects`, or `neither`, and uses those labels to list the side effects most often reported for each drug. The code is in `medreview/aspects`; usage is described in `medreview/aspects/README.md`.

## What was built

- **Sentence splitter.** A regex splitter that breaks each cleaned review into sentences.
- **Keyword classifier (baseline).** Counts effectiveness cues ("helped", "works") and side-effect cues ("made me", "side effect") in a sentence. A named symptom such as "nausea" counts double toward side effects, unless it is the problem being treated ("helps my nausea", or it matches the review's condition), in which case it counts toward effectiveness. No hits means `neither`.
- **Logistic regression classifier.** TF-IDF features (word 1-2 grams and character 3-5 grams) into a logistic regression model.
- **Side-effect extraction.** In sentences labeled `side_effects`, the system looks for about 40 side effects and their variants, skips negated mentions ("no nausea") and treated conditions, and counts how many reviews of each drug mention each one.

## Evaluation setup

We sampled 120 sentences from the test split. Half were chosen so that each keyword-predicted class appears, and half were chosen at random. Each sentence was given one label for what it is mainly about. The labels were drafted by Claude and reviewed by Alex. The label counts are 53 `neither`, 35 `side_effects`, and 32 `effectiveness`.

Three methods were compared on these labels:

- `keyword`: keyword matching with no training.
- `logreg-cv`: logistic regression trained and tested on the 120 labels with stratified 5-fold cross-validation.
- `logreg-weak`: logistic regression trained on sentences from 20,000 training-split reviews that the keyword classifier labeled automatically, then tested on the 120 labels. The labeled sentences come from the test split, so there is no overlap.

## Results

| Method | Class | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| keyword | effectiveness | 0.583 | 0.438 | 0.500 |
| keyword | side_effects | 0.625 | 0.571 | 0.597 |
| keyword | neither | 0.656 | 0.792 | 0.718 |
| keyword | **macro average** | 0.622 | 0.600 | **0.605** |
| logreg-cv | effectiveness | 0.524 | 0.344 | 0.415 |
| logreg-cv | side_effects | 0.562 | 0.514 | 0.537 |
| logreg-cv | neither | 0.522 | 0.660 | 0.583 |
| logreg-cv | **macro average** | 0.536 | 0.506 | **0.512** |
| logreg-weak | effectiveness | 0.591 | 0.406 | 0.481 |
| logreg-weak | side_effects | 0.656 | 0.600 | 0.627 |
| logreg-weak | neither | 0.652 | 0.811 | 0.723 |
| logreg-weak | **macro average** | 0.633 | 0.606 | **0.610** |

Accuracy is 0.633 for `keyword`, 0.533 for `logreg-cv`, and 0.642 for `logreg-weak`.

Keyword matching and the weakly trained model perform about the same, which is expected because the second learns from the first's labels. Logistic regression trained only on the hand labels is weakest; about 96 training sentences per fold is too few. With 120 sentences, differences of a few points are not reliable. Effectiveness has the lowest recall for every method.

## Example errors (keyword classifier)

The most common error is a missed sentence: 22 of the 44 keyword errors are effectiveness or side-effect sentences labeled `neither` because they contain no listed cue.

| Sentence | Correct | Predicted | Why |
| --- | --- | --- | --- |
| "After the first 2 months or so, my periods are lighter and shorter, and I rarely have cramps." | effectiveness | side_effects | A symptom word ("cramps") that is improving is read as a side effect. |
| "I'm not crying as much." | effectiveness | side_effects | Improvement is expressed as less of a symptom. |
| "Plus, anorgasmia so I discontinued it after 9 months." | side_effects | neither | The side effect is not in the fixed list. |
| "The first month was rough, my skin was really dry and it peeled alot." | side_effects | neither | Described in everyday words that the list does not cover. |
| "Now I absolutely LOVE it as a birth control." | neither | effectiveness | General opinion contains an effectiveness cue. |

## Side-effect table

`data/aspects/side_effects_by_drug.csv` covers 2,592 drugs (24,922 drug and side-effect pairs) built from all cleaned reviews with the keyword classifier. For example, the top reported side effects for Metformin are diarrhea (22% of its 338 reviews), weight loss (22%), and nausea (15%); for Trazodone the top one is drowsiness (19% of 786 reviews). Some entries are not true side effects: Trazodone also lists anxiety and depression, which many reviewers had before taking it. Weight loss for Metformin may be a wanted effect. These percentages are mentions in reviews, not clinical rates.

## Remaining work

The aspect code, evaluation, and side-effect table promised in the proposal are complete. What remains for this part:

- Final review of the 120 labels by Alex; re-run `python -m medreview.aspects evaluate` and update the results table if any label changes.
- Hand off to Lee for the summaries and Streamlit app: `classify_review` and `SideEffectTable` are ready to use (see `medreview/aspects/README.md`). The app should use the keyword classifier, which needs no model file and performs as well as the trained model.
- Add this part's files to the team's public repository before the Dec 10 submission.

## Challenges

- With only 120 labeled sentences, differences of a few points between methods are not reliable, and logistic regression trained on the hand labels alone has too little data.
- Many sentences describe effects in everyday words with no fixed cue, so keyword matching misses them.
- Symptoms the reviewer had before taking the drug can be counted as side effects.

Results are saved in `reports/aspect_eval.csv` and `reports/aspect_predictions.csv`.
