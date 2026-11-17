# Sentiment: VADER vs logistic regression

Labels a whole review `negative`, `neutral`, or `positive`. Labels come from the
user's 1-10 rating: 1-4 negative, 5-6 neutral, 7-10 positive.

Needs `pandas`, `scikit-learn`, `joblib`, and `vaderSentiment` (`pip install -r requirements.txt`).
Run commands from the `MedReview` folder, after `python -m medreview prepare`.

## Commands

```
python -m medreview.sentiment train      # tune C, fit on the train split, save data/sentiment/sentiment_logreg.joblib
python -m medreview.sentiment evaluate   # both methods on the test split -> reports/sentiment_eval.csv
python -m medreview.sentiment predict "It worked but the nausea was awful." [--method vader]
```

`train` takes about 3 minutes and `evaluate` takes about 30 seconds. The model file is gitignored; run `train` to create it.

## Use from the app

```python
from medreview.sentiment import predict_sentiment

predict_sentiment(review_text)                  # logistic regression (needs the trained model)
predict_sentiment(review_text, method="vader")  # no model file needed
```

## Notes

- `train` tunes `C` on 10% of the train split.
- The test split is used only by `evaluate`.
