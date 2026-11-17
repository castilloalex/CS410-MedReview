# AI-assisted code
# Model: Opus 5.5 (medium effort)
"""Command line for the sentiment work. Run from the MedReview folder.

    python -m medreview.sentiment train       # tune C on a held-out part of train, fit, save
    python -m medreview.sentiment evaluate    # VADER vs logistic regression on the test split
    python -m medreview.sentiment predict "It worked but the nausea was awful."

Labels come from ratings: 1-4 negative, 5-6 neutral, 7-10 positive.
"""
import argparse
from pathlib import Path

import pandas as pd

from ..aspects.data import load_reviews
from .api import MODEL
from .labels import LABELS, rating_to_label
from .model import LogRegSentimentClassifier
from .vader import VaderSentimentClassifier


def load_split(data_dir, split):
    df = load_reviews(data_dir, (split,))
    df["label"] = df["rating"].map(rating_to_label)
    return df


def train(a):
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupShuffleSplit

    df = load_split(a.data_dir, "train")
    print(f"{len(df)} training reviews: {df['label'].value_counts().to_dict()}")
    # Many reviews are posted under several drug names (brand and generic) with identical text.
    # Group by text so copies of one review never land on both sides of the validation split.
    split = GroupShuffleSplit(n_splits=1, test_size=a.val_size, random_state=a.seed)
    tr_idx, val_idx = next(split.split(df, groups=df["review"]))
    tr, val = df.iloc[tr_idx], df.iloc[val_idx]
    best_c, best_f1 = None, -1.0
    for c in a.C:
        pred = LogRegSentimentClassifier(C=c).fit(tr["review"], tr["label"]).predict(val["review"])
        f1 = f1_score(val["label"], pred, labels=LABELS, average="macro")
        print(f"  C={c:<6} validation macro-F1 {f1:.4f}")
        if f1 > best_f1:
            best_c, best_f1 = c, f1
    print(f"Best C={best_c}; refitting on all {len(df)} training reviews")
    model = LogRegSentimentClassifier(C=best_c).fit(df["review"], df["label"])
    a.model.parent.mkdir(parents=True, exist_ok=True)
    model.save(a.model)
    print(f"Saved {a.model}")


def evaluate(a):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

    if not a.model.exists():
        raise SystemExit(f"{a.model} not found; run `python -m medreview.sentiment train` first.")
    test = load_split(a.data_dir, "test")
    y = test["label"].tolist()
    print(f"{len(test)} test reviews: {test['label'].value_counts().to_dict()}\n")
    preds = {
        "vader": VaderSentimentClassifier().predict(test["review"]),
        "logreg": LogRegSentimentClassifier.load(a.model).predict(test["review"]),
    }
    rows, cms = [], []
    for name, p in preds.items():
        print(f"=== {name} ===")
        print(classification_report(y, p, labels=LABELS, digits=3, zero_division=0))
        pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
        rows += [(name, l, round(float(w), 3), round(float(x), 3), round(float(z), 3), int(n))
                 for l, w, x, z, n in zip(LABELS, pr, rc, f1, sup)]
        mp, mr, mf, _ = precision_recall_fscore_support(y, p, labels=LABELS, average="macro", zero_division=0)
        rows.append((name, "macro_avg", round(float(mp), 3), round(float(mr), 3), round(float(mf), 3), len(y)))
        acc = accuracy_score(y, p)
        rows.append((name, "accuracy", None, None, round(float(acc), 3), len(y)))
        cm = pd.DataFrame(confusion_matrix(y, p, labels=LABELS), index=LABELS, columns=LABELS)
        print(f"Confusion matrix (rows = true, columns = predicted):\n{cm}\n")
        cms.append(cm.rename_axis("true").reset_index().assign(method=name))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["method", "class", "precision", "recall", "f1", "support"]).to_csv(a.out, index=False)
    cm_path = a.out.with_name("sentiment_confusion.csv")
    pd.concat(cms)[["method", "true", *LABELS]].to_csv(cm_path, index=False)
    print(f"Saved {a.out} and {cm_path}")


def predict(a):
    clf = VaderSentimentClassifier() if a.method == "vader" else LogRegSentimentClassifier.load(a.model)
    print(clf.predict_one(a.text))


def main():
    ap = argparse.ArgumentParser(prog="python -m medreview.sentiment", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train", help="tune and fit the logistic regression model")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--C", type=float, nargs="+", default=[0.25, 1.0, 4.0, 16.0])
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=410)
    p.add_argument("--model", type=Path, default=MODEL)
    p.set_defaults(func=train)
    p = sub.add_parser("evaluate", help="compare VADER and logistic regression on the test split")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--model", type=Path, default=MODEL)
    p.add_argument("--out", type=Path, default=Path("reports/sentiment_eval.csv"))
    p.set_defaults(func=evaluate)
    p = sub.add_parser("predict", help="label one review")
    p.add_argument("text")
    p.add_argument("--method", choices=["logreg", "vader"], default="logreg")
    p.add_argument("--model", type=Path, default=MODEL)
    p.set_defaults(func=predict)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
