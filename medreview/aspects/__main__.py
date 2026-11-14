"""Command line for the aspect work. Run from the MedReview folder.

    python -m medreview.aspects sample        # write sentences to hand-label
    python -m medreview.aspects evaluate      # keyword vs logistic regression
    python -m medreview.aspects side-effects  # per-drug side-effect table
    python -m medreview.aspects classify "It helped but made me dizzy."
"""
import argparse
import random
from pathlib import Path

import pandas as pd

from . import (KeywordAspectClassifier, LogRegAspectClassifier, LABELS, build_side_effect_table,
               classify_review, load_reviews, split_sentences)

LABELS_CSV = Path("data/labels/aspect_labels.csv")
MODEL = Path("data/aspects/aspect_logreg.joblib")
TABLE = Path("data/aspects/side_effects_by_drug.csv")
ALIASES = {"side effects": "side_effects", "side effect": "side_effects", "se": "side_effects", "s": "side_effects",
           "eff": "effectiveness", "e": "effectiveness", "n": "neither"}


def sample(a):
    """Sample test-split sentences for hand labeling.

    Half are stratified by the keyword classifier's guess so all classes appear;
    half are uniformly random so the keyword method's misses are represented.
    The guess is not written to the file, so labeling stays blind.
    """
    if a.out.exists() and not a.force:
        raise SystemExit(f"{a.out} exists; pass --force to overwrite (this discards any labels in it).")
    rng = random.Random(a.seed)
    df = load_reviews(a.data_dir, ("test",)).sample(n=a.pool_reviews, random_state=a.seed)
    pool = [(r.review_id, r.drug_name, r.condition, s)
            for r in df.itertuples() for s in split_sentences(r.review) if 20 <= len(s) <= 300]
    rng.shuffle(pool)
    kw = KeywordAspectClassifier()
    per_class = a.n // 2 // len(LABELS)
    chosen, seen, got = [], set(), dict.fromkeys(LABELS, 0)
    for row in pool:
        lab = kw.predict_one(row[3], row[2])
        if got[lab] < per_class and row[3] not in seen:
            got[lab] += 1
            seen.add(row[3])
            chosen.append(row)
    for row in pool:
        if len(chosen) >= a.n:
            break
        if row[3] not in seen:
            seen.add(row[3])
            chosen.append(row)
    rng.shuffle(chosen)
    out = pd.DataFrame(chosen, columns=["review_id", "drug_name", "condition", "sentence"])
    out.insert(0, "id", range(1, len(out) + 1))
    out["label"] = ""
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"Wrote {len(out)} sentences to {a.out}. Fill in the label column with: {', '.join(LABELS)}.")


def load_labels(path):
    df = pd.read_csv(path, keep_default_na=False)
    df["label"] = df["label"].astype(str).str.strip().str.lower().replace(ALIASES)
    bad = (~df["label"].isin(LABELS)).sum()
    if bad:
        print(f"Skipping {bad} rows with a missing or invalid label.")
    return df[df["label"].isin(LABELS)].reset_index(drop=True)


def evaluate(a):
    """Precision/recall of keyword matching vs logistic regression on the hand labels.

    keyword      keyword matching, no training
    logreg-cv    logistic regression, stratified 5-fold cross-validation on the hand labels
    logreg-weak  logistic regression trained on train-split sentences that the keyword
                 classifier labeled automatically, tested on the hand labels (test split)
    """
    from sklearn.metrics import classification_report, precision_recall_fscore_support
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from .logreg import make_pipeline

    gold = load_labels(a.labels)
    if gold.empty:
        raise SystemExit(f"No labeled rows in {a.labels}.")
    X, y, cond = gold["sentence"].tolist(), gold["label"].tolist(), gold["condition"].tolist()
    print(f"{len(X)} labeled sentences: {gold['label'].value_counts().to_dict()}\n")
    kw = KeywordAspectClassifier()
    preds = {"keyword": kw.predict(X, cond)}
    folds = min(5, gold["label"].value_counts().reindex(LABELS, fill_value=0).min())
    if folds >= 2:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=a.seed)
        preds["logreg-cv"] = list(cross_val_predict(make_pipeline(), X, y, cv=cv))
    else:
        print("Too few examples of some class for cross-validation; skipping logreg-cv.\n")
    if not a.no_weak:
        train = load_reviews(a.data_dir, ("train",)).sample(n=a.weak_reviews, random_state=a.seed)
        Xw, cw = [], []
        for r in train.itertuples():
            for s in split_sentences(r.review):
                Xw.append(s)
                cw.append(r.condition)
        model = LogRegAspectClassifier(min_df=2).fit(Xw, kw.predict(Xw, cw))
        print(f"logreg-weak trained on {len(Xw)} keyword-labeled sentences from {a.weak_reviews} training reviews\n")
        preds["logreg-weak"] = model.predict(X)
        a.model.parent.mkdir(parents=True, exist_ok=True)
        model.save(a.model)
    rows = []
    for name, p in preds.items():
        print(f"=== {name} ===")
        print(classification_report(y, p, labels=LABELS, digits=3, zero_division=0))
        pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
        rows += [(name, l, round(float(w), 3), round(float(x), 3), round(float(z), 3), int(n))
                 for l, w, x, z, n in zip(LABELS, pr, rc, f1, sup)]
        mp, mr, mf, _ = precision_recall_fscore_support(y, p, labels=LABELS, average="macro", zero_division=0)
        rows.append((name, "macro_avg", round(float(mp), 3), round(float(mr), 3), round(float(mf), 3), len(y)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["method", "class", "precision", "recall", "f1", "support"]).to_csv(a.out, index=False)
    gold.assign(**{"pred_" + k.replace("-", "_"): v for k, v in preds.items()}).to_csv(
        a.out.with_name("aspect_predictions.csv"), index=False)
    print(f"Saved {a.out} and {a.out.with_name('aspect_predictions.csv')}")


def side_effects(a):
    clf = LogRegAspectClassifier.load(a.model) if a.use_model else None
    table = build_side_effect_table(load_reviews(a.data_dir, a.splits), classifier=clf)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    table.save(a.out)
    print(f"Wrote {len(table.df)} rows for {table.df['drug'].nunique()} drugs to {a.out}")


def classify(a):
    clf = LogRegAspectClassifier.load(a.model) if a.use_model else None
    for sent, label in classify_review(a.text, clf):
        print(f"{label:14s} {sent}")


def main():
    ap = argparse.ArgumentParser(prog="python -m medreview.aspects", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sample", help="write sentences to hand-label")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--pool-reviews", type=int, default=3000)
    p.add_argument("--seed", type=int, default=410)
    p.add_argument("--out", type=Path, default=LABELS_CSV)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=sample)
    p = sub.add_parser("evaluate", help="compare keyword matching and logistic regression")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--labels", type=Path, default=LABELS_CSV)
    p.add_argument("--weak-reviews", type=int, default=20000)
    p.add_argument("--no-weak", action="store_true", help="skip the keyword-labeled training variant")
    p.add_argument("--seed", type=int, default=410)
    p.add_argument("--out", type=Path, default=Path("reports/aspect_eval.csv"))
    p.add_argument("--model", type=Path, default=MODEL)
    p.set_defaults(func=evaluate)
    p = sub.add_parser("side-effects", help="build the per-drug side-effect table")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--splits", nargs="+", choices=["train", "test"], default=["train", "test"])
    p.add_argument("--use-model", action="store_true", help="pick side-effect sentences with the saved logreg model")
    p.add_argument("--model", type=Path, default=MODEL)
    p.add_argument("--out", type=Path, default=TABLE)
    p.set_defaults(func=side_effects)
    p = sub.add_parser("classify", help="label the sentences of one review")
    p.add_argument("text")
    p.add_argument("--use-model", action="store_true")
    p.add_argument("--model", type=Path, default=MODEL)
    p.set_defaults(func=classify)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
