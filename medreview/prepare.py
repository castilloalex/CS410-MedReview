"""Fetch the original UCI archive and produce audited JSONL splits."""
import csv
import hashlib
import io
import json
import os
import re
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

from .text import clean_text, duplicate_key, tokenize, TOKENIZER_VERSION

SOURCE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip"
MEMBERS = {"test": "drugsComTest_raw.tsv", "train": "drugsComTrain_raw.tsv"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(".download")
        urllib.request.urlretrieve(SOURCE_URL, temporary)
        with zipfile.ZipFile(temporary) as archive:
            if not set(MEMBERS.values()).issubset(archive.namelist()):
                raise ValueError("Archive is missing the expected UCI TSV files")
        os.replace(temporary, path)
    return path


def integer(value, minimum, maximum=None):
    number = float(value)
    if not number.is_integer() or number < minimum or (maximum is not None and number > maximum):
        raise ValueError("Invalid integer range")
    return int(number)


def prepare(archive_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    split_counts = {}
    seen_ids = set()
    seen_pairs = set()
    test_texts = set()
    drugs = set()
    conditions = set()
    # Test first: keep the held-out observations and remove matching train text.
    with zipfile.ZipFile(archive_path) as archive, (output_dir / "excluded.csv").open("w", encoding="utf-8", newline="") as audit:
        writer = csv.writer(audit)
        writer.writerow(["split", "source_row", "review_id", "reason"])
        for split, member in MEMBERS.items():
            local = Counter()
            with io.TextIOWrapper(archive.open(member), encoding="utf-8-sig", newline="") as source, (output_dir / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as target:
                reader = csv.DictReader(source, delimiter="\t")
                expected = {"", "drugName", "condition", "review", "rating", "date", "usefulCount"}
                if set(reader.fieldnames or []) != expected:
                    raise ValueError(f"Unexpected TSV schema in {member}: {reader.fieldnames}")
                for row_number, raw in enumerate(reader, 2):
                    local["input"] += 1
                    reason = None
                    try:
                        review_id = integer(raw[""], 0)
                        rating = integer(raw["rating"], 1, 10)
                        useful = integer(raw["usefulCount"], 0)
                    except (ValueError, TypeError, OverflowError):
                        reason = "invalid_numeric_field"
                    drug = clean_text(raw["drugName"])
                    review = clean_text(raw["review"])
                    if not drug or not tokenize(review):
                        reason = "missing_drug_or_review"
                    key = duplicate_key(review)
                    pair = (drug.casefold(), key)
                    if not reason and split == "train" and key in test_texts:
                        reason = "train_test_text_overlap"
                    if not reason and pair in seen_pairs:
                        reason = "duplicate_drug_review"
                    if reason:
                        writer.writerow([split, row_number, raw[""], reason])
                        local["excluded_" + reason] += 1
                        continue
                    if review_id in seen_ids:
                        raise ValueError(f"Conflicting UCI review ID: {review_id}")
                    condition = clean_text(raw["condition"])
                    if not condition or re.search(r"\bfound this comment useful\b", condition, re.I):
                        local["missing_or_malformed_condition"] += 1
                        condition = None
                    try:
                        date = datetime.strptime(raw["date"].strip(), "%B %d, %Y").date().isoformat()
                    except ValueError:
                        date = None
                        local["invalid_date"] += 1
                    record = {"review_id": review_id, "split": split, "drug_name": drug,
                              "drug_key": drug.casefold(), "condition": condition,
                              "review": review, "rating": rating, "date": date,
                              "useful_count": useful, "source_file": member, "source_row": row_number}
                    target.write(json.dumps(record, ensure_ascii=False) + "\n")
                    seen_ids.add(review_id)
                    seen_pairs.add(pair)
                    if split == "test":
                        test_texts.add(key)
                    drugs.add(drug.casefold())
                    if condition:
                        conditions.add(condition.casefold())
                    local["retained"] += 1
            split_counts[split] = dict(local)
            counts.update(local)
    report = {"schema_version": 1, "source_url": SOURCE_URL,
              "source_sha256": sha256(archive_path), "tokenizer": TOKENIZER_VERSION,
              "counts": dict(counts), "splits": split_counts,
              "unique_drugs": len(drugs), "unique_conditions": len(conditions),
              "deduplication": "Keep first (drug_key, casefolded cleaned review); test first. Remove train reviews whose normalized full text occurs in test, across all drugs.",
              "outputs": {f"{s}.jsonl": sha256(output_dir / f"{s}.jsonl") for s in MEMBERS}}
    (output_dir / "cleaning_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
