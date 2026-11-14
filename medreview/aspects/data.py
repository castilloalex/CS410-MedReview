# AI-assisted code
# Model: Fable 5.2 (medium effort)
"""Loading the cleaned reviews produced by `python -m medreview prepare`."""
import json
from pathlib import Path

import pandas as pd

COLUMNS = ["review_id", "split", "drug_name", "drug_key", "condition", "review", "rating"]


def load_reviews(data_dir="data", splits=("train", "test")) -> pd.DataFrame:
    """Read data/processed/<split>.jsonl into one DataFrame."""
    rows = []
    for split in splits:
        with open(Path(data_dir) / "processed" / f"{split}.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows.append([r.get(c) for c in COLUMNS])
    return pd.DataFrame(rows, columns=COLUMNS)
