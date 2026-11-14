import argparse
import json
from pathlib import Path

from .index import SearchIndex, build_index
from .prepare import download, prepare


def main():
    parser = argparse.ArgumentParser(description="MedReview cleaning and search-index pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "build", "all"):
        p = sub.add_parser(command)
        p.add_argument("--data-dir", type=Path, default=Path("data"))
        if command in ("build", "all"):
            p.add_argument("--splits", nargs="+", choices=["train", "test"], default=["train", "test"])
            p.add_argument("--index", type=Path, default=None)
    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("--drug")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--index", type=Path, default=Path("data/index/reviews.sqlite"))
    args = parser.parse_args()
    if args.command in ("prepare", "all"):
        archive = download(args.data_dir / "raw/uci_drug_reviews.zip")
        print(json.dumps(prepare(archive, args.data_dir / "processed"), indent=2))
    if args.command in ("build", "all"):
        print(json.dumps(build_index(args.data_dir / "processed", args.index or args.data_dir / "index/reviews.sqlite", args.splits), indent=2))
    if args.command == "search":
        with SearchIndex(args.index) as index:
            print(json.dumps(index.search(args.query, drug=args.drug, limit=args.limit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
