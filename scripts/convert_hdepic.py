"""One-off conversion of HD_EPIC_Narrations.pkl to JSON Lines.

The pickle needs numpy>=2 (the annotations were pickled with it), so run it isolated:

    uv run --isolated --no-project --with pandas --with "numpy>=2" \
        python scripts/convert_hdepic.py <in.pkl> <out.jsonl>
"""
import argparse

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HD_EPIC_Narrations.pkl to JSON Lines.")
    parser.add_argument("pkl", help="path to HD_EPIC_Narrations.pkl")
    parser.add_argument("out", help="output .jsonl path")
    args = parser.parse_args()
    df = pd.read_pickle(args.pkl)
    df.to_json(args.out, orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    main()
