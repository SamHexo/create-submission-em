"""
Data preparation script for Google Brain - Ventilator Pressure Prediction.

Run this once to generate test.csv, private_test.csv, and sample_submission.csv
from the raw Kaggle train.csv.

Usage:
    python prepare_data.py --raw-train-csv /path/to/kaggle/train.csv

Downloads:
    Download raw train.csv from:
    https://www.kaggle.com/competitions/ventilator-pressure-prediction/data
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare(raw_train_csv: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    dtypes = {
        "id": "int32",
        "breath_id": "int32",
        "R": "int8",
        "C": "int8",
        "time_step": "float64",
        "u_in": "float64",
        "u_out": "int8",
        "pressure": "float64",
    }

    print(f"Reading {raw_train_csv}...")
    old_train = pd.read_csv(raw_train_csv, dtype=dtypes)

    # Split by breath_id to avoid leakage
    groups = [df.index.tolist() for _, df in old_train.groupby("breath_id")]
    train_idx, test_idx = train_test_split(groups, test_size=0.1, random_state=0)

    train_idx = [idx for sublist in train_idx for idx in sublist]
    test_idx = [idx for sublist in test_idx for idx in sublist]

    new_train = old_train.loc[train_idx].copy()
    new_test = old_train.loc[test_idx].copy()

    # Reset id columns
    new_train["id"] = range(1, len(new_train) + 1)
    new_test["id"] = range(1, len(new_test) + 1)

    assert set(new_train["breath_id"]).isdisjoint(set(new_test["breath_id"])), \
        "Test set contains breath_ids that are in the train set"

    # Public test (no labels)
    public_test = new_test.drop(columns=["pressure"])

    # Sample submission
    sample_submission = public_test[["id"]].copy()
    sample_submission["pressure"] = 0.0

    # Write files
    new_train.to_csv(output_dir / "train.csv", index=False, float_format="%.10g")
    public_test.to_csv(output_dir / "test.csv", index=False, float_format="%.10g")
    sample_submission.to_csv(output_dir / "sample_submission.csv", index=False, float_format="%.10g")
    new_test.to_csv(output_dir / "private_test.csv", index=False, float_format="%.10g")

    print(f"  train.csv:             {len(new_train):,} rows")
    print(f"  test.csv:              {len(public_test):,} rows (no labels)")
    print(f"  private_test.csv:      {len(new_test):,} rows (with labels)")
    print(f"  sample_submission.csv: {len(sample_submission):,} rows")
    print(f"\nDone. Files written to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ventilator-pressure-prediction data")
    parser.add_argument(
        "--raw-train-csv",
        type=Path,
        required=True,
        help="Path to the raw Kaggle train.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory to write output files (default: same folder as this script)",
    )
    args = parser.parse_args()

    if not args.raw_train_csv.exists():
        print(f"Error: {args.raw_train_csv} does not exist")
        raise SystemExit(1)

    prepare(args.raw_train_csv, args.output_dir)
