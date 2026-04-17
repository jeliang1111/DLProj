#!/usr/bin/env python3
"""Split a CSV into Is_NASA true/false datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TRUE_VALUES = {"1", "true", "t", "yes", "y"}
FALSE_VALUES = {"0", "false", "f", "no", "n"}


def normalize_is_nasa(value: object) -> bool:
    """Normalize Is_NASA values to a strict boolean."""
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    raise ValueError(f"Unsupported Is_NASA value: {value!r}")


def split_dataset(input_csv: Path) -> tuple[Path, Path, int, int]:
    """Split input CSV into Is_NASA_true.csv and Is_NASA_false.csv in same folder."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if "Is_NASA" not in df.columns:
        raise KeyError("Column 'Is_NASA' was not found in the input CSV")

    is_nasa_bool = df["Is_NASA"].map(normalize_is_nasa)

    nasa_true_df = df[is_nasa_bool]
    nasa_false_df = df[~is_nasa_bool]

    output_true = input_csv.parent / "Is_NASA_true.csv"
    output_false = input_csv.parent / "Is_NASA_false.csv"

    nasa_true_df.to_csv(output_true, index=False)
    nasa_false_df.to_csv(output_false, index=False)

    return output_true, output_false, len(nasa_true_df), len(nasa_false_df)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a CSV into Is_NASA_true.csv and Is_NASA_false.csv"
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="combined_scaled_battery_data.csv",
        help="Path to the input CSV (default: combined_scaled_battery_data.csv)",
    )

    args = parser.parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()

    true_path, false_path, true_rows, false_rows = split_dataset(input_csv)

    print(f"Wrote {true_rows} rows -> {true_path}")
    print(f"Wrote {false_rows} rows -> {false_path}")


if __name__ == "__main__":
    main()
