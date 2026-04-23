#!/usr/bin/env python3
"""Remove the Is_NASA/IS_NASA column from the two standard scaled CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_FILES = ["HNEI_STANDARD_SCALED.csv", "NASA_STANDARD_SCALED.csv"]
COLUMN_CANDIDATES = ("Is_NASA", "IS_NASA")


def find_is_nasa_column(df: pd.DataFrame) -> str:
    """Return the matching Is_NASA column name, supporting common capitalizations."""
    for col in COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    raise KeyError("Column 'Is_NASA' or 'IS_NASA' was not found")


def remove_column(input_csv: Path, output_csv: Path) -> tuple[str, int, int]:
    """Remove Is_NASA column and write cleaned CSV."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    column_name = find_is_nasa_column(df)

    row_count = len(df)
    original_cols = len(df.columns)

    cleaned = df.drop(columns=[column_name])
    cleaned.to_csv(output_csv, index=False)

    return column_name, row_count, original_cols - len(cleaned.columns)


def build_output_path(input_csv: Path, in_place: bool) -> Path:
    """Choose output path based on in-place flag."""
    if in_place:
        return input_csv
    return input_csv.with_name(f"{input_csv.stem}_NO_IS_NASA{input_csv.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove Is_NASA/IS_NASA from HNEI and NASA standard scaled CSV files"
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=DEFAULT_FILES,
        help=(
            "CSV files to process "
            "(default: HNEI_STANDARD_SCALED.csv NASA_STANDARD_SCALED.csv)"
        ),
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input files instead of creating *_NO_IS_NASA.csv outputs",
    )

    args = parser.parse_args()

    for file_name in args.files:
        input_csv = Path(file_name).expanduser().resolve()
        output_csv = build_output_path(input_csv, args.in_place)

        column_name, rows, dropped = remove_column(input_csv, output_csv)
        print(
            f"Processed {input_csv.name}: removed '{column_name}' "
            f"({dropped} column) across {rows} rows -> {output_csv.name}"
        )


if __name__ == "__main__":
    main()
