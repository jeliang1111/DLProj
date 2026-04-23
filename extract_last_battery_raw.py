#!/usr/bin/env python3
"""Extract the last battery segment from HNEI and NASA raw cycle files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BATTERY_COLUMN_CANDIDATES = (
    "battery_id",
    "Battery_ID",
    "battery",
    "Battery",
    "cell_id",
    "Cell_ID",
)


def resolve_existing_path(preferred: Path, fallback: Path | None = None) -> Path:
    """Return the first existing path among preferred and optional fallback."""
    if preferred.exists():
        return preferred
    if fallback is not None and fallback.exists():
        return fallback
    if fallback is None:
        raise FileNotFoundError(f"Input file not found: {preferred}")
    raise FileNotFoundError(
        f"Input file not found. Tried: {preferred} and {fallback}"
    )


def extract_last_battery(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Extract last battery rows using battery id columns or cycle reset logic."""
    for col in BATTERY_COLUMN_CANDIDATES:
        if col in df.columns:
            last_value = df[col].iloc[-1]
            return df[df[col] == last_value].copy(), f"column '{col}' == {last_value!r}"

    if "Cycle_Index" not in df.columns:
        raise KeyError(
            "Could not identify battery boundaries. Expected one of battery id columns "
            f"{BATTERY_COLUMN_CANDIDATES} or 'Cycle_Index'."
        )

    cycle = pd.to_numeric(df["Cycle_Index"], errors="coerce")
    reset_points = cycle.diff().fillna(0) < 0

    if not reset_points.any():
        return df.copy(), "no reset in Cycle_Index (single battery file)"

    start_idx = reset_points[reset_points].index[-1]
    return df.loc[start_idx:].copy(), f"Cycle_Index reset at row index {start_idx}"


def process_file(input_path: Path, output_path: Path) -> None:
    """Load one CSV, extract last battery, and save to output CSV."""
    df = pd.read_csv(input_path)
    extracted, method = extract_last_battery(df)
    extracted.to_csv(output_path, index=False)
    print(
        f"{input_path.name} -> {output_path.name}: "
        f"wrote {len(extracted)} rows ({method})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract last battery segment from Battery_RUL.csv and "
            "nasa/nassa_battery_cycles.csv"
        )
    )
    parser.add_argument(
        "--hnei-input",
        default="Battery_RUL.csv",
        help="Input path for HNEI dataset (default: Battery_RUL.csv)",
    )
    parser.add_argument(
        "--hnei-output",
        default="HNEI_DATA_AUG_RAW.csv",
        help="Output path for extracted HNEI battery (default: HNEI_DATA_AUG_RAW.csv)",
    )
    parser.add_argument(
        "--nasa-input",
        default="nassa_battery_cycles.csv",
        help=(
            "Input path for NASA dataset (default: nassa_battery_cycles.csv; "
            "falls back to nasa_battery_cycles.csv if needed)"
        ),
    )
    parser.add_argument(
        "--nasa-output",
        default="NASA_DATA_AUG_RAW.csv",
        help="Output path for extracted NASA battery (default: NASA_DATA_AUG_RAW.csv)",
    )

    args = parser.parse_args()

    hnei_input = resolve_existing_path(Path(args.hnei_input).expanduser().resolve())
    nasa_input = resolve_existing_path(
        Path(args.nasa_input).expanduser().resolve(),
        Path("nasa_battery_cycles.csv").expanduser().resolve(),
    )

    hnei_output = Path(args.hnei_output).expanduser().resolve()
    nasa_output = Path(args.nasa_output).expanduser().resolve()

    process_file(hnei_input, hnei_output)
    process_file(nasa_input, nasa_output)


if __name__ == "__main__":
    main()
