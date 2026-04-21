#!/usr/bin/env python3
"""Extract demo battery CSVs from combined_scaled_battery_data.csv.

This script finds battery boundaries by detecting when Cycle_Index resets
(non-increasing compared to previous row), then exports:
- last Is_NASA=true battery -> NASA_STANDARD_SCALED.csv
- last Is_NASA=false battery -> HNEI_STANDARD_SCALED.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


TRUE_VALUES = {"1", "1.0", "true", "t", "yes", "y"}
FALSE_VALUES = {"0", "0.0", "false", "f", "no", "n"}


def normalize_flag(value: str) -> bool:
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    raise ValueError(f"Unsupported Is_NASA value: {value!r}")


def find_battery_segments(rows: List[Dict[str, str]]) -> List[Tuple[int, int]]:
    if not rows:
        return []

    starts = [0]
    prev_cycle = float(rows[0]["Cycle_Index"])

    for idx in range(1, len(rows)):
        cycle = float(rows[idx]["Cycle_Index"])
        if cycle <= prev_cycle:
            starts.append(idx)
        prev_cycle = cycle

    starts.append(len(rows))
    return [(starts[i], starts[i + 1]) for i in range(len(starts) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract standard-scaled demo battery CSVs")
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="combined_scaled_battery_data.csv",
        help="Input combined CSV path (default: combined_scaled_battery_data.csv)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("Input CSV has no headers")
        if "Is_NASA" not in fieldnames:
            raise KeyError("Column 'Is_NASA' was not found in the input CSV")
        if "Cycle_Index" not in fieldnames:
            raise KeyError("Column 'Cycle_Index' was not found in the input CSV")
        rows = list(reader)

    segments = find_battery_segments(rows)
    if not segments:
        raise ValueError("Input CSV contains no data rows")

    last_true_segment = None
    last_false_segment = None

    for start, end in reversed(segments):
        is_nasa = normalize_flag(rows[start]["Is_NASA"])
        if is_nasa and last_true_segment is None:
            last_true_segment = (start, end)
        if (not is_nasa) and last_false_segment is None:
            last_false_segment = (start, end)
        if last_true_segment and last_false_segment:
            break

    if last_true_segment is None:
        raise ValueError("No Is_NASA=true battery segment found")
    if last_false_segment is None:
        raise ValueError("No Is_NASA=false battery segment found")

    nasa_output = input_path.parent / "NASA_STANDARD_SCALED.csv"
    hnei_output = input_path.parent / "HNEI_STANDARD_SCALED.csv"

    t_start, t_end = last_true_segment
    f_start, f_end = last_false_segment

    with nasa_output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows[t_start:t_end])

    with hnei_output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows[f_start:f_end])

    print(f"Wrote {t_end - t_start} rows to {nasa_output.name}")
    print(f"Wrote {f_end - f_start} rows to {hnei_output.name}")


if __name__ == "__main__":
    main()
