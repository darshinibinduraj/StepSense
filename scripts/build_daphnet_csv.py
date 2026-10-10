"""Combine raw Daphnet txt files into data/daphnet.csv.

Expected raw files are plain-text rows of numeric values (for example, S01R01.txt).
The script keeps raw files untouched and writes a new CSV with:
subject,time,ankle_hori_fwd_accel,ankle_vertical,ankle_lateral,label
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

OUTPUT_COLUMNS = [
    "subject",
    "time",
    "ankle_hori_fwd_accel",
    "ankle_vertical",
    "ankle_lateral",
    "label",
]

# Canonical Daphnet raw row has at least 11 columns:
# 0 time, 1-3 ankle axes, 10 label.
TIME_INDEX = 0
ANKLE_INDICES = (1, 2, 3)
LABEL_INDEX = 10


@dataclass
class ParseSummary:
    files_seen: int = 0
    files_with_rows: int = 0
    rows_written: int = 0
    malformed_rows: int = 0
    empty_lines: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine raw Daphnet files in data/ into data/daphnet.csv while preserving raw files."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing raw txt files (default: data).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/daphnet.csv"),
        help="Output CSV path (default: data/daphnet.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = ParseSummary()
    raw_files = find_raw_files(args.input_dir)

    if not raw_files:
        print(f"No raw Daphnet txt files found in: {args.input_dir.resolve()}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(OUTPUT_COLUMNS)
        for raw_file in raw_files:
            process_file(raw_file, writer, summary)

    print("Built daphnet CSV:")
    print(f"- output: {args.output.resolve()}")
    print(f"- files seen: {summary.files_seen}")
    print(f"- files with written rows: {summary.files_with_rows}")
    print(f"- rows written: {summary.rows_written}")
    print(f"- malformed rows skipped: {summary.malformed_rows}")
    print(f"- empty lines skipped: {summary.empty_lines}")


def find_raw_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    return sorted(path for path in input_dir.glob("*.txt") if path.is_file())


def process_file(raw_file: Path, writer: csv.writer, summary: ParseSummary) -> None:
    summary.files_seen += 1
    subject = subject_from_filename(raw_file)
    file_rows_written = 0

    with raw_file.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                summary.empty_lines += 1
                continue

            fields = split_fields(line)
            row = select_output_fields(fields, subject=subject)
            if row is None:
                summary.malformed_rows += 1
                # Keep this concise; aggregate counts are printed at end.
                if summary.malformed_rows <= 5:
                    print(f"Skipped malformed row: {raw_file.name}:{line_number}")
                continue

            writer.writerow(row)
            summary.rows_written += 1
            file_rows_written += 1

    if file_rows_written > 0:
        summary.files_with_rows += 1


def split_fields(line: str) -> list[str]:
    # Accept whitespace-, comma-, or semicolon-separated rows.
    return [token for token in re.split(r"[\s,;]+", line) if token]


def select_output_fields(fields: list[str], *, subject: str) -> list[str] | None:
    if len(fields) <= LABEL_INDEX:
        return None

    time_value = fields[TIME_INDEX]
    ankle_1, ankle_2, ankle_3 = (fields[index] for index in ANKLE_INDICES)
    label_value = fields[LABEL_INDEX]
    return [subject, time_value, ankle_1, ankle_2, ankle_3, label_value]


def subject_from_filename(path: Path) -> str:
    stem = path.stem
    match = re.match(r"^(S\d+)", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return stem


if __name__ == "__main__":
    main()

