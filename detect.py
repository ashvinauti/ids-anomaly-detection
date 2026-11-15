"""Score network records with the saved model and print analyst-facing alerts.

This SIMULATES a live feed: records are read one at a time from a file,
optionally with a pause between them, and each is scored as it arrives.
It is not connected to a real network or a production IDS.

Usage:
    python detect.py                                  # all of KDDTest+, no delay
    python detect.py --limit 200 --delay 0.05         # first 200 records, 50 ms apart
    python detect.py --input my_records.txt --quiet   # summary only

Input: NSL-KDD format, comma-separated, no header. Lines may have just the
41 features, or 41 features + label + difficulty (as in KDDTest+.txt).
When labels are present, the summary also reports how many alerts were
correct, how many were false alarms, and how many attacks were missed.

Output: one JSON alert per line on stdout for each record flagged as an
anomaly, then a summary.
"""

import argparse
import json
import sys
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ids.data import FEATURE_COLUMNS, NUMERIC_COLUMNS, TEST_FILE  # noqa: E402

DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "best_model.joblib"


def read_records(path: Path) -> Iterator[tuple[int, pd.DataFrame, str | None]]:
    """Yield (record number, one-row feature DataFrame, label or None), one line at a time.

    Reading line by line, rather than loading the whole file, mirrors how a
    live sensor would hand records over as they happen.
    """
    with path.open() as handle:
        for record_number, line in enumerate(handle, start=1):
            values = line.strip().split(",")
            if len(values) < len(FEATURE_COLUMNS):
                continue  # skip blank or malformed lines
            label = values[len(FEATURE_COLUMNS)] if len(values) > len(FEATURE_COLUMNS) else None
            row = pd.DataFrame([values[: len(FEATURE_COLUMNS)]], columns=FEATURE_COLUMNS)
            row[NUMERIC_COLUMNS] = row[NUMERIC_COLUMNS].astype(float)
            yield record_number, row, label


def build_alert(record_number: int, row: pd.DataFrame, model_name: str) -> dict:
    """The fields an analyst needs to start triage on one flagged record."""
    r = row.iloc[0]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "record": record_number,
        "verdict": "anomaly",
        "model": model_name,
        "protocol": r["protocol_type"],
        "service": r["service"],
        "flag": r["flag"],
        "src_bytes": int(r["src_bytes"]),
        "dst_bytes": int(r["dst_bytes"]),
    }


class Summary:
    """Running totals for the end-of-run report."""

    def __init__(self) -> None:
        self.records = 0
        self.alerts = 0
        self.labelled = 0
        self.true_alerts = 0     # alert on a real attack
        self.false_alarms = 0    # alert on normal traffic
        self.missed_attacks = 0  # attack with no alert

    def add(self, is_alert: bool, label: str | None) -> None:
        """Update the totals with one scored record."""
        self.records += 1
        self.alerts += int(is_alert)
        if label is None:
            return
        self.labelled += 1
        is_attack = label != "normal"
        self.true_alerts += int(is_alert and is_attack)
        self.false_alarms += int(is_alert and not is_attack)
        self.missed_attacks += int(not is_alert and is_attack)

    def as_dict(self, model_name: str, seconds: float) -> dict:
        """The summary as a dictionary, ready to print as JSON."""
        result = {
            "model": model_name,
            "records_processed": self.records,
            "alerts_raised": self.alerts,
            "alert_rate": round(self.alerts / self.records, 4) if self.records else 0.0,
            "elapsed_seconds": round(seconds, 2),
        }
        if self.labelled:
            result["using_labels_in_file"] = {
                "true_alerts": self.true_alerts,
                "false_alarms": self.false_alarms,
                "missed_attacks": self.missed_attacks,
            }
        return result


def run(input_path: Path, model_path: Path, delay: float = 0.0,
        limit: int | None = None, quiet: bool = False) -> dict:
    """Score records from input_path, print alerts, and return the summary."""
    bundle = joblib.load(model_path)
    pipeline, model_name = bundle["pipeline"], bundle["model_name"]

    summary = Summary()
    start = time.perf_counter()
    for record_number, row, label in read_records(input_path):
        if limit is not None and summary.records >= limit:
            break
        is_alert = int(pipeline.predict(row)[0]) == 1
        summary.add(is_alert, label)
        if is_alert and not quiet:
            print(json.dumps(build_alert(record_number, row, model_name)), flush=True)
        if delay:
            time.sleep(delay)  # pretend records arrive over time
    return summary.as_dict(model_name, time.perf_counter() - start)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=TEST_FILE, help="records file (NSL-KDD format)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="saved model bundle")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds to wait between records")
    parser.add_argument("--limit", type=int, default=None, help="stop after this many records")
    parser.add_argument("--quiet", action="store_true", help="print the summary only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        sys.exit(f"Model not found: {args.model}. Run train.py first.")
    summary = run(args.input, args.model, args.delay, args.limit, args.quiet)
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
