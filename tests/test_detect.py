"""Tests for detect.py: reading records, building alerts and the summary."""

import json
import sys
from pathlib import Path

import joblib
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import detect  # noqa: E402
from ids.data import FEATURE_COLUMNS  # noqa: E402
from ids.models import build_kmeans_pipeline  # noqa: E402


def _line(src_bytes: float, label: str | None) -> str:
    values = ["0"] * len(FEATURE_COLUMNS)
    values[1], values[2], values[3] = "tcp", "http", "SF"
    values[4] = str(src_bytes)
    if label is not None:
        values += [label, "21"]
    return ",".join(values)


@pytest.fixture
def model_file(tmp_path: Path, synthetic_features) -> Path:
    """A small K-Means pipeline fitted on synthetic data and saved like train.py does."""
    X, y = synthetic_features
    path = tmp_path / "model.joblib"
    joblib.dump({"model_name": "kmeans", "pipeline": build_kmeans_pipeline().fit(X, y)}, path)
    return path


def test_read_records_with_and_without_labels(tmp_path: Path):
    path = tmp_path / "records.txt"
    path.write_text(_line(10, "normal") + "\n" + _line(20, None) + "\n\n")
    records = list(detect.read_records(path))
    assert len(records) == 2  # the blank line is skipped
    (n1, row1, label1), (n2, _, label2) = records
    assert (n1, label1, n2, label2) == (1, "normal", 2, None)
    assert list(row1.columns) == FEATURE_COLUMNS
    assert row1["src_bytes"].iloc[0] == 10.0


def test_build_alert_has_the_required_fields(tmp_path: Path):
    path = tmp_path / "records.txt"
    path.write_text(_line(1234, None))
    number, row, _ = next(detect.read_records(path))
    alert = detect.build_alert(number, row, "svm")
    for field in ("record", "verdict", "protocol", "service", "flag", "src_bytes", "dst_bytes"):
        assert field in alert
    assert alert["src_bytes"] == 1234
    json.dumps(alert)  # must be valid JSON


def test_summary_counts_with_labels():
    s = detect.Summary()
    s.add(True, "neptune")   # true alert
    s.add(True, "normal")    # false alarm
    s.add(False, "satan")    # missed attack
    s.add(False, "normal")   # correctly quiet
    d = s.as_dict("svm", 1.0)
    assert d["records_processed"] == 4 and d["alerts_raised"] == 2
    assert d["using_labels_in_file"] == {"true_alerts": 1, "false_alarms": 1,
                                         "missed_attacks": 1}


def test_summary_without_labels_has_no_label_section():
    s = detect.Summary()
    s.add(True, None)
    assert "using_labels_in_file" not in s.as_dict("svm", 1.0)


def test_run_end_to_end(tmp_path: Path, model_file: Path, capsys):
    path = tmp_path / "records.txt"
    # Two ordinary records and one with a huge byte count that should be flagged.
    path.write_text("\n".join([_line(0, "normal"), _line(0, "normal"),
                               _line(1e9, "neptune")]))
    summary = detect.run(path, model_file, limit=3)
    assert summary["records_processed"] == 3
    alerts = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
    assert any(a["record"] == 3 for a in alerts)


def test_limit_stops_early(tmp_path: Path, model_file: Path):
    path = tmp_path / "records.txt"
    path.write_text("\n".join(_line(0, "normal") for _ in range(10)))
    assert detect.run(path, model_file, limit=4, quiet=True)["records_processed"] == 4
