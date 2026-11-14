"""Tests for src/ids/data.py: column layout, attack mapping and file loading."""

from pathlib import Path

import pytest

from ids.data import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    RAW_COLUMNS,
    TEST_FILE,
    TRAIN_FILE,
    attack_category,
    load_nsl_kdd,
    split_features_target,
)


def _fake_row(protocol: str, service: str, flag: str, label: str) -> str:
    """Build one line in NSL-KDD format: 41 features, label, difficulty.

    Numeric features are all set to 0; only the text fields and label matter here.
    """
    values = ["0"] * len(FEATURE_COLUMNS)
    values[1], values[2], values[3] = protocol, service, flag
    return ",".join(values + [label, "21"])


@pytest.fixture
def tiny_file(tmp_path: Path) -> Path:
    """A three-row file in the same format as KDDTrain+.txt (no header)."""
    lines = [
        _fake_row("tcp", "http", "SF", "normal"),
        _fake_row("tcp", "private", "S0", "neptune"),
        _fake_row("icmp", "eco_i", "SF", "ipsweep"),
    ]
    path = tmp_path / "tiny.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


# --- Column layout ---------------------------------------------------------

def test_there_are_41_features_and_43_raw_columns():
    assert len(FEATURE_COLUMNS) == 41
    assert len(RAW_COLUMNS) == 43
    assert len(set(FEATURE_COLUMNS)) == 41  # no duplicate names


def test_categorical_and_numeric_columns_cover_all_features():
    assert set(CATEGORICAL_COLUMNS) | set(NUMERIC_COLUMNS) == set(FEATURE_COLUMNS)
    assert len(NUMERIC_COLUMNS) == 38


# --- Attack-category mapping ----------------------------------------------

@pytest.mark.parametrize(
    "name, expected",
    [
        ("normal", "normal"),
        ("neptune", "DoS"),
        ("smurf", "DoS"),
        ("portsweep", "Probe"),
        ("satan", "Probe"),
        ("guess_passwd", "R2L"),
        ("warezclient", "R2L"),
        ("buffer_overflow", "U2R"),
        ("rootkit", "U2R"),
        ("mscan", "Probe"),       # only appears in the test set
        ("apache2", "DoS"),       # only appears in the test set
    ],
)
def test_attack_category_known_names(name: str, expected: str):
    assert attack_category(name) == expected


def test_attack_category_is_case_and_space_tolerant():
    assert attack_category("  Neptune ") == "DoS"


def test_attack_category_unknown_name_does_not_crash():
    assert attack_category("brand_new_attack") == "unknown"


# --- Loading ---------------------------------------------------------------

def test_load_adds_target_and_category(tiny_file: Path):
    df = load_nsl_kdd(tiny_file)
    assert len(df) == 3
    assert df["is_attack"].tolist() == [0, 1, 1]
    assert df["attack_category"].tolist() == ["normal", "DoS", "Probe"]


def test_split_features_target_excludes_label_columns(tiny_file: Path):
    X, y = split_features_target(load_nsl_kdd(tiny_file))
    assert list(X.columns) == FEATURE_COLUMNS
    for leaky in ("attack_name", "attack_category", "difficulty", "is_attack"):
        assert leaky not in X.columns
    assert y.tolist() == [0, 1, 1]


def test_missing_file_gives_helpful_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Download NSL-KDD"):
        load_nsl_kdd(tmp_path / "nope.txt")


# --- Real dataset (skipped if not downloaded) -----------------------------

@pytest.mark.skipif(not TRAIN_FILE.exists(), reason="KDDTrain+.txt not downloaded")
def test_real_train_file_loads_with_no_unknown_attacks():
    df = load_nsl_kdd(TRAIN_FILE)
    assert df.shape[1] == 45  # 43 raw columns + 2 added
    assert (df["attack_category"] != "unknown").all()


@pytest.mark.skipif(not TEST_FILE.exists(), reason="KDDTest+.txt not downloaded")
def test_real_test_file_loads_with_no_unknown_attacks():
    df = load_nsl_kdd(TEST_FILE)
    assert df.shape[1] == 45
    assert (df["attack_category"] != "unknown").all()
