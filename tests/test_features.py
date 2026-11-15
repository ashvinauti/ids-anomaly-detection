"""Tests for src/ids/features.py: preprocessing output shape and behaviour."""

import numpy as np

from ids.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from ids.features import build_preprocessor, output_feature_names


def test_output_shape_is_one_hot_columns_plus_numeric(synthetic_features):
    X, _ = synthetic_features
    pre = build_preprocessor().fit(X)
    out = pre.transform(X)

    # Expected width: one column per distinct text value + the 38 numeric columns.
    n_one_hot = sum(X[c].nunique() for c in CATEGORICAL_COLUMNS)
    assert out.shape == (len(X), n_one_hot + len(NUMERIC_COLUMNS))
    assert len(output_feature_names(pre)) == out.shape[1]


def test_numeric_columns_are_standardised(synthetic_features):
    X, _ = synthetic_features
    pre = build_preprocessor().fit(X)
    out = pre.transform(X)
    numeric_part = out[:, -len(NUMERIC_COLUMNS):]
    assert np.allclose(numeric_part.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(numeric_part.std(axis=0), 1, atol=1e-9)


def test_one_hot_rows_have_exactly_one_value_per_text_column(synthetic_features):
    X, _ = synthetic_features
    pre = build_preprocessor().fit(X)
    one_hot_part = pre.transform(X)[:, : -len(NUMERIC_COLUMNS)]
    # Each row has one "1" for protocol, one for service, one for flag.
    assert (one_hot_part.sum(axis=1) == len(CATEGORICAL_COLUMNS)).all()


def test_unseen_service_does_not_crash(synthetic_features):
    X, _ = synthetic_features
    pre = build_preprocessor().fit(X)
    new_row = X.iloc[[0]].copy()
    new_row["service"] = "never_seen_before"
    out = pre.transform(new_row)
    assert out.shape[1] == pre.transform(X.iloc[[0]]).shape[1]


def test_label_columns_are_dropped(synthetic_features):
    X, y = synthetic_features
    X_with_label = X.assign(is_attack=y, attack_name="normal")
    pre = build_preprocessor().fit(X_with_label)
    names = output_feature_names(pre)
    assert not any("is_attack" in n or "attack_name" in n for n in names)
