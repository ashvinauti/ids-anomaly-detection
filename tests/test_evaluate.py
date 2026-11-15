"""Tests for src/ids/evaluate.py, using small cases worked out by hand."""

import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ids.evaluate import (
    compute_metrics,
    confusion_counts,
    detection_rate_by_category,
    detection_rate_seen_vs_unseen,
    pick_best_model,
    plot_confusion_matrix,
    plot_detection_by_category,
    plot_error_rates,
    summarise_folds,
)

# 10 records: 4 normal, 6 attack.
# Predictions give TN=3, FP=1, FN=2, TP=4.
Y_TRUE = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
Y_PRED = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]


def test_confusion_counts_hand_worked_example():
    assert confusion_counts(Y_TRUE, Y_PRED) == {"tn": 3, "fp": 1, "fn": 2, "tp": 4}


def test_metrics_hand_worked_example():
    m = compute_metrics(Y_TRUE, Y_PRED)
    assert m["accuracy"] == pytest.approx(7 / 10)
    assert m["precision"] == pytest.approx(4 / 5)
    assert m["recall"] == pytest.approx(4 / 6)
    assert m["false_positive_rate"] == pytest.approx(1 / 4)
    assert m["false_negative_rate"] == pytest.approx(2 / 6)
    p, r = 4 / 5, 4 / 6
    assert m["f1"] == pytest.approx(2 * p * r / (p + r))


def test_metrics_agree_with_scikit_learn():
    m = compute_metrics(Y_TRUE, Y_PRED)
    assert m["accuracy"] == pytest.approx(accuracy_score(Y_TRUE, Y_PRED))
    assert m["precision"] == pytest.approx(precision_score(Y_TRUE, Y_PRED))
    assert m["recall"] == pytest.approx(recall_score(Y_TRUE, Y_PRED))
    assert m["f1"] == pytest.approx(f1_score(Y_TRUE, Y_PRED))


def test_false_negative_rate_is_one_minus_recall():
    m = compute_metrics(Y_TRUE, Y_PRED)
    assert m["false_negative_rate"] == pytest.approx(1 - m["recall"])


def test_perfect_detector():
    m = compute_metrics(Y_TRUE, Y_TRUE)
    assert m["accuracy"] == m["f1"] == 1.0
    assert m["false_positive_rate"] == m["false_negative_rate"] == 0.0


def test_detector_that_never_alerts_does_not_crash():
    """Precision is undefined (0/0); it should be reported as 0, not raise."""
    m = compute_metrics(Y_TRUE, [0] * len(Y_TRUE))
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["false_negative_rate"] == 1.0
    assert m["accuracy"] == pytest.approx(0.4)  # still "40% accurate" while useless


def test_summarise_folds_mean_and_std():
    folds = [compute_metrics(Y_TRUE, Y_PRED), compute_metrics(Y_TRUE, Y_TRUE)]
    summary = summarise_folds(folds)
    assert summary["accuracy"]["mean"] == pytest.approx((0.7 + 1.0) / 2)
    assert summary["accuracy"]["std"] == pytest.approx(0.15)
    assert "tp" not in summary


def test_pick_best_model_uses_f1_by_default():
    results = {"a": {"f1": 0.7, "accuracy": 0.9}, "b": {"f1": 0.8, "accuracy": 0.6}}
    assert pick_best_model(results) == "b"
    assert pick_best_model(results, metric="accuracy") == "a"


# --- Per-category breakdown and charts -------------------------------------

CATEGORIES = ["normal", "DoS", "DoS", "Probe", "R2L", "R2L", "R2L", "U2R"]
PREDICTED = [1, 1, 0, 1, 1, 0, 0, 0]


def test_detection_rate_by_category_hand_worked():
    result = detection_rate_by_category(CATEGORIES, PREDICTED)
    assert result["DoS"] == {"records": 2, "detected": 1, "detection_rate": 0.5}
    assert result["Probe"]["detection_rate"] == 1.0
    assert result["R2L"]["detection_rate"] == pytest.approx(1 / 3)
    assert result["U2R"]["detection_rate"] == 0.0
    assert "normal" not in result  # the false alarm on row 0 is not an attack detection


def test_detection_rate_for_missing_category_is_zero_not_error():
    result = detection_rate_by_category(["normal", "DoS"], [0, 1])
    assert result["U2R"] == {"records": 0, "detected": 0, "detection_rate": 0.0}


def test_seen_vs_unseen_split():
    names = ["normal", "neptune", "neptune", "mscan", "apache2"]
    preds = [1, 1, 1, 0, 1]
    result = detection_rate_seen_vs_unseen(names, preds, training_attack_names={"neptune"})
    assert result["seen_in_training"] == {"records": 2, "detected": 2, "detection_rate": 1.0}
    assert result["unseen_in_training"]["detection_rate"] == 0.5


def test_charts_are_written(tmp_path):
    m = compute_metrics(Y_TRUE, Y_PRED)
    plot_confusion_matrix(m, "svm", tmp_path / "cm.png")
    plot_error_rates({"svm": m, "kmeans": m}, tmp_path / "err.png")
    by_cat = {"svm": detection_rate_by_category(CATEGORIES, PREDICTED)}
    plot_detection_by_category(by_cat, tmp_path / "cat.png")
    for name in ("cm.png", "err.png", "cat.png"):
        assert (tmp_path / name).stat().st_size > 1000
