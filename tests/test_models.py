"""Tests for src/ids/models.py: KMeansDetector behaviour and pipeline wiring."""

import numpy as np
import pytest

from ids.models import (
    KMeansDetector,
    build_all_models,
    build_kmeans_pipeline,
    stratified_sample,
)


@pytest.fixture
def normal_points() -> np.ndarray:
    """1,000 2-D points clustered around the origin, standing in for normal traffic."""
    return np.random.default_rng(1).normal(0, 1, size=(1000, 2))


# --- KMeansDetector --------------------------------------------------------

def test_fit_sets_threshold_at_requested_percentile(normal_points):
    det = KMeansDetector(n_clusters=3, percentile=95).fit(normal_points)
    train_scores = det.anomaly_score(normal_points)
    assert det.threshold_ == pytest.approx(np.percentile(train_scores, 95))


def test_about_five_percent_of_training_normals_are_flagged(normal_points):
    det = KMeansDetector(n_clusters=3, percentile=95).fit(normal_points)
    flagged_rate = det.predict(normal_points).mean()
    assert flagged_rate == pytest.approx(0.05, abs=0.01)


def test_far_away_point_is_flagged_and_centre_is_not(normal_points):
    det = KMeansDetector(n_clusters=3).fit(normal_points)
    assert det.predict(np.array([[50.0, 50.0]]))[0] == 1
    assert det.predict(np.array([[0.0, 0.0]]))[0] == 0


def test_predict_returns_only_zero_and_one(normal_points):
    det = KMeansDetector(n_clusters=3).fit(normal_points)
    assert set(np.unique(det.predict(normal_points))) <= {0, 1}


def test_attack_rows_are_ignored_during_fit(normal_points):
    """Adding labelled attacks to the training data must not change the model."""
    attacks = np.full((200, 2), 40.0)
    X = np.vstack([normal_points, attacks])
    y = np.array([0] * len(normal_points) + [1] * len(attacks))

    with_attacks = KMeansDetector(n_clusters=3).fit(X, y)
    normal_only = KMeansDetector(n_clusters=3).fit(normal_points)

    assert with_attacks.threshold_ == pytest.approx(normal_only.threshold_)
    assert with_attacks.predict(attacks).all()  # attacks are still far from normal


def test_higher_percentile_flags_fewer_records(normal_points):
    strict = KMeansDetector(n_clusters=3, percentile=99).fit(normal_points)
    loose = KMeansDetector(n_clusters=3, percentile=90).fit(normal_points)
    assert strict.predict(normal_points).sum() < loose.predict(normal_points).sum()


def test_same_seed_gives_same_result(normal_points):
    a = KMeansDetector(n_clusters=3, random_state=7).fit(normal_points)
    b = KMeansDetector(n_clusters=3, random_state=7).fit(normal_points)
    assert a.threshold_ == b.threshold_


def test_too_few_normal_rows_raises():
    with pytest.raises(ValueError):
        KMeansDetector(n_clusters=5).fit(np.zeros((3, 2)))


# --- Pipelines and sampling -----------------------------------------------

def test_all_three_pipelines_fit_and_predict(synthetic_features):
    X, y = synthetic_features
    for name, pipeline in build_all_models().items():
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        assert preds.shape == (len(X),), name
        assert set(np.unique(preds)) <= {0, 1}, name


def test_kmeans_pipeline_separates_synthetic_attacks(synthetic_features):
    X, y = synthetic_features
    preds = build_kmeans_pipeline().fit(X, y).predict(X)
    assert preds[y == 1].mean() == 1.0  # every synthetic attack is flagged


def test_stratified_sample_keeps_class_ratio(synthetic_features):
    X, y = synthetic_features
    X_s, y_s = stratified_sample(X, y, n_rows=100)
    assert len(X_s) == 100
    assert y_s.mean() == pytest.approx(y.mean(), abs=0.01)


def test_stratified_sample_returns_everything_if_asked_for_too_much(synthetic_features):
    X, y = synthetic_features
    X_s, _ = stratified_sample(X, y, n_rows=10_000)
    assert len(X_s) == len(X)
