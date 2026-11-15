"""The three detection approaches compared in this project.

- Neural network (MLPClassifier): supervised, learns from labelled
  normal and attack examples.
- Support vector machine (SVC, RBF kernel): supervised, trained on a
  stratified sample because SVC training time grows roughly with the
  square of the number of rows.
- K-Means detector: clustering-based anomaly detection. It learns what
  normal traffic looks like and flags anything far from it. It never
  sees an attack during training, so in principle it can flag attack
  types it has never seen. Strictly, this is "novelty detection": it
  uses the labels only to select the normal rows to train on.

Every model is wrapped in a Pipeline with the same preprocessor, so the
saved file contains both the data preparation and the model.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from ids.features import build_preprocessor

# One seed used everywhere so every run gives the same results.
RANDOM_STATE = 42

# Rows used to train the SVM. Large enough to cover the main traffic
# patterns, small enough to train in a few minutes on a laptop.
SVM_SAMPLE_SIZE = 20_000


class KMeansDetector(BaseEstimator):
    """Anomaly detector: distance to the nearest cluster of normal traffic.

    How it works:
      1. fit(): cluster the normal training records into n_clusters groups.
      2. Measure each normal record's distance to its nearest cluster centre.
      3. Set the threshold at the given percentile of those distances.
         With percentile=95, about 5% of normal training traffic sits above
         the threshold, so roughly 5% is the expected false positive rate on
         similar normal traffic.
      4. predict(): any record further than the threshold is an anomaly (1).

    The percentile is the main control: raise it to cut false positives
    (fewer alerts for analysts), lower it to catch more attacks.

    Subclassing BaseEstimator gives get_params/set_params, which lets this
    class be used inside a Pipeline and in GridSearchCV like any built-in model.
    """

    def __init__(self, n_clusters: int = 8, percentile: float = 95.0,
                 random_state: int = RANDOM_STATE):
        # scikit-learn convention: __init__ only stores parameters.
        self.n_clusters = n_clusters
        self.percentile = percentile
        self.random_state = random_state

    def fit(self, X, y=None) -> "KMeansDetector":
        """Learn clusters from normal traffic only.

        If y is given, rows labelled 1 (attack) are removed first. Training
        on attacks would place cluster centres on top of attack traffic and
        teach the detector that attacks are "normal".
        """
        X = np.asarray(X, dtype=float)
        if y is not None:
            X = X[np.asarray(y) == 0]
        if len(X) < self.n_clusters:
            raise ValueError("Need at least n_clusters normal rows to fit.")

        self.kmeans_ = KMeans(n_clusters=self.n_clusters, n_init=10,
                              random_state=self.random_state)
        self.kmeans_.fit(X)

        train_distances = self._nearest_centre_distance(X)
        self.threshold_ = float(np.percentile(train_distances, self.percentile))
        # Pipeline and metrics code expect a classifier to expose its classes.
        self.classes_ = np.array([0, 1])
        return self

    def _nearest_centre_distance(self, X: np.ndarray) -> np.ndarray:
        """Euclidean distance from each row to its closest cluster centre."""
        # KMeans.transform returns the distance to every centre; take the smallest.
        return self.kmeans_.transform(X).min(axis=1)

    def anomaly_score(self, X) -> np.ndarray:
        """Higher score = further from normal traffic = more suspicious."""
        return self._nearest_centre_distance(np.asarray(X, dtype=float))

    def predict(self, X) -> np.ndarray:
        """Return 1 (anomaly) when the score is above the threshold, else 0."""
        return (self.anomaly_score(X) > self.threshold_).astype(int)


def build_mlp_pipeline() -> Pipeline:
    """Preprocessing + a small two-layer neural network.

    early_stopping holds back 10% of the training data and stops when the
    score on it stops improving, which reduces overfitting and run time.
    """
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        early_stopping=True,
        max_iter=200,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def build_svm_pipeline() -> Pipeline:
    """Preprocessing + an RBF-kernel SVM (scikit-learn defaults for C and gamma).

    The RBF kernel lets the SVM draw curved boundaries between normal and
    attack traffic. Train it on stratified_sample(...) output, not the full set.
    """
    model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE)
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def build_kmeans_pipeline() -> Pipeline:
    """Preprocessing + the K-Means anomaly detector."""
    return Pipeline([("preprocess", build_preprocessor()),
                     ("model", KMeansDetector())])


def build_all_models() -> dict[str, Pipeline]:
    """All three untrained pipelines, keyed by a short display name."""
    return {
        "neural_network": build_mlp_pipeline(),
        "svm": build_svm_pipeline(),
        "kmeans": build_kmeans_pipeline(),
    }


def stratified_sample(X: pd.DataFrame, y: pd.Series, n_rows: int,
                      random_state: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.Series]:
    """Pick n_rows at random while keeping the normal/attack ratio of y.

    A plain random sample could, by chance, under-represent attacks; the
    stratify option guarantees the sample has the same class balance.
    """
    if n_rows >= len(X):
        return X, y
    X_sample, _, y_sample, _ = train_test_split(
        X, y, train_size=n_rows, stratify=y, random_state=random_state
    )
    return X_sample, y_sample
