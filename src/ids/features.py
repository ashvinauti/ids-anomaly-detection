"""Preprocessing: turn raw NSL-KDD records into numbers a model can use.

Two problems have to be solved before any model sees the data:

1. Three columns are text (protocol_type, service, flag). Models need
   numbers, so each text value becomes its own 0/1 column (one-hot encoding).
   One-hot is used instead of numbering the values 1, 2, 3... because a
   number would imply an order ("http" < "ftp") that does not exist.

2. The numeric columns are on very different scales. src_bytes can be in the
   millions while serror_rate is between 0 and 1. The SVM and K-Means both
   measure distances, so without scaling the byte counts would dominate every
   decision. StandardScaler gives each column mean 0 and standard deviation 1.

Both steps live in one ColumnTransformer so they can be placed inside a
scikit-learn Pipeline. The scaler and encoder are then learned from the
training data only and reused unchanged on test or live data, which avoids
leaking information from the test set into training.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ids.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def build_preprocessor() -> ColumnTransformer:
    """Create the (unfitted) preprocessing step used by every model.

    handle_unknown="ignore" matters for an IDS: if live traffic uses a
    service the model never saw in training, that record gets all-zero
    one-hot columns instead of crashing the detector.
    """
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
        ],
        # Any column not listed above (there should be none) is dropped,
        # so label columns can never slip into the model by accident.
        remainder="drop",
    )


def output_feature_names(fitted_preprocessor: ColumnTransformer) -> list[str]:
    """Return the column names after preprocessing, e.g. 'categorical__service_http'.

    Useful for checking the output shape and explaining what the model sees.
    """
    return list(fitted_preprocessor.get_feature_names_out())
