"""Shared test helpers: a small synthetic dataset in NSL-KDD column format.

The values are made up purely to exercise the code quickly. They are never
used for any reported result.
"""

import numpy as np
import pandas as pd
import pytest

from ids.data import FEATURE_COLUMNS, NUMERIC_COLUMNS


@pytest.fixture
def synthetic_features() -> tuple[pd.DataFrame, pd.Series]:
    """200 rows: 150 'normal' near zero, 50 'attack' with large values."""
    rng = np.random.default_rng(0)
    n_normal, n_attack = 150, 50
    n = n_normal + n_attack

    df = pd.DataFrame(index=range(n), columns=FEATURE_COLUMNS)
    numeric = rng.normal(0, 1, size=(n, len(NUMERIC_COLUMNS)))
    numeric[n_normal:] += 6  # push attack rows far away from normal ones
    df[NUMERIC_COLUMNS] = numeric
    df["protocol_type"] = rng.choice(["tcp", "udp", "icmp"], size=n)
    df["service"] = rng.choice(["http", "ftp", "private"], size=n)
    df["flag"] = rng.choice(["SF", "S0"], size=n)

    y = pd.Series([0] * n_normal + [1] * n_attack, name="is_attack")
    return df, y
