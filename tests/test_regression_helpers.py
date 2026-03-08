from __future__ import annotations

import pandas as pd

from src.modeling.regression import _make_stratification_bins, _prune_feature_columns


def test_prune_feature_columns_drops_year_when_age_exists() -> None:
    feature_cols = ["brand", "year", "age", "mileage"]

    pruned, dropped = _prune_feature_columns(feature_cols)

    assert "year" not in pruned
    assert "age" in pruned
    assert dropped == ["year"]


def test_make_stratification_bins_returns_balanced_bins_when_possible() -> None:
    y = pd.Series([1_000_000] * 8 + [2_000_000] * 8 + [3_000_000] * 8 + [4_000_000] * 8)

    bins = _make_stratification_bins(y, max_bins=6, min_count_per_bin=4)

    assert bins is not None
    assert len(bins) == len(y)
    assert bins.value_counts().min() >= 4
