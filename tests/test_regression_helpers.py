from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyRegressor

from src.modeling.regression import (
    SegmentedRegressor,
    _build_test_results_frame,
    _coerce_feature_matrix,
    _group_metrics_table,
    _make_stratification_bins,
    _prune_feature_columns,
    _split_feature_columns,
)


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


def test_group_metrics_table_builds_price_and_mileage_reports() -> None:
    x_test = pd.DataFrame({"mileage": [0, 10_000, 50_000, None]})
    y_test = pd.Series([1_500_000, 2_500_000, 5_500_000, 13_000_000], dtype=float)
    predictions = pd.Series([1_450_000, 2_450_000, 5_600_000, 12_700_000], dtype=float)

    results = _build_test_results_frame(x_test, y_test, predictions.to_numpy())
    price_metrics = _group_metrics_table(results, "price_segment")
    mileage_metrics = _group_metrics_table(results, "mileage_group")

    assert set(price_metrics["price_segment"]) == {"<=2M", "2-4M", "4-7M", "12M+"}
    assert set(mileage_metrics["mileage_group"]) == {"with_mileage", "without_mileage"}


def test_feature_matrix_coercion_casts_categorical_to_object_and_numeric_to_float() -> None:
    X = pd.DataFrame(
        {
            "brand": pd.Series(["bmw", "audi"], dtype="string"),
            "is_commercial_like": [True, False],
            "mileage": pd.Series([10_000, 20_000], dtype="Int64"),
        }
    )

    numeric_cols, categorical_cols = _split_feature_columns(X)
    coerced = _coerce_feature_matrix(X, numeric_cols, categorical_cols)

    assert str(coerced["mileage"].dtype) == "float64"
    assert str(coerced["brand"].dtype) == "object"
    assert str(coerced["is_commercial_like"].dtype) == "float64"


def test_segmented_regressor_routes_predictions_by_condition() -> None:
    X_train = pd.DataFrame(
        {
            "condition": ["new", "new", "used", "used"],
            "mileage": [0.0, 0.0, 100_000.0, 120_000.0],
        }
    )
    y_train = pd.Series([5_000_000.0, 5_200_000.0, 1_200_000.0, 1_400_000.0])

    estimator = SegmentedRegressor(
        segment_column="condition",
        base_estimator=DummyRegressor(strategy="mean"),
        drop_segment_feature=True,
    )
    estimator.fit(X_train, y_train)

    X_test = pd.DataFrame(
        {
            "condition": ["new", "used", "unknown"],
            "mileage": [0.0, 110_000.0, 50_000.0],
        }
    )
    predictions = estimator.predict(X_test)

    assert predictions[0] == 5_100_000.0
    assert predictions[1] == 1_300_000.0
    assert predictions[2] == 3_200_000.0
