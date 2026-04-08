"""Regression training and evaluation pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.constants import FIGURES_DIR, REPORTS_DIR
from src.utils.io_utils import ensure_dir

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RegressionConfig:
    """Configuration for regression experiments."""

    reports_dir: Path = REPORTS_DIR
    figures_dir: Path = FIGURES_DIR
    models_dir: Path = Path("models")
    target_column: str = "price"
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    max_stratification_bins: int = 10


class SegmentedRegressor(BaseEstimator, RegressorMixin):
    """Route rows to segment-specific regressors using a known feature such as condition."""

    def __init__(
        self,
        *,
        segment_column: str,
        base_estimator: Pipeline,
        drop_segment_feature: bool = True,
    ) -> None:
        self.segment_column = segment_column
        self.base_estimator = base_estimator
        self.drop_segment_feature = drop_segment_feature

    def _prepare_segment_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        if self.drop_segment_feature and self.segment_column in frame.columns:
            frame = frame.drop(columns=[self.segment_column])
        return frame

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SegmentedRegressor":
        frame = pd.DataFrame(X).copy()
        target = pd.Series(y).reset_index(drop=True)
        frame = frame.reset_index(drop=True)

        segments = frame[self.segment_column].astype(str).fillna("unknown")
        self.segment_models_: dict[str, Pipeline] = {}
        self.segment_counts_: dict[str, int] = {}

        for segment_value in sorted(segments.unique()):
            mask = segments == segment_value
            estimator = clone(self.base_estimator)
            estimator.fit(self._prepare_segment_frame(frame.loc[mask]), target.loc[mask])
            self.segment_models_[segment_value] = estimator
            self.segment_counts_[segment_value] = int(mask.sum())

        self.global_model_ = clone(self.base_estimator)
        self.global_model_.fit(self._prepare_segment_frame(frame), target)
        self.feature_names_in_ = np.array(frame.columns.tolist(), dtype=object)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        frame = pd.DataFrame(X).copy()
        frame = frame.reset_index(drop=True)
        segments = frame[self.segment_column].astype(str).fillna("unknown")
        predictions = np.zeros(len(frame), dtype=float)

        for segment_value in segments.unique():
            mask = segments == segment_value
            estimator = self.segment_models_.get(segment_value, self.global_model_)
            predictions[mask.to_numpy()] = estimator.predict(self._prepare_segment_frame(frame.loc[mask]))

        return predictions


def _make_ohe() -> OneHotEncoder:
    """Create OneHotEncoder compatible across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _build_preprocessor(
    numeric_cols: list[str], categorical_cols: list[str], scale_numeric: bool
) -> ColumnTransformer:
    """Build column transformer for preprocessing."""
    if scale_numeric:
        num_pipeline: Pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )
    else:
        num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_ohe()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_cols),
            ("cat", cat_pipeline, categorical_cols),
        ]
    )
    return preprocessor


def _build_pipeline_regressor(
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
    scale_numeric: bool,
    model: Any,
) -> Pipeline:
    """Build a preprocessing + model pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor(numeric_cols, categorical_cols, scale_numeric=scale_numeric)),
            ("model", model),
        ]
    )


def _split_feature_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split columns into numeric and categorical groups using pandas dtype checks."""
    numeric_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    categorical_cols = [col for col in X.columns if col not in numeric_cols]
    return numeric_cols, categorical_cols


def _coerce_feature_matrix(
    X: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> pd.DataFrame:
    """Cast numeric features to float and categorical features to plain object dtype."""
    coerced = X.copy()

    for column in numeric_cols:
        coerced[column] = pd.to_numeric(coerced[column], errors="coerce").astype(float)

    for column in categorical_cols:
        coerced[column] = coerced[column].astype(object)
        coerced[column] = coerced[column].where(pd.notna(coerced[column]), None)

    return coerced[numeric_cols + categorical_cols]


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAPE avoiding division by zero."""
    non_zero = y_true != 0
    if non_zero.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100)


def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": _mape(y_true, y_pred),
    }


def _prune_feature_columns(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """Drop redundant features that duplicate the same signal."""
    dropped_features: list[str] = []
    pruned = feature_cols.copy()

    # age = current_year - year; keeping both adds multicollinearity without helping tree models much.
    if "age" in pruned and "year" in pruned:
        pruned.remove("year")
        dropped_features.append("year")

    return pruned, dropped_features


def _linear_baseline_feature_columns(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """Build a compact, interpretable feature set for plain linear models."""
    pruned, dropped_features = _prune_feature_columns(feature_cols)

    excluded_for_linear = [
        "seller_type",
        "commercial_signal_count",
        "is_commercial_like",
        "description_len",
        "owners_count",
        "color",
        "body_type",
        "steering_wheel",
        "pts_type",
        "customs",
        "region",
    ]

    linear_pruned = pruned.copy()
    for column in excluded_for_linear:
        if column in linear_pruned:
            linear_pruned.remove(column)
            dropped_features.append(column)

    return linear_pruned, dropped_features


def _make_stratification_bins(
    y: pd.Series,
    *,
    max_bins: int,
    min_count_per_bin: int,
) -> pd.Series | None:
    """Create quantile bins suitable for stratified splitting."""
    unique_values = int(y.nunique(dropna=True))
    upper_bins = min(max_bins, unique_values)

    for bins_count in range(upper_bins, 1, -1):
        try:
            bins = pd.qcut(y, q=bins_count, labels=False, duplicates="drop")
        except ValueError:
            continue

        if bins is None:
            continue

        counts = bins.value_counts(dropna=True)
        if len(counts) < 2:
            continue
        if int(counts.min()) < min_count_per_bin:
            continue
        return bins.astype(int)

    return None


def _build_cv_splits(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: RegressionConfig,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], bool]:
    """Build CV splits, preferring stratification over price bins when possible."""
    stratify_bins = _make_stratification_bins(
        y_train,
        max_bins=config.max_stratification_bins,
        min_count_per_bin=config.cv_folds,
    )
    if stratify_bins is not None:
        splitter = StratifiedKFold(
            n_splits=config.cv_folds,
            shuffle=True,
            random_state=config.random_state,
        )
        return list(splitter.split(X_train, stratify_bins)), True

    splitter = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    return list(splitter.split(X_train, y_train)), False


def _wrap_target_transform(pipeline: Pipeline) -> TransformedTargetRegressor:
    """Train regressors on log-price while exposing predictions in the original price scale."""
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def _extract_feature_importance(trained_pipeline: TransformedTargetRegressor) -> pd.DataFrame:
    """Extract feature importance / coefficient magnitudes."""
    regressor = trained_pipeline.regressor_
    if isinstance(regressor, SegmentedRegressor):
        total = max(1, sum(regressor.segment_counts_.values()))
        aggregated: dict[str, float] = {}
        for segment_value, estimator in regressor.segment_models_.items():
            segment_weight = regressor.segment_counts_.get(segment_value, 0) / total
            preprocessor = estimator.named_steps["preprocessor"]
            model = estimator.named_steps["model"]

            feature_names = preprocessor.get_feature_names_out()
            values = None
            if hasattr(model, "feature_importances_"):
                values = getattr(model, "feature_importances_")
            elif hasattr(model, "coef_"):
                values = np.abs(getattr(model, "coef_").ravel())

            if values is None:
                continue

            for feature_name, value in zip(feature_names, values, strict=False):
                aggregated[feature_name] = aggregated.get(feature_name, 0.0) + float(value) * segment_weight

        if not aggregated:
            return pd.DataFrame(columns=["feature", "importance"])

        return (
            pd.DataFrame(
                [{"feature": feature, "importance": importance} for feature, importance in aggregated.items()]
            )
            .sort_values(by="importance", ascending=False)
            .reset_index(drop=True)
        )

    preprocessor = regressor.named_steps["preprocessor"]
    model = regressor.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    values = None

    if hasattr(model, "feature_importances_"):
        values = getattr(model, "feature_importances_")
    elif hasattr(model, "coef_"):
        coef = getattr(model, "coef_")
        values = np.abs(coef.ravel())

    if values is None:
        return pd.DataFrame(columns=["feature", "importance"])

    size = min(len(feature_names), len(values))
    return (
        pd.DataFrame({"feature": feature_names[:size], "importance": values[:size]})
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )


def _plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, figures_dir: Path, prefix: str) -> None:
    """Save prediction diagnostics plots."""
    fig = plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    diagonal_min = min(float(y_true.min()), float(y_pred.min()))
    diagonal_max = max(float(y_true.max()), float(y_pred.max()))
    plt.plot([diagonal_min, diagonal_max], [diagonal_min, diagonal_max], "r--", linewidth=1)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Predicted vs Actual")
    plt.tight_layout()
    fig.savefig(figures_dir / f"{prefix}_predicted_vs_actual.png", dpi=150)
    plt.close(fig)

    residuals = y_true - y_pred
    fig = plt.figure(figsize=(7, 4))
    sns.histplot(residuals, kde=True, color="#264653")
    plt.title("Residual Distribution")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    fig.savefig(figures_dir / f"{prefix}_residuals_hist.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(6, 5))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual")
    plt.title("Residuals vs Predicted")
    plt.tight_layout()
    fig.savefig(figures_dir / f"{prefix}_residuals_vs_predicted.png", dpi=150)
    plt.close(fig)


def _build_test_results_frame(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Assemble test predictions with derived error slices."""
    results = X_test.reset_index(drop=True).copy()
    results["actual"] = y_test.reset_index(drop=True).astype(float)
    results["predicted"] = predictions.astype(float)
    results["abs_error"] = (results["actual"] - results["predicted"]).abs()
    results["ape"] = np.where(
        results["actual"] != 0,
        results["abs_error"] / results["actual"] * 100,
        np.nan,
    )

    price_bins = [0, 2_000_000, 4_000_000, 7_000_000, 12_000_000, float("inf")]
    price_labels = ["<=2M", "2-4M", "4-7M", "7-12M", "12M+"]
    results["price_segment"] = pd.cut(
        results["actual"],
        bins=price_bins,
        labels=price_labels,
        include_lowest=True,
    )

    mileage = pd.to_numeric(results.get("mileage"), errors="coerce")
    results["mileage_group"] = np.where(mileage.fillna(0) > 0, "with_mileage", "without_mileage")
    return results


def _group_metrics_table(results: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Compute evaluation metrics for each group in the test set."""
    rows: list[dict[str, object]] = []
    for group_value, group_df in results.groupby(group_col, dropna=False, observed=False):
        if len(group_df) == 0:
            continue

        y_true = group_df["actual"].to_numpy()
        y_pred = group_df["predicted"].to_numpy()
        metrics = _evaluate_predictions(y_true, y_pred) if len(group_df) >= 2 else {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float("nan"),
            "mape": _mape(y_true, y_pred),
        }
        rows.append(
            {
                group_col: str(group_value),
                "n": int(len(group_df)),
                "mean_actual": float(group_df["actual"].mean()),
                "median_actual": float(group_df["actual"].median()),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "mape": float(group_df["ape"].dropna().mean()),
                "median_abs_error": float(group_df["abs_error"].median()),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[group_col, "n", "mean_actual", "median_actual", "mae", "rmse", "r2", "mape", "median_abs_error"]
        )

    return pd.DataFrame(rows)


def _write_group_report(df: pd.DataFrame, group_col: str, title: str, path: Path) -> None:
    """Write a short markdown report for a grouped metrics table."""
    lines = [f"# {title}", ""]
    for row in df.to_dict(orient="records"):
        lines.append(
            "- {group} | n={n} | MAE={mae:.2f} | RMSE={rmse:.2f} | R2={r2:.4f} | MAPE={mape:.2f}%".format(
                group=row[group_col],
                n=row["n"],
                mae=row["mae"],
                rmse=row["rmse"],
                r2=row["r2"],
                mape=row["mape"],
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_regression_experiment(df: pd.DataFrame, config: RegressionConfig) -> dict[str, Any]:
    """Train, compare, and persist regression models."""
    ensure_dir(config.reports_dir)
    ensure_dir(config.figures_dir)
    ensure_dir(config.models_dir)

    if config.target_column not in df.columns:
        raise ValueError(f"Target column '{config.target_column}' is missing")

    model_df = df.copy()
    model_df = model_df[model_df[config.target_column].notna()].copy()

    if "description_text" in model_df.columns:
        model_df["description_len"] = model_df["description_text"].astype(str).str.len()

    drop_cols = {
        config.target_column,
        "url",
        "parsed_at",
        "description_text",
        "price_log1p",
    }
    feature_cols = [col for col in model_df.columns if col not in drop_cols]
    feature_cols, dropped_features = _prune_feature_columns(feature_cols)
    linear_feature_cols, linear_dropped_features = _linear_baseline_feature_columns(feature_cols)

    X = model_df[feature_cols]
    y = model_df[config.target_column].astype(float)

    numeric_cols, categorical_cols = _split_feature_columns(X)

    numeric_cols = [col for col in numeric_cols if X[col].notna().any()]
    categorical_cols = [col for col in categorical_cols if X[col].notna().any()]
    X = _coerce_feature_matrix(X[numeric_cols + categorical_cols], numeric_cols, categorical_cols)

    linear_numeric_cols = [col for col in numeric_cols if col in linear_feature_cols]
    linear_categorical_cols = [col for col in categorical_cols if col in linear_feature_cols]

    stratify_bins = _make_stratification_bins(
        y,
        max_bins=config.max_stratification_bins,
        min_count_per_bin=2,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify_bins,
    )

    cv_splits, used_stratified_cv = _build_cv_splits(X_train, y_train, config)
    used_stratified_split = stratify_bins is not None

    model_defs: dict[str, TransformedTargetRegressor] = {
        "linear_regression": (
            _wrap_target_transform(
                _build_pipeline_regressor(
                    numeric_cols=linear_numeric_cols,
                    categorical_cols=linear_categorical_cols,
                    scale_numeric=True,
                    model=LinearRegression(),
                )
            )
        ),
        "ridge": (
            _wrap_target_transform(
                _build_pipeline_regressor(
                    numeric_cols=linear_numeric_cols,
                    categorical_cols=linear_categorical_cols,
                    scale_numeric=True,
                    model=Ridge(random_state=config.random_state),
                )
            )
        ),
        "lasso": (
            _wrap_target_transform(
                _build_pipeline_regressor(
                    numeric_cols=linear_numeric_cols,
                    categorical_cols=linear_categorical_cols,
                    scale_numeric=True,
                    model=Lasso(random_state=config.random_state, max_iter=50_000),
                )
            )
        ),
        "random_forest": (
            _wrap_target_transform(
                _build_pipeline_regressor(
                    numeric_cols=numeric_cols,
                    categorical_cols=categorical_cols,
                    scale_numeric=False,
                    model=RandomForestRegressor(
                        n_estimators=350,
                        random_state=config.random_state,
                        n_jobs=-1,
                        min_samples_leaf=1,
                    ),
                )
            )
        ),
        "gradient_boosting": (
            _wrap_target_transform(
                _build_pipeline_regressor(
                    numeric_cols=numeric_cols,
                    categorical_cols=categorical_cols,
                    scale_numeric=False,
                    model=GradientBoostingRegressor(random_state=config.random_state),
                )
            )
        ),
    }

    try:
        from xgboost import XGBRegressor

        xgb_pipeline = _build_pipeline_regressor(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            scale_numeric=False,
            model=XGBRegressor(
                random_state=config.random_state,
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                n_jobs=-1,
            ),
        )
        model_defs["xgboost"] = _wrap_target_transform(xgb_pipeline)
        if "condition" in X_train.columns and X_train["condition"].nunique(dropna=True) >= 2:
            segmented_numeric_cols = [col for col in numeric_cols if col != "condition"]
            segmented_categorical_cols = [col for col in categorical_cols if col != "condition"]
            segmented_xgb_pipeline = _build_pipeline_regressor(
                numeric_cols=segmented_numeric_cols,
                categorical_cols=segmented_categorical_cols,
                scale_numeric=False,
                model=XGBRegressor(
                    random_state=config.random_state,
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="reg:squarederror",
                    n_jobs=-1,
                ),
            )
            model_defs["condition_segmented_xgboost"] = _wrap_target_transform(
                SegmentedRegressor(
                    segment_column="condition",
                    base_estimator=segmented_xgb_pipeline,
                    drop_segment_feature=True,
                )
            )
    except Exception:  # noqa: BLE001
        LOGGER.info("xgboost is not available, skipping this model.")

    metrics_rows: list[dict[str, Any]] = []
    cv_rows: list[dict[str, Any]] = []
    trained_models: dict[str, TransformedTargetRegressor] = {}

    for name, pipeline in model_defs.items():
        LOGGER.info("Training model: %s", name)
        pipeline.fit(X_train, y_train)
        predictions = np.maximum(pipeline.predict(X_test), 0.0)

        metrics = _evaluate_predictions(y_test.to_numpy(), predictions)
        metrics_rows.append({"model": name, **metrics})

        cv = cross_validate(
            pipeline,
            X_train,
            y_train,
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2",
            },
            n_jobs=-1,
            cv=cv_splits,
        )

        cv_rows.append(
            {
                "model": name,
                "cv_mae_mean": float(-np.mean(cv["test_mae"])),
                "cv_rmse_mean": float(-np.mean(cv["test_rmse"])),
                "cv_r2_mean": float(np.mean(cv["test_r2"])),
            }
        )
        trained_models[name] = pipeline

    LOGGER.info("Hyperparameter tuning for ridge and random_forest...")
    ridge_search = GridSearchCV(
        estimator=_wrap_target_transform(
            Pipeline(
                steps=[
                    (
                        "preprocessor",
                        _build_preprocessor(
                            linear_numeric_cols,
                            linear_categorical_cols,
                            scale_numeric=True,
                        ),
                    ),
                    ("model", Ridge(random_state=config.random_state)),
                ]
            )
        ),
        param_grid={"regressor__model__alpha": [0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]},
        scoring="neg_root_mean_squared_error",
        cv=cv_splits,
        n_jobs=-1,
    )
    ridge_search.fit(X_train, y_train)
    ridge_best = ridge_search.best_estimator_
    ridge_pred = np.maximum(ridge_best.predict(X_test), 0.0)
    metrics_rows.append({"model": "ridge_tuned", **_evaluate_predictions(y_test.to_numpy(), ridge_pred)})
    ridge_cv = cross_validate(
        ridge_best,
        X_train,
        y_train,
        cv=cv_splits,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        n_jobs=-1,
    )
    cv_rows.append(
        {
            "model": "ridge_tuned",
            "cv_mae_mean": float(-np.mean(ridge_cv["test_mae"])),
            "cv_rmse_mean": float(-np.mean(ridge_cv["test_rmse"])),
            "cv_r2_mean": float(np.mean(ridge_cv["test_r2"])),
        }
    )
    trained_models["ridge_tuned"] = ridge_best

    rf_search = RandomizedSearchCV(
        estimator=_wrap_target_transform(
            Pipeline(
                steps=[
                    ("preprocessor", _build_preprocessor(numeric_cols, categorical_cols, scale_numeric=False)),
                    ("model", RandomForestRegressor(random_state=config.random_state, n_jobs=-1)),
                ]
            )
        ),
        param_distributions={
            "regressor__model__n_estimators": [200, 300, 500],
            "regressor__model__max_depth": [None, 10, 20, 30],
            "regressor__model__min_samples_split": [2, 5, 10],
            "regressor__model__min_samples_leaf": [1, 2, 4],
            "regressor__model__max_features": ["sqrt", "log2", None],
        },
        n_iter=12,
        scoring="neg_root_mean_squared_error",
        cv=cv_splits,
        random_state=config.random_state,
        n_jobs=-1,
    )
    rf_search.fit(X_train, y_train)
    rf_best = rf_search.best_estimator_
    rf_pred = np.maximum(rf_best.predict(X_test), 0.0)
    metrics_rows.append({"model": "random_forest_tuned", **_evaluate_predictions(y_test.to_numpy(), rf_pred)})
    rf_cv = cross_validate(
        rf_best,
        X_train,
        y_train,
        cv=cv_splits,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        n_jobs=-1,
    )
    cv_rows.append(
        {
            "model": "random_forest_tuned",
            "cv_mae_mean": float(-np.mean(rf_cv["test_mae"])),
            "cv_rmse_mean": float(-np.mean(rf_cv["test_rmse"])),
            "cv_r2_mean": float(np.mean(rf_cv["test_r2"])),
        }
    )
    trained_models["random_forest_tuned"] = rf_best

    metrics_df = pd.DataFrame(metrics_rows).sort_values(by="rmse", ascending=True).reset_index(drop=True)
    cv_df = pd.DataFrame(cv_rows).sort_values(by="cv_rmse_mean", ascending=True).reset_index(drop=True)

    metrics_df.to_csv(config.reports_dir / "stage2_model_metrics.csv", index=False)
    cv_df.to_csv(config.reports_dir / "stage2_cv_metrics.csv", index=False)

    best_model_name = str(metrics_df.iloc[0]["model"])
    best_model = trained_models[best_model_name]
    best_pred = np.maximum(best_model.predict(X_test), 0.0)
    test_results_df = _build_test_results_frame(X_test, y_test, best_pred)

    model_path = config.models_dir / "best_price_model.joblib"
    joblib.dump(best_model, model_path)

    _plot_predictions(y_test.to_numpy(), best_pred, config.figures_dir, prefix="stage2_best_model")

    importance_df = _extract_feature_importance(best_model)
    importance_path = config.reports_dir / "stage2_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    price_segment_metrics = _group_metrics_table(test_results_df, "price_segment")
    price_segment_metrics_path = config.reports_dir / "stage2_metrics_by_price_segment.csv"
    price_segment_metrics.to_csv(price_segment_metrics_path, index=False)
    price_segment_report_path = config.reports_dir / "stage2_price_segment_report.md"
    _write_group_report(
        price_segment_metrics,
        group_col="price_segment",
        title="Stage 2 Metrics by Price Segment",
        path=price_segment_report_path,
    )

    mileage_group_metrics = _group_metrics_table(test_results_df, "mileage_group")
    mileage_group_metrics_path = config.reports_dir / "stage2_metrics_by_mileage_group.csv"
    mileage_group_metrics.to_csv(mileage_group_metrics_path, index=False)
    mileage_group_report_path = config.reports_dir / "stage2_mileage_group_report.md"
    _write_group_report(
        mileage_group_metrics,
        group_col="mileage_group",
        title="Stage 2 Metrics by Mileage Group",
        path=mileage_group_report_path,
    )

    condition_metrics = pd.DataFrame()
    condition_metrics_path = config.reports_dir / "stage2_metrics_by_condition.csv"
    condition_report_path = config.reports_dir / "stage2_condition_report.md"
    if "condition" in test_results_df.columns:
        condition_metrics = _group_metrics_table(test_results_df, "condition")
        condition_metrics.to_csv(condition_metrics_path, index=False)
        _write_group_report(
            condition_metrics,
            group_col="condition",
            title="Stage 2 Metrics by Condition",
            path=condition_report_path,
        )

    if not importance_df.empty:
        fig = plt.figure(figsize=(9, 6))
        top = importance_df.head(25)
        sns.barplot(data=top, y="feature", x="importance", hue="feature", palette="mako", legend=False)
        plt.title("Best Model: Top Feature Importance")
        plt.tight_layout()
        fig.savefig(config.figures_dir / "stage2_feature_importance.png", dpi=150)
        plt.close(fig)

    summary = {
        "best_model": best_model_name,
        "best_model_path": str(model_path),
        "best_metrics": metrics_df.iloc[0].to_dict(),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features_count": int(X.shape[1]),
        "dropped_features": dropped_features,
        "linear_baseline_features_count": int(len(linear_feature_cols)),
        "linear_baseline_dropped_features": linear_dropped_features,
        "target_transform": "log1p/expm1",
        "used_stratified_split": used_stratified_split,
        "used_stratified_cv": used_stratified_cv,
        "uses_condition_routing": "condition_segmented" in best_model_name,
    }

    summary_path = config.reports_dir / "stage2_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# Stage 2 Regression Report",
        "",
        f"Train rows: **{len(X_train)}**",
        f"Test rows: **{len(X_test)}**",
        f"Features: **{X.shape[1]}**",
        f"Target transform: **log1p(price)** during training, inverse **expm1** for predictions",
        f"Stratified train/test split by price bins: **{used_stratified_split}**",
        f"Stratified cross-validation by price bins: **{used_stratified_cv}**",
        "",
        "## Best model",
        f"- Name: `{best_model_name}`",
        f"- MAE: {metrics_df.iloc[0]['mae']:.2f}",
        f"- RMSE: {metrics_df.iloc[0]['rmse']:.2f}",
        f"- R2: {metrics_df.iloc[0]['r2']:.4f}",
        f"- MAPE: {metrics_df.iloc[0]['mape']:.2f}%",
        "",
        "## Feature handling",
        f"- Dropped redundant features: `{', '.join(dropped_features) if dropped_features else 'none'}`",
        f"- Dry linear baseline features count: `{len(linear_feature_cols)}`",
        f"- Dry linear baseline excluded features: `{', '.join(linear_dropped_features) if linear_dropped_features else 'none'}`",
        "",
        "## Saved artifacts",
        f"- Best model: `{model_path}`",
        f"- Metrics table: `{config.reports_dir / 'stage2_model_metrics.csv'}`",
        f"- CV metrics table: `{config.reports_dir / 'stage2_cv_metrics.csv'}`",
        f"- Importance table: `{importance_path}`",
        f"- Price segment metrics: `{price_segment_metrics_path}`",
        f"- Price segment report: `{price_segment_report_path}`",
        f"- Mileage group metrics: `{mileage_group_metrics_path}`",
        f"- Mileage group report: `{mileage_group_report_path}`",
        f"- Condition metrics: `{condition_metrics_path}`",
        f"- Condition report: `{condition_report_path}`",
        "",
        "## Price Segment Errors",
    ]

    for row in price_segment_metrics.to_dict(orient="records"):
        report_lines.append(
            "- {segment}: n={n}, MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}, MAPE={mape:.2f}%".format(
                segment=row["price_segment"],
                n=row["n"],
                mae=row["mae"],
                rmse=row["rmse"],
                r2=row["r2"],
                mape=row["mape"],
            )
        )

    report_lines.extend(["", "## Mileage Group Errors"])

    for row in mileage_group_metrics.to_dict(orient="records"):
        report_lines.append(
            "- {group}: n={n}, MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}, MAPE={mape:.2f}%".format(
                group=row["mileage_group"],
                n=row["n"],
                mae=row["mae"],
                rmse=row["rmse"],
                r2=row["r2"],
                mape=row["mape"],
            )
        )

    if not condition_metrics.empty:
        report_lines.extend(["", "## Condition Errors"])
        for row in condition_metrics.to_dict(orient="records"):
            report_lines.append(
                "- {group}: n={n}, MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}, MAPE={mape:.2f}%".format(
                    group=row["condition"],
                    n=row["n"],
                    mae=row["mae"],
                    rmse=row["rmse"],
                    r2=row["r2"],
                    mape=row["mape"],
                )
            )

    (config.reports_dir / "stage2_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    LOGGER.info("Stage 2 regression completed. Best model: %s", best_model_name)

    return summary
