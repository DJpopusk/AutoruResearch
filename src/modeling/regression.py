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

    X = model_df[feature_cols]
    y = model_df[config.target_column].astype(float)

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]

    numeric_cols = [col for col in numeric_cols if X[col].notna().any()]
    categorical_cols = [col for col in categorical_cols if X[col].notna().any()]
    X = X[numeric_cols + categorical_cols]

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
                Pipeline(
                    steps=[
                        ("preprocessor", _build_preprocessor(numeric_cols, categorical_cols, scale_numeric=True)),
                        ("model", LinearRegression()),
                    ]
                )
            )
        ),
        "ridge": (
            _wrap_target_transform(
                Pipeline(
                    steps=[
                        ("preprocessor", _build_preprocessor(numeric_cols, categorical_cols, scale_numeric=True)),
                        ("model", Ridge(random_state=config.random_state)),
                    ]
                )
            )
        ),
        "lasso": (
            _wrap_target_transform(
                Pipeline(
                    steps=[
                        ("preprocessor", _build_preprocessor(numeric_cols, categorical_cols, scale_numeric=True)),
                        ("model", Lasso(random_state=config.random_state, max_iter=50_000)),
                    ]
                )
            )
        ),
        "random_forest": (
            _wrap_target_transform(
                Pipeline(
                    steps=[
                        ("preprocessor", _build_preprocessor(numeric_cols, categorical_cols, scale_numeric=False)),
                        (
                            "model",
                            RandomForestRegressor(
                                n_estimators=350,
                                random_state=config.random_state,
                                n_jobs=-1,
                                min_samples_leaf=1,
                            ),
                        ),
                    ]
                )
            )
        ),
        "gradient_boosting": (
            _wrap_target_transform(
                Pipeline(
                    steps=[
                        ("preprocessor", _build_preprocessor(numeric_cols, categorical_cols, scale_numeric=False)),
                        ("model", GradientBoostingRegressor(random_state=config.random_state)),
                    ]
                )
            )
        ),
    }

    try:
        from xgboost import XGBRegressor

        model_defs["xgboost"] = _wrap_target_transform(
            Pipeline(
                steps=[
                    ("preprocessor", _build_preprocessor(numeric_cols, categorical_cols, scale_numeric=False)),
                    (
                        "model",
                        XGBRegressor(
                            random_state=config.random_state,
                            n_estimators=500,
                            learning_rate=0.05,
                            max_depth=6,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            objective="reg:squarederror",
                            n_jobs=-1,
                        ),
                    ),
                ]
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
                    ("preprocessor", _build_preprocessor(numeric_cols, categorical_cols, scale_numeric=True)),
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

    model_path = config.models_dir / "best_price_model.joblib"
    joblib.dump(best_model, model_path)

    _plot_predictions(y_test.to_numpy(), best_pred, config.figures_dir, prefix="stage2_best_model")

    importance_df = _extract_feature_importance(best_model)
    importance_path = config.reports_dir / "stage2_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)

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
        "target_transform": "log1p/expm1",
        "used_stratified_split": used_stratified_split,
        "used_stratified_cv": used_stratified_cv,
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
        "",
        "## Saved artifacts",
        f"- Best model: `{model_path}`",
        f"- Metrics table: `{config.reports_dir / 'stage2_model_metrics.csv'}`",
        f"- CV metrics table: `{config.reports_dir / 'stage2_cv_metrics.csv'}`",
        f"- Importance table: `{importance_path}`",
    ]

    (config.reports_dir / "stage2_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    LOGGER.info("Stage 2 regression completed. Best model: %s", best_model_name)

    return summary
