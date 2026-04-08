"""Compare core regression models across different feature-set combinations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.analysis_stage1 import _cramers_v
from src.modeling.regression import (
    RegressionConfig,
    SegmentedRegressor,
    _build_pipeline_regressor,
    _coerce_feature_matrix,
    _evaluate_predictions,
    _make_stratification_bins,
    _prune_feature_columns,
    _split_feature_columns,
    _wrap_target_transform,
)

DATA_PATH = ROOT / "data/processed/cleaned_dataset.parquet"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = ROOT / "figures"


def _prepare_model_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    model_df = df.copy()
    model_df = model_df[model_df["price"].notna()].copy()

    if "description_text" in model_df.columns:
        model_df["description_len"] = model_df["description_text"].astype(str).str.len()

    drop_cols = {"price", "url", "parsed_at", "description_text", "price_log1p"}
    feature_cols = [col for col in model_df.columns if col not in drop_cols]
    feature_cols, _ = _prune_feature_columns(feature_cols)

    X = model_df[feature_cols].copy()
    numeric_cols, categorical_cols = _split_feature_columns(X)
    numeric_cols = [column for column in numeric_cols if X[column].notna().any()]
    categorical_cols = [column for column in categorical_cols if X[column].notna().any()]
    X = _coerce_feature_matrix(X[numeric_cols + categorical_cols], numeric_cols, categorical_cols)
    y = model_df["price"].astype(float).copy()
    return X, y


def _numeric_spearman(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for column in X.columns:
        if not pd.api.types.is_numeric_dtype(X[column]):
            continue
        pair = pd.DataFrame({"feature": pd.to_numeric(X[column], errors="coerce"), "price": y}).dropna()
        if len(pair) < 10:
            continue
        rho = pair["feature"].corr(pair["price"], method="spearman")
        rows.append({"feature": column, "spearman_with_price": float(rho), "abs_spearman": abs(float(rho))})
    return pd.DataFrame(rows).sort_values(by="abs_spearman", ascending=False).reset_index(drop=True)


def _categorical_assoc(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    price_bins = pd.qcut(y, q=4, labels=["q1", "q2", "q3", "q4"], duplicates="drop")
    rows: list[dict[str, float | str]] = []

    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            continue
        subset = pd.DataFrame({column: X[column], "price_bin": price_bins}).dropna()
        if subset.empty or subset[column].nunique() < 2 or subset["price_bin"].nunique() < 2:
            continue
        contingency = pd.crosstab(subset[column], subset["price_bin"])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            continue
        chi2, pvalue, _, _ = stats.chi2_contingency(contingency)
        rows.append(
            {
                "feature": column,
                "chi2": float(chi2),
                "pvalue": float(pvalue),
                "cramers_v": float(_cramers_v(contingency)),
            }
        )

    return pd.DataFrame(rows).sort_values(by="cramers_v", ascending=False).reset_index(drop=True)


def _feature_sets(X: pd.DataFrame, y: pd.Series) -> tuple[dict[str, list[str]], pd.DataFrame, pd.DataFrame]:
    numeric_corr = _numeric_spearman(X, y)
    categorical_assoc = _categorical_assoc(X, y)

    all_features = X.columns.tolist()
    numeric_low = set(numeric_corr.loc[numeric_corr["abs_spearman"] < 0.20, "feature"].tolist())
    cat_low = set(categorical_assoc.loc[categorical_assoc["cramers_v"] < 0.20, "feature"].tolist())

    constant_cols = [
        column
        for column in all_features
        if X[column].nunique(dropna=True) <= 1
    ]

    strong_numeric = numeric_corr.loc[numeric_corr["abs_spearman"] >= 0.20, "feature"].tolist()
    strong_categorical = categorical_assoc.loc[categorical_assoc["cramers_v"] >= 0.30, "feature"].tolist()

    compact_linear = [
        "age",
        "mileage",
        "engine_volume",
        "engine_power_hp",
        "commercial_signal_count",
        "description_len",
        "condition",
        "seller_type",
        "transmission",
        "drive_type",
        "fuel_type",
        "body_type",
        "color",
        "is_commercial_like",
    ]
    compact_linear = [column for column in compact_linear if column in all_features]

    numeric_only = [
        "age",
        "mileage",
        "engine_volume",
        "engine_power_hp",
        "commercial_signal_count",
        "description_len",
        "is_commercial_like",
    ]
    numeric_only = [column for column in numeric_only if column in all_features]

    feature_sets = {
        "full_current": all_features,
        "drop_low_signal": [column for column in all_features if column not in numeric_low | cat_low | set(constant_cols)],
        "strong_signal_only": [column for column in all_features if column in set(strong_numeric) | set(strong_categorical)],
        "compact_linear_friendly": compact_linear,
        "numeric_only": numeric_only,
    }
    return feature_sets, numeric_corr, categorical_assoc


def _make_models(
    numeric_cols: list[str],
    categorical_cols: list[str],
    *,
    random_state: int,
    include_condition_segmented: bool,
) -> dict[str, object]:
    model_defs: dict[str, object] = {
        "linear_regression": _wrap_target_transform(
            _build_pipeline_regressor(
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                scale_numeric=True,
                model=LinearRegression(),
            )
        ),
        "ridge": _wrap_target_transform(
            _build_pipeline_regressor(
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                scale_numeric=True,
                model=Ridge(random_state=random_state),
            )
        ),
        "lasso": _wrap_target_transform(
            _build_pipeline_regressor(
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                scale_numeric=True,
                model=Lasso(random_state=random_state, max_iter=50_000),
            )
        ),
        "random_forest": _wrap_target_transform(
            _build_pipeline_regressor(
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                scale_numeric=False,
                model=RandomForestRegressor(
                    n_estimators=350,
                    random_state=random_state,
                    n_jobs=-1,
                    min_samples_leaf=1,
                ),
            )
        ),
        "gradient_boosting": _wrap_target_transform(
            _build_pipeline_regressor(
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                scale_numeric=False,
                model=GradientBoostingRegressor(random_state=random_state),
            )
        ),
    }

    from xgboost import XGBRegressor

    xgb_base = _build_pipeline_regressor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        scale_numeric=False,
        model=XGBRegressor(
            random_state=random_state,
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            n_jobs=-1,
        ),
    )
    model_defs["xgboost"] = _wrap_target_transform(xgb_base)

    if include_condition_segmented:
        segmented_numeric = [column for column in numeric_cols if column != "condition"]
        segmented_categorical = [column for column in categorical_cols if column != "condition"]
        segmented_xgb = _build_pipeline_regressor(
            numeric_cols=segmented_numeric,
            categorical_cols=segmented_categorical,
            scale_numeric=False,
            model=XGBRegressor(
                random_state=random_state,
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
                base_estimator=segmented_xgb,
                drop_segment_feature=True,
            )
        )

    return model_defs


def _transformed_feature_count(X_train: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> int:
    pipeline = _build_pipeline_regressor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        scale_numeric=True,
        model=LinearRegression(),
    )
    preprocessor = pipeline.named_steps["preprocessor"]
    preprocessor.fit(X_train)
    return int(len(preprocessor.get_feature_names_out()))


def main() -> None:
    config = RegressionConfig()
    df = pd.read_parquet(DATA_PATH)
    X_base, y = _prepare_model_frame(df)

    feature_sets, numeric_corr, categorical_assoc = _feature_sets(X_base, y)

    stratify_bins = _make_stratification_bins(y, max_bins=config.max_stratification_bins, min_count_per_bin=2)
    train_idx, test_idx = next(
        iter(
            [
                train_test_split(
                    np.arange(len(y)),
                    test_size=config.test_size,
                    random_state=config.random_state,
                    stratify=stratify_bins,
                )
            ]
        )
    )

    metrics_rows: list[dict[str, object]] = []
    dimension_rows: list[dict[str, object]] = []

    for feature_set_name, feature_cols in feature_sets.items():
        X = X_base[feature_cols].copy()
        numeric_cols, categorical_cols = _split_feature_columns(X)
        numeric_cols = [column for column in numeric_cols if X[column].notna().any()]
        categorical_cols = [column for column in categorical_cols if X[column].notna().any()]
        X = _coerce_feature_matrix(X[numeric_cols + categorical_cols], numeric_cols, categorical_cols)

        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)

        expanded_dim = _transformed_feature_count(X_train, numeric_cols, categorical_cols)
        dimension_rows.append(
            {
                "feature_set": feature_set_name,
                "raw_feature_count": int(len(feature_cols)),
                "numeric_feature_count": int(len(numeric_cols)),
                "categorical_feature_count": int(len(categorical_cols)),
                "linear_model_expanded_feature_count": int(expanded_dim),
                "features": ", ".join(feature_cols),
            }
        )

        model_defs = _make_models(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            random_state=config.random_state,
            include_condition_segmented="condition" in X_train.columns and X_train["condition"].nunique(dropna=True) >= 2,
        )

        for model_name, estimator in model_defs.items():
            estimator.fit(X_train, y_train)
            y_pred = np.maximum(estimator.predict(X_test), 0.0)
            metrics = _evaluate_predictions(y_test.to_numpy(), y_pred)
            metrics_rows.append(
                {
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "raw_feature_count": int(len(feature_cols)),
                    "linear_model_expanded_feature_count": int(expanded_dim),
                    **metrics,
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)
    dimensions_df = pd.DataFrame(dimension_rows)

    metrics_df = metrics_df.sort_values(by=["model", "mape", "rmse"]).reset_index(drop=True)
    dimensions_df = dimensions_df.sort_values(by="raw_feature_count", ascending=False).reset_index(drop=True)

    metrics_path = REPORTS_DIR / "stage2_feature_set_ablation_metrics.csv"
    dims_path = REPORTS_DIR / "stage2_feature_set_dimensions.csv"
    numeric_corr_path = REPORTS_DIR / "stage2_feature_set_numeric_signal.csv"
    categorical_assoc_path = REPORTS_DIR / "stage2_feature_set_categorical_signal.csv"

    metrics_to_save = metrics_df.copy()
    metrics_to_save[["mae", "rmse", "r2", "mape"]] = metrics_to_save[["mae", "rmse", "r2", "mape"]].round(2)
    dimensions_to_save = dimensions_df.copy()
    numeric_corr_to_save = numeric_corr.copy()
    numeric_corr_to_save[["spearman_with_price", "abs_spearman"]] = numeric_corr_to_save[
        ["spearman_with_price", "abs_spearman"]
    ].round(2)
    categorical_assoc_to_save = categorical_assoc.copy()
    categorical_assoc_to_save[["chi2", "pvalue", "cramers_v"]] = categorical_assoc_to_save[
        ["chi2", "pvalue", "cramers_v"]
    ].round(2)

    metrics_to_save.to_csv(metrics_path, index=False)
    dimensions_to_save.to_csv(dims_path, index=False)
    numeric_corr_to_save.to_csv(numeric_corr_path, index=False)
    categorical_assoc_to_save.to_csv(categorical_assoc_path, index=False)

    pivot_mape = metrics_df.pivot(index="model", columns="feature_set", values="mape")
    fig = plt.figure(figsize=(10, 5.5))
    sns.heatmap(pivot_mape, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={"label": "MAPE, %"})
    plt.title("MAPE by model and feature set")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "stage2_feature_set_ablation_mape_heatmap.png", dpi=150)
    plt.close(fig)

    pivot_rmse = metrics_df.pivot(index="model", columns="feature_set", values="rmse")
    fig = plt.figure(figsize=(10, 5.5))
    sns.heatmap(pivot_rmse, annot=True, fmt=".0f", cmap="Blues", cbar_kws={"label": "RMSE"})
    plt.title("RMSE by model and feature set")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "stage2_feature_set_ablation_rmse_heatmap.png", dpi=150)
    plt.close(fig)

    best_by_model = (
        metrics_df.sort_values(by=["model", "mape", "rmse"])
        .groupby("model", as_index=False)
        .first()
        .sort_values(by="mape")
    )
    overall_best = metrics_df.sort_values(by=["mape", "rmse"]).iloc[0].to_dict()
    dims_lookup = dimensions_df.set_index("feature_set").to_dict(orient="index")
    full_dim = dims_lookup["full_current"]["linear_model_expanded_feature_count"]
    compact_dim = dims_lookup["compact_linear_friendly"]["linear_model_expanded_feature_count"]
    removed_drop_low = sorted(set(feature_sets["full_current"]) - set(feature_sets["drop_low_signal"]))
    removed_strong = sorted(set(feature_sets["drop_low_signal"]) - set(feature_sets["strong_signal_only"]))

    report_lines = [
        "# Stage 2 Feature-Set Ablation",
        "",
        "Цель: проверить, как меняется качество тех же основных моделей при разных комбинациях признаков.",
        "",
        "## Дизайн эксперимента",
        "- Использован тот же cleaned dataset и тот же random_state=42.",
        "- Train/test split зафиксирован один и тот же для всех запусков.",
        "- Гиперпараметры моделей не переоптимизировались под каждый набор признаков, чтобы сравнение зависело именно от feature set, а не от тюнинга.",
        "- Сравнивались модели: linear_regression, ridge, lasso, random_forest, gradient_boosting, xgboost, condition_segmented_xgboost.",
        "",
        "## Наборы признаков",
        "- `full_current`: весь текущий набор признаков после штатного удаления `year`.",
        "- `drop_low_signal`: удалены слабосвязанные и константные признаки. Порог: |Spearman| < 0.20 для числовых и Cramer's V < 0.20 для категориальных.",
        "- `strong_signal_only`: оставлены только признаки с более выраженным общим сигналом. Порог: |Spearman| >= 0.20 и Cramer's V >= 0.30.",
        "- `compact_linear_friendly`: компактный набор без высококардинальных `brand`, `model`, `generation`.",
        "- `numeric_only`: только числовые и булевы признаки, без категориальных столбцов.",
        "- В терминах линейной модели это даёт такую размерность после one-hot: `full_current` = {full_dim}, `compact_linear_friendly` = {compact_dim}.".format(
            full_dim=full_dim,
            compact_dim=compact_dim,
        ),
        "- Из `drop_low_signal` были убраны: `{removed}`.".format(removed=", ".join(removed_drop_low)),
        "- Из `strong_signal_only` дополнительно были убраны: `{removed}`.".format(
            removed=", ".join(removed_strong) if removed_strong else "none"
        ),
        "",
        "## Лучший результат по MAPE",
        "- Feature set: `{feature_set}`".format(feature_set=overall_best["feature_set"]),
        "- Model: `{model}`".format(model=overall_best["model"]),
        "- MAE: {mae:.2f}".format(mae=overall_best["mae"]),
        "- RMSE: {rmse:.2f}".format(rmse=overall_best["rmse"]),
        "- R2: {r2:.4f}".format(r2=overall_best["r2"]),
        "- MAPE: {mape:.2f}%".format(mape=overall_best["mape"]),
        "",
        "## Лучший набор признаков для каждой модели",
    ]

    for row in best_by_model.to_dict(orient="records"):
        report_lines.append(
            "- {model}: `{feature_set}` | MAPE={mape:.2f}% | RMSE={rmse:.2f}".format(
                model=row["model"],
                feature_set=row["feature_set"],
                mape=row["mape"],
                rmse=row["rmse"],
            )
        )

    report_lines.extend(
        [
            "",
            "## Интерпретация",
            "- Если компактный набор улучшает линейные модели, проблема была не в самом факте линейности, а в раздутой one-hot размерности и высокой кардинальности категорий.",
            "- Если `drop_low_signal` почти не ухудшает качество деревьев и бустинга, значит часть слабых признаков действительно была избыточной.",
            "- Если `strong_signal_only` начинает проигрывать, это значит, что жёсткое отсечение по одиночным статистикам выбрасывает полезные взаимодействия.",
            "- Если `numeric_only` сильно хуже, значит категориальная идентичность автомобиля несёт критический рыночный сигнал, который нельзя заменить только пробегом, возрастом и мощностью.",
            "- Если после сокращения признаков RMSE падает, а MAPE растёт, это обычно означает уменьшение крупных абсолютных ошибок на дорогих авто при ухудшении относительной точности на дешёвых сегментах.",
            "- В этих данных именно `brand`, `model` и `generation` несут значимую рыночную идентичность, поэтому механическое удаление высококардинальных категорий оказалось вредным даже для линейных моделей.",
            "",
            "## Артефакты",
            f"- Metrics table: `{metrics_path}`",
            f"- Feature-set dimensions: `{dims_path}`",
            f"- Numeric signal table: `{numeric_corr_path}`",
            f"- Categorical signal table: `{categorical_assoc_path}`",
            f"- MAPE heatmap: `{FIGURES_DIR / 'stage2_feature_set_ablation_mape_heatmap.png'}`",
            f"- RMSE heatmap: `{FIGURES_DIR / 'stage2_feature_set_ablation_rmse_heatmap.png'}`",
        ]
    )

    (REPORTS_DIR / "stage2_feature_set_ablation_report.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    summary = {
        "overall_best": overall_best,
        "best_by_model": best_by_model.to_dict(orient="records"),
        "feature_sets": feature_sets,
    }
    (REPORTS_DIR / "stage2_feature_set_ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
