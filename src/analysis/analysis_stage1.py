"""Stage 1 exploratory and statistical analysis."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.utils.constants import CATEGORICAL_FEATURES, FIGURES_DIR, NUMERIC_FEATURES, REPORTS_DIR
from src.utils.io_utils import ensure_dir, load_tabular

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Stage1Config:
    """Configuration for stage 1 analysis."""

    input_path: Path
    reports_dir: Path = REPORTS_DIR
    figures_dir: Path = FIGURES_DIR
    max_categories_for_group_test: int = 10
    min_group_size: int = 10


def _effect_strength_label(value: float, *, metric: str) -> str:
    """Map association magnitude to a coarse verbal interpretation."""
    absolute = abs(float(value))

    if metric == "spearman":
        if absolute < 0.10:
            return "negligible"
        if absolute < 0.30:
            return "weak"
        if absolute < 0.50:
            return "moderate"
        if absolute < 0.70:
            return "strong"
        return "very_strong"

    if metric == "cramers_v":
        if absolute < 0.10:
            return "negligible"
        if absolute < 0.30:
            return "weak"
        if absolute < 0.50:
            return "moderate"
        return "strong"

    raise ValueError(f"Unknown metric for effect interpretation: {metric}")


def _cramers_v(confusion_matrix: pd.DataFrame) -> float:
    """Compute Cramer's V with bias correction."""
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.to_numpy().sum()
    if n == 0:
        return np.nan

    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / max(1, (n - 1)))
    r_corr = r - ((r - 1) ** 2) / max(1, (n - 1))
    k_corr = k - ((k - 1) ** 2) / max(1, (n - 1))
    denom = min((k_corr - 1), (r_corr - 1))
    if denom <= 0:
        return np.nan
    return float(np.sqrt(phi2_corr / denom))


def _save_numeric_distribution_plots(df: pd.DataFrame, feature: str, figures_dir: Path) -> None:
    """Create histogram+box and QQ plot for numeric feature."""
    data = df[feature].dropna()
    if data.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(data, kde=True, ax=axes[0], color="#2a9d8f")
    axes[0].set_title(f"Histogram: {feature}")

    sns.boxplot(x=data, ax=axes[1], color="#e9c46a")
    axes[1].set_title(f"Boxplot: {feature}")

    fig.tight_layout()
    fig.savefig(figures_dir / f"normality_hist_box_{feature}.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(6, 6))
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f"QQ-plot: {feature}")
    fig.tight_layout()
    fig.savefig(figures_dir / f"normality_qq_{feature}.png", dpi=150)
    plt.close(fig)


def _normality_analysis(df: pd.DataFrame, numeric_cols: list[str], figures_dir: Path) -> pd.DataFrame:
    """Run normality diagnostics and tests for numeric columns."""
    rows: list[dict[str, object]] = []

    for feature in numeric_cols:
        if feature not in df.columns:
            continue

        data = df[feature].dropna()
        if len(data) < 8:
            continue

        _save_numeric_distribution_plots(df, feature, figures_dir)

        test_data = data.sample(n=min(5000, len(data)), random_state=42)
        statistic, p_value = stats.shapiro(test_data)

        rows.append(
            {
                "feature": feature,
                "n": int(len(data)),
                "shapiro_stat": float(statistic),
                "shapiro_pvalue": float(p_value),
                "is_normal_at_0_05": bool(p_value >= 0.05),
                "skewness": float(data.skew()),
                "kurtosis": float(data.kurtosis()),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature",
                "n",
                "shapiro_stat",
                "shapiro_pvalue",
                "is_normal_at_0_05",
                "skewness",
                "kurtosis",
            ]
        )

    return pd.DataFrame(rows).sort_values(by="shapiro_pvalue", ascending=True)


def _group_price_tests(
    df: pd.DataFrame,
    categorical_cols: list[str],
    max_categories: int,
    min_group_size: int,
    figures_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare price distribution across categories."""
    test_rows: list[dict[str, object]] = []
    posthoc_rows: list[dict[str, object]] = []

    price = df["price"].dropna()
    is_price_normal = False
    if len(price) >= 8:
        _, pval = stats.shapiro(price.sample(n=min(5000, len(price)), random_state=42))
        is_price_normal = pval >= 0.05

    for feature in categorical_cols:
        if feature not in df.columns:
            continue

        subset = df[[feature, "price"]].dropna()
        if subset.empty:
            continue

        counts = subset[feature].value_counts()
        valid_levels = counts[counts >= min_group_size].index.tolist()
        if not (2 <= len(valid_levels) <= max_categories):
            continue

        subset = subset[subset[feature].isin(valid_levels)]
        groups = [g["price"].to_numpy() for _, g in subset.groupby(feature)]

        if is_price_normal and all(len(g) >= 20 for g in groups):
            stat, pvalue = stats.f_oneway(*groups)
            test_name = "anova"
        else:
            stat, pvalue = stats.kruskal(*groups)
            test_name = "kruskal"

        test_rows.append(
            {
                "feature": feature,
                "test": test_name,
                "groups": len(groups),
                "statistic": float(stat),
                "pvalue": float(pvalue),
                "significant_at_0_05": bool(pvalue < 0.05),
            }
        )

        fig = plt.figure(figsize=(max(7, len(valid_levels) * 1.0), 4.5))
        sns.boxplot(
            data=subset,
            x=feature,
            y="price",
            hue=feature,
            showfliers=False,
            palette="viridis",
            legend=False,
        )
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Price by {feature}")
        plt.tight_layout()
        fig.savefig(figures_dir / f"price_by_{feature}.png", dpi=150)
        plt.close(fig)

        if pvalue < 0.05 and len(valid_levels) <= 8:
            pair_results: list[tuple[str, str, float]] = []
            combinations = list(itertools.combinations(valid_levels, 2))
            for left, right in combinations:
                left_vals = subset.loc[subset[feature] == left, "price"]
                right_vals = subset.loc[subset[feature] == right, "price"]
                _, p = stats.mannwhitneyu(left_vals, right_vals, alternative="two-sided")
                pair_results.append((left, right, p))

            bonf = max(1, len(pair_results))
            for left, right, p in pair_results:
                adjusted = min(1.0, p * bonf)
                posthoc_rows.append(
                    {
                        "feature": feature,
                        "group_left": left,
                        "group_right": right,
                        "pvalue_raw": float(p),
                        "pvalue_bonferroni": float(adjusted),
                        "significant_at_0_05": bool(adjusted < 0.05),
                    }
                )

    return pd.DataFrame(test_rows), pd.DataFrame(posthoc_rows)


def _factor_influence(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Estimate factor influence on price with multiple methods."""
    corr_rows: list[dict[str, object]] = []
    for feature in numeric_cols:
        if feature not in df.columns or feature == "price":
            continue
        pair = df[[feature, "price"]].dropna()
        if len(pair) < 10:
            continue
        pearson_stat, pearson_pvalue = stats.pearsonr(pair[feature], pair["price"])
        spearman_stat, spearman_pvalue = stats.spearmanr(pair[feature], pair["price"])
        corr_rows.append(
            {
                "feature": feature,
                "pearson_with_price": float(pearson_stat),
                "pearson_pvalue": float(pearson_pvalue),
                "pearson_significant_at_0_05": bool(pearson_pvalue < 0.05),
                "spearman_with_price": float(spearman_stat),
                "spearman_pvalue": float(spearman_pvalue),
                "spearman_significant_at_0_05": bool(spearman_pvalue < 0.05),
                "spearman_strength": _effect_strength_label(float(spearman_stat), metric="spearman"),
                "abs_spearman": abs(float(spearman_stat)),
            }
        )

    price_bins = pd.qcut(df["price"], q=4, labels=["q1", "q2", "q3", "q4"], duplicates="drop")
    cat_assoc_rows: list[dict[str, object]] = []
    for feature in categorical_cols:
        if feature not in df.columns:
            continue
        subset = pd.DataFrame({feature: df[feature], "price_bin": price_bins}).dropna()
        if subset.empty or subset[feature].nunique() < 2 or subset["price_bin"].nunique() < 2:
            continue

        contingency = pd.crosstab(subset[feature], subset["price_bin"])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            continue

        chi2, pvalue, _, _ = stats.chi2_contingency(contingency)
        cramers_v = _cramers_v(contingency)
        cat_assoc_rows.append(
            {
                "feature": feature,
                "chi2": float(chi2),
                "pvalue": float(pvalue),
                "chi2_pvalue": float(pvalue),
                "cramers_v": cramers_v,
                "cramers_v_strength": _effect_strength_label(cramers_v, metric="cramers_v"),
                "significant_at_0_05": bool(pvalue < 0.05),
                "chi2_significant_at_0_05": bool(pvalue < 0.05),
            }
        )

    model_features = [
        col
        for col in numeric_cols + categorical_cols
        if col in df.columns and col != "price"
    ]
    model_frame = df[model_features + ["price"]].dropna(subset=["price"])

    X_num = model_frame[[col for col in numeric_cols if col in model_features]].copy()
    X_num = X_num.dropna(axis=1, how="all")
    X_num = X_num.fillna(X_num.median(numeric_only=True)).fillna(0)

    cat_cols_used = [col for col in categorical_cols if col in model_features]
    X_cat = pd.get_dummies(
        model_frame[cat_cols_used].fillna("unknown"),
        prefix_sep="=",
        dtype=int,
    )

    X = pd.concat([X_num, X_cat], axis=1)
    X = X.dropna(axis=1, how="all")
    y = model_frame["price"].astype(float)

    mi_df = pd.DataFrame(columns=["feature", "mutual_information"])
    importance_df = pd.DataFrame(columns=["feature", "importance"])

    if len(X) >= 30 and X.shape[1] > 1:
        mi_values = mutual_info_regression(X, y, random_state=42)
        mi_df = pd.DataFrame({"feature": X.columns, "mutual_information": mi_values}).sort_values(
            by="mutual_information", ascending=False
        )

        model = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)
        model.fit(X, y)
        importance_df = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_}).sort_values(
            by="importance", ascending=False
        )

    corr_df = pd.DataFrame(corr_rows)
    if not corr_df.empty:
        corr_df = corr_df.sort_values(by="abs_spearman", ascending=False)

    cat_assoc_df = pd.DataFrame(cat_assoc_rows)
    if not cat_assoc_df.empty:
        cat_assoc_df = cat_assoc_df.sort_values(by="cramers_v", ascending=False)
    return corr_df, cat_assoc_df, mi_df, importance_df


def _independence_analysis(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    figures_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Analyze dependencies and potential multicollinearity among predictors."""
    num_cols = [col for col in numeric_cols if col in df.columns and col != "price" and df[col].notna().any()]

    corr_matrix = pd.DataFrame()
    if len(num_cols) >= 2:
        corr_matrix = df[num_cols].corr(method="spearman")
        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, cmap="RdYlBu_r", center=0, annot=False)
        plt.title("Numeric Feature Correlation (Spearman)")
        plt.tight_layout()
        fig.savefig(figures_dir / "numeric_correlation_heatmap.png", dpi=150)
        plt.close(fig)

    vif_df = pd.DataFrame(columns=["feature", "vif"])
    if len(num_cols) >= 2:
        vif_data = df[num_cols].copy()
        vif_data = vif_data.dropna(axis=1, how="all")
        vif_data = vif_data.fillna(vif_data.median(numeric_only=True)).fillna(0)
        vif_data = vif_data.astype(float)
        if vif_data.shape[1] >= 2:
            vif_rows: list[dict[str, float | str]] = []
            for i, feature in enumerate(vif_data.columns):
                vif_rows.append(
                    {
                        "feature": feature,
                        "vif": float(variance_inflation_factor(vif_data.values, i)),
                    }
                )
            vif_df = pd.DataFrame(vif_rows).sort_values(by="vif", ascending=False)

    cat_rows: list[dict[str, object]] = []
    cat_cols = [col for col in categorical_cols if col in df.columns]
    for left, right in itertools.combinations(cat_cols, 2):
        subset = df[[left, right]].dropna()
        if subset.empty:
            continue

        # Skip highly sparse combinations.
        if subset[left].nunique() > 15 or subset[right].nunique() > 15:
            continue

        contingency = pd.crosstab(subset[left], subset[right])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            continue

        chi2, pvalue, _, _ = stats.chi2_contingency(contingency)
        cat_rows.append(
            {
                "feature_left": left,
                "feature_right": right,
                "chi2": float(chi2),
                "pvalue": float(pvalue),
                "cramers_v": _cramers_v(contingency),
                "significant_at_0_05": bool(pvalue < 0.05),
            }
        )

    cat_independence_df = pd.DataFrame(cat_rows)
    if not cat_independence_df.empty:
        cat_independence_df = cat_independence_df.sort_values(by="cramers_v", ascending=False)
    return corr_matrix, vif_df, cat_independence_df


def _top_rows(df: pd.DataFrame, by: str, n: int = 5) -> list[dict[str, object]]:
    if df.empty or by not in df.columns:
        return []
    return df.sort_values(by=by, ascending=False).head(n).to_dict(orient="records")


def run_stage1_analysis(config: Stage1Config) -> None:
    """Execute stage-1 analysis pipeline and save reports/figures."""
    ensure_dir(config.reports_dir)
    ensure_dir(config.figures_dir)

    LOGGER.info("Loading cleaned dataset from %s", config.input_path)
    df = load_tabular(config.input_path)
    if df.empty:
        raise ValueError("Input dataset is empty")

    numeric_cols = [col for col in NUMERIC_FEATURES + ["age", "price_log1p"] if col in df.columns]
    categorical_cols = [col for col in CATEGORICAL_FEATURES if col in df.columns]

    LOGGER.info("Running normality checks...")
    normality_df = _normality_analysis(df, numeric_cols=numeric_cols, figures_dir=config.figures_dir)
    if normality_df.empty:
        normality_df = pd.DataFrame(
            columns=[
                "feature",
                "n",
                "shapiro_stat",
                "shapiro_pvalue",
                "is_normal_at_0_05",
                "skewness",
                "kurtosis",
            ]
        )
    normality_df.to_csv(config.reports_dir / "stage1_normality.csv", index=False)

    LOGGER.info("Running group homogeneity tests...")
    group_tests_df, posthoc_df = _group_price_tests(
        df,
        categorical_cols=categorical_cols,
        max_categories=config.max_categories_for_group_test,
        min_group_size=config.min_group_size,
        figures_dir=config.figures_dir,
    )
    group_tests_df.to_csv(config.reports_dir / "stage1_group_tests.csv", index=False)
    posthoc_df.to_csv(config.reports_dir / "stage1_group_posthoc.csv", index=False)

    LOGGER.info("Running factor influence analysis...")
    corr_df, cat_assoc_df, mi_df, importance_df = _factor_influence(
        df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )
    corr_df.to_csv(config.reports_dir / "stage1_numeric_correlations.csv", index=False)
    cat_assoc_df.to_csv(config.reports_dir / "stage1_price_bin_categorical_assoc.csv", index=False)
    mi_df.to_csv(config.reports_dir / "stage1_mutual_information.csv", index=False)
    importance_df.to_csv(config.reports_dir / "stage1_feature_importance.csv", index=False)

    if not importance_df.empty:
        fig = plt.figure(figsize=(9, 5))
        top_imp = importance_df.head(20)
        sns.barplot(data=top_imp, y="feature", x="importance", hue="feature", palette="crest", legend=False)
        plt.title("Top-20 Feature Importances (RandomForest)")
        plt.tight_layout()
        fig.savefig(config.figures_dir / "stage1_top_feature_importance.png", dpi=150)
        plt.close(fig)

    LOGGER.info("Running independence and multicollinearity checks...")
    corr_matrix_df, vif_df, cat_independence_df = _independence_analysis(
        df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        figures_dir=config.figures_dir,
    )

    if not corr_matrix_df.empty:
        corr_matrix_df.to_csv(config.reports_dir / "stage1_numeric_corr_matrix.csv")
    vif_df.to_csv(config.reports_dir / "stage1_vif.csv", index=False)
    cat_independence_df.to_csv(config.reports_dir / "stage1_categorical_independence.csv", index=False)

    report_path = config.reports_dir / "stage1_report.md"
    top_numeric = _top_rows(corr_df, by="abs_spearman", n=5)
    top_categorical = _top_rows(cat_assoc_df, by="cramers_v", n=5)
    top_mi = _top_rows(mi_df, by="mutual_information", n=10)
    high_vif = vif_df[vif_df["vif"] >= 5].to_dict(orient="records") if not vif_df.empty else []

    significant_group_tests = int((group_tests_df.get("significant_at_0_05", pd.Series(dtype=bool)) == True).sum())

    lines = [
        "# Stage 1 Analysis Report",
        "",
        f"Rows analyzed: **{len(df)}**",
        f"Numeric features analyzed: **{len(numeric_cols)}**",
        f"Categorical features analyzed: **{len(categorical_cols)}**",
        "",
        "## Normality",
        f"- Features tested: {len(normality_df)}",
        f"- Approx. normal at p>=0.05: {(normality_df['is_normal_at_0_05'] == True).sum() if not normality_df.empty else 0}",
        "",
        "## Homogeneity of Price by Categories",
        f"- Group tests performed: {len(group_tests_df)}",
        f"- Significant tests (p<0.05): {significant_group_tests}",
        "",
        "## Top Numeric Associations With Price (Spearman)",
    ]

    for row in top_numeric:
        lines.append(f"- {row['feature']}: spearman={row['spearman_with_price']:.3f}")

    lines.append("")
    lines.append("## Top Categorical Associations With Price Quartile (Cramer's V)")
    for row in top_categorical:
        lines.append(f"- {row['feature']}: Cramer's V={row['cramers_v']:.3f}, p={row['pvalue']:.3g}")

    lines.append("")
    lines.append("## Top Mutual Information Features")
    for row in top_mi:
        lines.append(f"- {row['feature']}: MI={row['mutual_information']:.4f}")

    lines.append("")
    lines.append("## Multicollinearity (VIF >= 5)")
    if high_vif:
        for row in high_vif:
            lines.append(f"- {row['feature']}: VIF={row['vif']:.2f}")
    else:
        lines.append("- No strong multicollinearity found by VIF>=5 threshold.")

    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- Reports directory: `{config.reports_dir}`")
    lines.append(f"- Figures directory: `{config.figures_dir}`")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Stage 1 analysis completed. Report: %s", report_path)
