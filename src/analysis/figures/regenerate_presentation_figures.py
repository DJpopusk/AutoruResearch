from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess


BASE = Path(__file__).resolve().parents[3]
FIG = BASE / "figures"
REPORTS = BASE / "reports"
DATA = BASE / "data/processed/cleaned_dataset.parquet"

TITLE = "#162D50"
ACCENT = "#2F5597"
ORANGE = "#C55A11"
GREEN = "#70AD47"
TEAL = "#2B7A78"
RED = "#A61C3C"
GRAY = "#5F6368"
LIGHT = "#F3F7FC"
LIGHT2 = "#FCF4EC"
LIGHT3 = "#EFF7EF"
GRID = "#D8E1EB"


def setup_theme() -> None:
    sns.set_theme(style="whitegrid")
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D0D7DE",
            "axes.labelcolor": TITLE,
            "axes.titlecolor": TITLE,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.color": TITLE,
            "ytick.color": TITLE,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "legend.frameon": False,
        }
    )


def mln_formatter(x: float, pos: int) -> str:
    return f"{x / 1_000_000:.1f}"


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def clip_series(series: pd.Series, upper_q: float = 0.995) -> tuple[pd.Series, float]:
    upper = float(series.quantile(upper_q))
    return series.clip(upper=upper), upper


def add_source_note(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.99,
        -0.18,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color=GRAY,
    )


def add_corner_tag(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color=TITLE,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": GRID, "boxstyle": "round,pad=0.28"},
    )


def load_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    df = pd.read_parquet(DATA).copy()
    tables = {
        "price_bands": pd.read_csv(REPORTS / "segment_price_band_summary.csv"),
        "numeric_corr": pd.read_csv(REPORTS / "stage1_numeric_correlations.csv"),
        "categorical_assoc": pd.read_csv(REPORTS / "stage1_price_bin_categorical_assoc.csv"),
        "categorical_matrix": pd.read_csv(REPORTS / "stage1_categorical_cramers_v_matrix.csv").rename(columns={"Unnamed: 0": "feature"}),
        "mi": pd.read_csv(REPORTS / "stage1_mutual_information.csv"),
        "corr_matrix": pd.read_csv(REPORTS / "stage1_numeric_corr_matrix.csv").rename(columns={"Unnamed: 0": "feature"}),
        "missing": pd.read_csv(REPORTS / "overview_missingness.csv"),
        "price_segment_metrics": pd.read_csv(REPORTS / "stage2_metrics_by_price_segment.csv"),
        "condition_metrics": pd.read_csv(REPORTS / "stage2_metrics_by_condition.csv"),
        "segmented_numeric": pd.read_csv(REPORTS / "stage1_segmented_numeric_correlations.csv"),
        "complete_case_models": pd.read_csv(REPORTS / "stage2_complete_case_model_metrics.csv"),
        "linear_reduced": pd.read_csv(REPORTS / "stage2_linear_reduced_feature_metrics.csv"),
        "compact11_full_vs_complete": pd.read_csv(REPORTS / "stage2_compact11_full_vs_complete_metrics.csv"),
    }
    return df, tables


def make_condition_counts(df: pd.DataFrame) -> None:
    counts = df["condition"].value_counts().reindex(["new", "used"]).fillna(0).astype(int)
    total = counts.sum()
    labels = ["Новые", "С пробегом"]
    colors = [ACCENT, ORANGE]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    bars = ax.bar(labels, counts.values, color=colors, width=0.58)
    ax.set_ylabel("Количество объявлений")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    for bar, count in zip(bars, counts.values):
        pct = 100 * count / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.015,
            f"{count}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            color=TITLE,
            fontweight="bold",
        )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    save(fig, FIG / "condition_counts_presentation_ru.png")


def make_price_bands(tables: dict[str, pd.DataFrame]) -> None:
    bands = tables["price_bands"].copy()
    bands["median_mln"] = bands["median"] / 1_000_000
    colors = [GREEN, ACCENT, ORANGE]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    bars = ax.bar(["Дешёвый", "Средний", "Дорогой"], bands["count"], color=colors, width=0.58)
    ax.set_ylabel("Количество объявлений")
    for bar, count in zip(bars, bands["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 25, f"{int(count)}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    bars = ax.barh(["Дешёвый", "Средний", "Дорогой"], bands["median_mln"], color=colors)
    ax.set_xlabel("млн руб.")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.1f}"))
    for bar, value in zip(bars, bands["median_mln"]):
        ax.text(value + 0.05, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=11, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    save(fig, FIG / "presentation_price_bands.png")


def make_price_distribution(df: pd.DataFrame) -> None:
    price = df["price"].astype(float)
    clipped, upper = clip_series(price, 0.99)

    fig, ax_hist = plt.subplots(figsize=(10.5, 5.4))

    sns.histplot(clipped / 1_000_000, bins=30, color=ACCENT, alpha=0.9, edgecolor="white", ax=ax_hist)
    median = price.median() / 1_000_000
    mean = price.mean() / 1_000_000
    p90 = price.quantile(0.90) / 1_000_000
    ax_hist.axvline(median, color=ORANGE, linestyle="--", linewidth=2, label=f"Медиана: {median:.2f}")
    ax_hist.axvline(mean, color=GREEN, linestyle="-.", linewidth=2, label=f"Среднее: {mean:.2f}")
    ax_hist.axvline(p90, color=TEAL, linestyle=":", linewidth=2, label=f"P90: {p90:.2f}")
    ax_hist.set_ylabel("Количество объявлений")
    ax_hist.set_xlabel("Цена, млн руб.")
    ax_hist.legend(loc="upper right")
    ax_hist.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.1f}"))
    trimmed_share = (price > upper).mean() * 100
    ax_hist.text(
        0.02,
        0.95,
        "Сильный правый хвост,\nпоэтому ось ограничена",
        transform=ax_hist.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color=TITLE,
        bbox={"facecolor": LIGHT, "edgecolor": GRID, "boxstyle": "round,pad=0.4"},
    )
    add_source_note(ax_hist, f"Ось ограничена по p99 = {upper/1_000_000:.2f} млн руб.; вне графика {trimmed_share:.2f}% объявлений")

    for spine in ["top", "right"]:
        ax_hist.spines[spine].set_visible(False)

    save(fig, FIG / "normality_hist_box_price.png")


def make_price_qq(df: pd.DataFrame) -> None:
    sample = df["price"].astype(float).sample(min(3000, len(df)), random_state=42)
    z = (sample - sample.mean()) / sample.std(ddof=0)
    (osm, osr), (slope, intercept, _) = stats.probplot(z, dist="norm")

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.scatter(osm, osr, s=14, color=ACCENT, alpha=0.55)
    xline = np.linspace(min(osm), max(osm), 100)
    ax.plot(xline, slope * xline + intercept, color=ORANGE, linewidth=2.2)
    ax.set_xlabel("Теоретические квантили")
    ax.set_ylabel("Наблюдаемые квантили")
    ax.text(0.03, 0.95, "Сильное отклонение от прямой\n=> нормальности нет", transform=ax.transAxes, ha="left", va="top", fontsize=11, color=RED)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    save(fig, FIG / "normality_qq_price.png")


def make_condition_boxplot(df: pd.DataFrame) -> None:
    data = df[["condition", "price"]].copy()
    upper = float(data["price"].quantile(0.99))
    data["price_mln"] = data["price"].clip(upper=upper) / 1_000_000
    order = ["new", "used"]
    palette = {"new": ACCENT, "used": ORANGE}

    fig, ax = plt.subplots(figsize=(7.0, 5.1))
    sns.boxplot(data=data, x="condition", y="price_mln", order=order, palette=palette, showfliers=False, width=0.55, linewidth=1.5, ax=ax)
    ax.set_xticklabels(["Новые", "С пробегом"])
    ax.set_xlabel("")
    ax.set_ylabel("Цена, млн руб.")
    medians = data.groupby("condition")["price"].median().reindex(order) / 1_000_000
    for i, (_, value) in enumerate(medians.items()):
        ax.text(i, value + 0.12, f"медиана {value:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold", color=TITLE)
    add_source_note(ax, f"Boxplot без выбросов; ось ограничена по p99 = {upper/1_000_000:.2f} млн руб.")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    save(fig, FIG / "condition_price_boxplot_presentation_ru.png")


def make_condition_histograms(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    for ax, condition, color, label in zip(
        axes,
        ["new", "used"],
        [ACCENT, ORANGE],
        ["Новые автомобили", "Автомобили с пробегом"],
    ):
        subset = df.loc[df["condition"] == condition, "price"].astype(float)
        clipped, upper = clip_series(subset, 0.995)
        sns.histplot(clipped / 1_000_000, bins=28, color=color, alpha=0.9, edgecolor="white", ax=ax)
        ax.axvline(subset.median() / 1_000_000, color=GREEN, linestyle="--", linewidth=2)
        ax.set_xlabel("Цена, млн руб.")
        add_source_note(ax, f"ось до p99.5 = {upper/1_000_000:.2f}")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Количество объявлений")
    axes[1].set_ylabel("")
    save(fig, FIG / "condition_price_histograms_presentation_ru.png")


def make_dataset_quality(tables: dict[str, pd.DataFrame]) -> None:
    missing = tables["missing"].sort_values("missing_pct", ascending=False).head(6).copy()
    missing["feature_ru"] = missing["feature"]
    cards = (
        tables["missing"]
        .query("dtype == 'object'")
        .sort_values("n_unique_non_null", ascending=False)
        .head(6)
        .copy()
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})

    ax = axes[0]
    bars = ax.barh(missing["feature_ru"].iloc[::-1], missing["missing_pct"].iloc[::-1], color=ORANGE, alpha=0.9)
    ax.set_xlabel("% пропусков")
    for bar, value in zip(bars, missing["missing_pct"].iloc[::-1], strict=False):
        ax.text(value + 1.2, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%", va="center", fontsize=10, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    add_corner_tag(ax, "Пропуски")

    ax = axes[1]
    bars = ax.barh(cards["feature"].iloc[::-1], cards["n_unique_non_null"].iloc[::-1], color=ACCENT, alpha=0.9)
    ax.set_xlabel("Число уникальных значений")
    for bar, value in zip(bars, cards["n_unique_non_null"].iloc[::-1], strict=False):
        ax.text(value + max(cards["n_unique_non_null"]) * 0.02, bar.get_y() + bar.get_height() / 2, f"{int(value)}", va="center", fontsize=10, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    add_corner_tag(ax, "Кардинальность категорий")

    save(fig, FIG / "presentation_dataset_quality.png")


def make_price_vs_log_distribution(df: pd.DataFrame) -> None:
    price = df["price"].astype(float)
    log_price = df["price_log1p"].astype(float)
    price_clip, price_upper = clip_series(price, 0.99)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9))

    ax = axes[0]
    sns.histplot(price_clip / 1_000_000, bins=32, stat="density", kde=True, color=ACCENT, alpha=0.85, edgecolor="white", ax=ax)
    ax.set_xlabel("Цена, млн руб.")
    ax.set_ylabel("Плотность")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.1f}"))
    ax.text(
        0.03,
        0.94,
        f"skew = {stats.skew(price, bias=False):.2f}\nkurt = {stats.kurtosis(price, fisher=False, bias=False):.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color=TITLE,
        bbox={"facecolor": "white", "edgecolor": GRID, "boxstyle": "round,pad=0.28"},
    )
    add_corner_tag(ax, "Исходная шкала")
    add_source_note(ax, f"Ось ограничена по p99 = {price_upper/1_000_000:.2f} млн руб.")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    sns.histplot(log_price, bins=32, stat="density", kde=True, color=GREEN, alpha=0.85, edgecolor="white", ax=ax)
    ax.set_xlabel("log(1 + price)")
    ax.set_ylabel("")
    ax.text(
        0.03,
        0.94,
        f"skew = {stats.skew(log_price, bias=False):.2f}\nkurt = {stats.kurtosis(log_price, fisher=False, bias=False):.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color=TITLE,
        bbox={"facecolor": "white", "edgecolor": GRID, "boxstyle": "round,pad=0.28"},
    )
    add_corner_tag(ax, "Логарифмическая шкала")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    save(fig, FIG / "presentation_price_vs_log_distribution.png")


def _qq_axes(ax: plt.Axes, series: pd.Series, label: str) -> None:
    sample = series.sample(min(3000, len(series)), random_state=42)
    z = (sample - sample.mean()) / sample.std(ddof=0)
    (osm, osr), (slope, intercept, _) = stats.probplot(z, dist="norm")
    ax.scatter(osm, osr, s=14, color=ACCENT, alpha=0.55)
    xline = np.linspace(min(osm), max(osm), 100)
    ax.plot(xline, slope * xline + intercept, color=ORANGE, linewidth=2.2)
    ax.set_xlabel("Теоретические квантили")
    ax.set_ylabel("Наблюдаемые квантили")
    add_corner_tag(ax, label)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def make_qq_raw_vs_log(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.8))
    _qq_axes(axes[0], df["price"].astype(float), "price")
    _qq_axes(axes[1], df["price_log1p"].astype(float), "log(price)")
    axes[1].set_ylabel("")
    save(fig, FIG / "presentation_qq_raw_vs_log.png")


def make_condition_violin(df: pd.DataFrame) -> None:
    work = df[["condition", "price_log1p"]].dropna().copy()
    order = ["new", "used"]
    palette = {"new": ACCENT, "used": ORANGE}

    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    # 1. violin как форма распределения
    sns.violinplot(
        data=work,
        x="condition",
        y="price_log1p",
        order=order,
        palette=palette,
        inner=None,
        cut=0,
        linewidth=1.4,
        saturation=1,
        ax=ax
    )

    # делаем violin чуть прозрачнее
    for coll in ax.collections:
        try:
            coll.set_alpha(0.45)
        except Exception:
            pass

    # 2. boxplot поверх violin
    sns.boxplot(
        data=work,
        x="condition",
        y="price_log1p",
        order=order,
        width=0.22,
        showfliers=False,
        showcaps=True,
        boxprops={
            "facecolor": "white",
            "edgecolor": TITLE,
            "linewidth": 1.5,
            "zorder": 4,
        },
        whiskerprops={
            "color": TITLE,
            "linewidth": 1.4,
        },
        capprops={
            "color": TITLE,
            "linewidth": 1.4,
        },
        medianprops={
            "color": TITLE,
            "linewidth": 1.8,
        },
        ax=ax
    )

    # 3. точки поверх всего
    sns.stripplot(
        data=work.sample(min(len(work), 800), random_state=42),
        x="condition",
        y="price_log1p",
        order=order,
        palette=palette,
        jitter=0.16,
        size=3.2,
        alpha=0.45,
        dodge=False,
        linewidth=0,
        ax=ax,
        zorder=5
    )

    ax.set_xticklabels(["Новые", "С пробегом"])
    ax.set_xlabel("")
    ax.set_ylabel("log(1 + price)")
    ax.grid(axis="y", alpha=0.35)
    ax.grid(axis="x", visible=False)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    save(fig, FIG / "presentation_condition_violin.png")

def _top_levels(df: pd.DataFrame, column: str, top_n: int, exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    counts = df[column].value_counts()
    levels = [x for x in counts.index.tolist() if x not in exclude]
    return levels[:top_n]


def make_body_type_boxplot(df: pd.DataFrame) -> None:
    levels = _top_levels(df, "body_type", 6, exclude={"unknown"})
    work = df.loc[df["body_type"].isin(levels), ["body_type", "price_log1p"]].dropna().copy()
    medians = work.groupby("body_type")["price_log1p"].median().sort_values()
    order = medians.index.tolist()
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    sns.boxplot(data=work, x="body_type", y="price_log1p", order=order, color=ACCENT, showfliers=False, width=0.6, linewidth=1.3, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("log(1 + price)")
    ax.tick_params(axis="x", rotation=18)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    save(fig, FIG / "presentation_body_type_boxplot.png")


def make_brand_boxplot(df: pd.DataFrame) -> None:
    levels = _top_levels(df, "brand", 8)
    work = df.loc[df["brand"].isin(levels), ["brand", "price_log1p"]].dropna().copy()
    medians = work.groupby("brand")["price_log1p"].median().sort_values()
    order = medians.index.tolist()
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    sns.boxplot(data=work, y="brand", x="price_log1p", order=order, color=ORANGE, showfliers=False, width=0.6, linewidth=1.3, ax=ax)
    ax.set_xlabel("log(1 + price)")
    ax.set_ylabel("")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    save(fig, FIG / "presentation_brand_boxplot.png")


def make_small_category_boxplots(df: pd.DataFrame) -> None:
    cols = [
        ("transmission", ["automatic", "robot", "manual", "cvt"], "Трансмиссия"),
        ("fuel_type", ["petrol", "diesel", "hybrid", "electric"], "Топливо"),
        ("drive_type", ["awd", "unknown"], "Привод"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.8))
    colors = [ACCENT, GREEN, ORANGE]
    for ax, (col, order, tag), color in zip(axes, cols, colors, strict=False):
        work = df.loc[df[col].isin(order), [col, "price_log1p"]].dropna().copy()
        sns.boxplot(data=work, x=col, y="price_log1p", order=order, color=color, showfliers=False, width=0.58, linewidth=1.2, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("log(1 + price)" if col == "transmission" else "")
        ax.tick_params(axis="x", rotation=15)
        add_corner_tag(ax, tag)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    save(fig, FIG / "presentation_small_category_boxplots.png")


def make_smooth_vs_jump_example(df: pd.DataFrame) -> None:
    work_num = df[["age", "price_log1p"]].dropna().copy()
    x = work_num["age"].astype(float).clip(upper=float(work_num["age"].quantile(0.99)))
    y = work_num["price_log1p"].astype(float)
    sample = pd.DataFrame({"x": x, "y": y}).sample(min(2200, len(work_num)), random_state=42)
    smooth = lowess(y, x, frac=0.18, return_sorted=True)

    top_brands = df["brand"].value_counts().head(8).index.tolist()
    work_cat = df.loc[df["brand"].isin(top_brands), ["brand", "price_log1p"]].dropna().copy()
    brand_order = work_cat.groupby("brand")["price_log1p"].median().sort_values().index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9), gridspec_kw={"wspace": 0.3})

    ax = axes[0]
    ax.scatter(sample["x"], sample["y"], s=15, alpha=0.16, color=ACCENT, edgecolors="none")
    ax.plot(smooth[:, 0], smooth[:, 1], color=ORANGE, linewidth=2.8)
    ax.set_xlabel("Возраст, лет")
    ax.set_ylabel("log(1 + price)")
    add_corner_tag(ax, "Числовой признак: плавная зависимость")
    add_source_note(ax, "LOWESS показывает непрерывный тренд без дискретных скачков между группами")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    sns.boxplot(
        data=work_cat,
        y="brand",
        x="price_log1p",
        order=brand_order,
        color=GREEN,
        showfliers=False,
        width=0.6,
        linewidth=1.2,
        ax=ax,
    )
    medians = work_cat.groupby("brand")["price_log1p"].median().reindex(brand_order)
    for idx, (_, value) in enumerate(medians.items()):
        ax.text(value + 0.03, idx, f"{value:.2f}", va="center", fontsize=10, fontweight="bold", color=TITLE)
    ax.set_xlabel("log(1 + price)")
    ax.set_ylabel("")
    add_corner_tag(ax, "Категориальный признак: скачки между группами")
    add_source_note(ax, "Группы задают дискретные ценовые уровни, а не непрерывный градиент по оси x")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    save(fig, FIG / "presentation_smooth_vs_jump_example.png")


def _dual_dependence_axes(ax: plt.Axes, x: pd.Series, y: pd.Series, x_label: str, y_label: str, panel_label: str, lowess_frac: float) -> None:
    sample = pd.DataFrame({"x": x, "y": y}).dropna()
    sample = sample.sample(min(2200, len(sample)), random_state=42)
    ax.scatter(sample["x"], sample["y"], s=14, alpha=0.17, color=ACCENT, edgecolors="none")
    smooth = lowess(y, x, frac=lowess_frac, return_sorted=True)
    ax.plot(smooth[:, 0], smooth[:, 1], color=ORANGE, linewidth=2.6)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    add_corner_tag(ax, panel_label)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def make_dual_dependency_plot(
    df: pd.DataFrame,
    *,
    feature: str,
    out_name: str,
    x_label: str,
    x_transform=lambda s: s,
    x_upper_q: float = 0.99,
    raw_upper_q: float = 0.99,
    lowess_frac: float = 0.18,
) -> None:
    work = df[[feature, "price", "price_log1p"]].dropna().copy()
    x = x_transform(work[feature].astype(float))
    x_upper = float(x.quantile(x_upper_q))
    x = x.clip(upper=x_upper)

    raw = (work["price"].astype(float) / 1_000_000).clip(upper=float((work["price"] / 1_000_000).quantile(raw_upper_q)))
    log_y = work["price_log1p"].astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9))
    _dual_dependence_axes(axes[0], x, raw, x_label, "Цена, млн руб.", "price", lowess_frac)
    _dual_dependence_axes(axes[1], x, log_y, x_label, "log(1 + price)", "log(price)", lowess_frac)
    add_source_note(axes[1], "Линия LOWESS показывает нелинейный локальный тренд")
    save(fig, FIG / out_name)


def make_presentation_numeric_heatmap(df: pd.DataFrame) -> None:
    cols = ["price_log1p", "age", "year", "mileage", "engine_volume", "engine_power_hp"]
    corr = df[cols].corr(method="spearman", numeric_only=True)
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    sns.heatmap(
        corr,
        cmap=sns.diverging_palette(245, 15, as_cmap=True),
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.7,
        cbar_kws={"shrink": 0.82},
        ax=ax,
    )
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    save(fig, FIG / "presentation_numeric_corr_heatmap.png")


def make_presentation_categorical_heatmap(tables: dict[str, pd.DataFrame]) -> None:
    raw = tables["categorical_matrix"].copy()
    matrix = raw.set_index("feature")
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    cols = ["body_type", "color", "fuel_type", "transmission", "drive_type", "condition", "seller_type"]
    matrix = matrix.loc[cols, cols]
    matrix = matrix.fillna(matrix.T).fillna(0.0)
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    sns.heatmap(
        matrix,
        cmap=sns.light_palette(ACCENT, as_cmap=True),
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.7,
        cbar_kws={"shrink": 0.82},
        ax=ax,
    )
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    save(fig, FIG / "presentation_categorical_cramers_heatmap.png")


def make_top_associations(tables: dict[str, pd.DataFrame]) -> None:
    numeric = (
        tables["numeric_corr"]
        .query("feature != 'price_log1p'")
        .sort_values("abs_spearman", ascending=False)
        .head(5)
        .copy()
    )
    numeric["value"] = numeric["spearman_with_price"].astype(float)

    categorical = tables["categorical_assoc"].head(5).copy()
    categorical["value"] = categorical["cramers_v"].astype(float)

    mi = (
        tables["mi"]
        .query("~feature.str.startswith('price_log1p')", engine="python")
        .head(5)
        .copy()
    )
    mi["value"] = mi["mutual_information"].astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))
    groups = [
        (numeric, "Числовые признаки\n(Spearman)", ACCENT, "value"),
        (categorical, "Категориальные признаки\n(Cramer's V)", ORANGE, "value"),
        (mi, "Mutual Information", GREEN, "value"),
    ]
    for ax, (table, title, color, value_col) in zip(axes, groups):
        plot_df = table.iloc[::-1]
        ax.barh(plot_df["feature"], plot_df[value_col], color=color, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("значение")
        for idx, value in enumerate(plot_df[value_col]):
            ax.text(value + 0.01 * max(plot_df[value_col]), idx, f"{value:.2f}", va="center", fontsize=10, fontweight="bold")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    fig.suptitle("Наиболее сильные связи признаков с ценой", fontsize=19, fontweight="bold", color=TITLE, y=1.03)
    save(fig, FIG / "presentation_top_associations.png")


def make_numeric_associations(tables: dict[str, pd.DataFrame]) -> None:
    numeric = (
        tables["numeric_corr"]
        .query("feature != 'price_log1p'")
        .sort_values("abs_spearman", ascending=False)
        .head(6)
        .copy()
    )
    numeric["value"] = numeric["spearman_with_price"].astype(float)
    numeric = numeric.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    bars = ax.barh(numeric["feature"], numeric["value"], color=ACCENT, alpha=0.92)
    ax.set_xlabel("Spearman")
    for bar, value in zip(bars, numeric["value"], strict=False):
        ax.text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=11, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    save(fig, FIG / "presentation_numeric_associations.png")


def make_categorical_associations(tables: dict[str, pd.DataFrame]) -> None:
    cat = tables["categorical_assoc"].head(8).copy().iloc[::-1]

    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    bars = ax.barh(cat["feature"], cat["cramers_v"], color=ORANGE, alpha=0.92)
    ax.set_xlabel("Cramer's V")
    for bar, value in zip(bars, cat["cramers_v"], strict=False):
        ax.text(value + 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=11, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    save(fig, FIG / "presentation_categorical_associations.png")


def make_signal_types_summary(tables: dict[str, pd.DataFrame]) -> None:
    numeric = (
        tables["numeric_corr"]
        .set_index("feature")
        .loc[["age", "mileage", "engine_power_hp"], ["abs_spearman"]]
        .reset_index()
    )
    numeric["feature_ru"] = numeric["feature"].map(
        {
            "age": "Возраст",
            "mileage": "Пробег",
            "engine_power_hp": "Мощность",
        }
    )

    categorical = (
        tables["categorical_assoc"]
        .set_index("feature")
        .loc[["brand", "model", "condition", "body_type"], ["cramers_v"]]
        .reset_index()
    )
    categorical["feature_ru"] = categorical["feature"].map(
        {
            "brand": "Марка",
            "model": "Модель",
            "condition": "Состояние",
            "body_type": "Кузов",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), gridspec_kw={"wspace": 0.28})

    ax = axes[0]
    plot_df = numeric.sort_values("abs_spearman").copy()
    bars = ax.barh(plot_df["feature_ru"], plot_df["abs_spearman"], color=ACCENT, alpha=0.92)
    ax.set_xlabel("|Spearman|")
    ax.set_xlim(0, 0.7)
    add_corner_tag(ax, "Технические признаки -> плавная зависимость")
    for bar, value in zip(bars, plot_df["abs_spearman"], strict=False):
        ax.text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=11, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    plot_df = categorical.sort_values("cramers_v").copy()
    bars = ax.barh(plot_df["feature_ru"], plot_df["cramers_v"], color=ORANGE, alpha=0.92)
    ax.set_xlabel("Cramér's V")
    ax.set_xlim(0, 0.8)
    add_corner_tag(ax, "Категориальные признаки -> скачки между группами")
    for bar, value in zip(bars, plot_df["cramers_v"], strict=False):
        ax.text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=11, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    save(fig, FIG / "presentation_signal_types_summary.png")


def make_dependency_plot(
    df: pd.DataFrame,
    *,
    feature: str,
    out_name: str,
    x_label: str,
    x_transform=lambda s: s,
    x_upper_q: float = 0.99,
    y_upper_q: float = 0.99,
) -> None:
    work = df[[feature, "price"]].copy()
    work = work.dropna()
    if work.empty:
        return

    x = x_transform(work[feature].astype(float))
    y = work["price"].astype(float) / 1_000_000
    x_upper = float(x.quantile(x_upper_q))
    y_upper = float(y.quantile(y_upper_q))
    work = pd.DataFrame({"x": x.clip(upper=x_upper), "y": y.clip(upper=y_upper)})

    sample = work.sample(min(2200, len(work)), random_state=42)
    bins = pd.qcut(work["x"], q=min(14, work["x"].nunique()), duplicates="drop")
    trend = (
        work.groupby(bins, observed=False)
        .agg(x=("x", "median"), y=("y", "median"))
        .dropna()
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.scatter(sample["x"], sample["y"], s=16, alpha=0.18, color=ACCENT, edgecolors="none")
    ax.plot(trend["x"], trend["y"], color=ORANGE, linewidth=2.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Цена, млн руб.")
    add_source_note(ax, f"Оси ограничены по p99; линия показывает медиану цены по квантильным бинам")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    save(fig, FIG / out_name)


def make_corr_heatmap(tables: dict[str, pd.DataFrame]) -> None:
    corr = tables["corr_matrix"].set_index("feature")
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    sns.heatmap(
        corr,
        cmap=sns.diverging_palette(245, 15, as_cmap=True),
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.7,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    save(fig, FIG / "numeric_correlation_heatmap.png")


def make_feature_constraints(tables: dict[str, pd.DataFrame]) -> None:
    missing = tables["missing"].sort_values("missing_pct", ascending=False).head(5).copy()
    high_card = tables["missing"].query("dtype == 'object'").sort_values("n_unique_non_null", ascending=False).head(5).copy()
    constants = tables["missing"].query("n_unique_non_null <= 1").copy()

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.8))

    ax = axes[0]
    plot_df = missing.iloc[::-1]
    ax.barh(plot_df["feature"], plot_df["missing_pct"], color=RED, alpha=0.85)
    ax.set_title("Пропуски")
    ax.set_xlabel("% пропусков")
    for i, value in enumerate(plot_df["missing_pct"]):
        ax.text(value + 1, i, f"{value:.1f}%", va="center", fontsize=10, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    plot_df = high_card.iloc[::-1]
    ax.barh(plot_df["feature"], plot_df["n_unique_non_null"], color=TEAL, alpha=0.9)
    ax.set_title("Высокая кардинальность")
    ax.set_xlabel("Уникальных значений")
    for i, value in enumerate(plot_df["n_unique_non_null"]):
        ax.text(value + max(plot_df["n_unique_non_null"]) * 0.02, i, f"{int(value)}", va="center", fontsize=10, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax = axes[2]
    reasons = pd.Series(
        {
            "Почти константные": int((tables["missing"]["n_unique_non_null"] <= 1).sum()),
            "Сильные пропуски": int((tables["missing"]["missing_pct"] >= 20).sum()),
            "Высокая кардинальность": int((tables["missing"]["n_unique_non_null"] >= 100).sum()),
        }
    )
    wedges, _ = ax.pie(reasons.values, colors=[GREEN, ORANGE, ACCENT], startangle=90, wedgeprops={"linewidth": 1, "edgecolor": "white"})
    ax.set_title("Главные ограничения")
    ax.legend(wedges, [f"{k}: {v}" for k, v in reasons.items()], loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=1)

    fig.suptitle("Какие признаки ограничивают качество анализа", fontsize=19, fontweight="bold", color=TITLE, y=1.03)
    save(fig, FIG / "presentation_feature_constraints.png")


def make_feature_processing_counts() -> None:
    prep = pd.read_json(BASE / "data/processed/preprocessing_summary.json", typ="series")
    rows = pd.Series(
        {
            "Сырые строки": int(prep["raw_rows"]),
            "После приведения схемы": int(prep["after_schema_rows"]),
            "После очистки": int(prep["cleaned_rows"]),
            "Исключено": int(prep["dropped_rows"]),
        }
    )
    colors = [GRAY, ACCENT, GREEN, ORANGE]

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    bars = ax.bar(rows.index, rows.values, color=colors, width=0.62)
    ax.set_ylabel("Количество строк")
    plt.xticks(rotation=15, ha="right")
    for bar, value in zip(bars, rows.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(rows.values) * 0.015, f"{int(value)}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    save(fig, FIG / "presentation_feature_processing_counts.png")


def make_mape_by_segments(tables: dict[str, pd.DataFrame]) -> None:
    price_seg = tables["price_segment_metrics"].copy()
    cond = tables["condition_metrics"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.8), gridspec_kw={"width_ratios": [1.25, 0.9]})

    ax = axes[0]
    colors = sns.color_palette("blend:#2F5597,#C55A11", n_colors=len(price_seg))
    bars = ax.bar(price_seg["price_segment"], price_seg["mape"], color=colors, width=0.65)
    ax.set_title("MAPE по ценовым сегментам")
    ax.set_ylabel("MAPE, %")
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    for bar, value in zip(bars, price_seg["mape"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.25, f"{value:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    bars = ax.bar(["Новые", "С пробегом"], cond["mape"], color=[ACCENT, ORANGE], width=0.6)
    ax.set_title("MAPE по сегментам new/used")
    ax.set_ylabel("MAPE, %")
    ax.set_xlabel("")
    for bar, value in zip(bars, cond["mape"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{value:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.suptitle("Качество заметно зависит от рыночного сегмента", fontsize=19, fontweight="bold", color=TITLE, y=1.03)
    save(fig, FIG / "stage2_mape_by_segments.png")


def make_segmented_effects(tables: dict[str, pd.DataFrame]) -> None:
    corr = tables["segmented_numeric"].copy()
    feature_map = {
        "engine_power_hp": "Мощность",
        "mileage": "Пробег",
        "age": "Возраст",
        "engine_volume": "Объём",
    }
    corr["feature_ru"] = corr["feature"].map(feature_map)
    pivot = corr.pivot(index="feature_ru", columns="segment", values="spearman").reindex(["Мощность", "Пробег", "Возраст", "Объём"])

    fig, ax = plt.subplots(figsize=(9.6, 5.1))
    x = np.arange(len(pivot.index))
    width = 0.34
    ax.bar(x - width / 2, pivot["new"], width=width, color=ACCENT, label="Новые")
    ax.bar(x + width / 2, pivot["used"], width=width, color=ORANGE, label="С пробегом")
    ax.axhline(0, color=GRAY, linewidth=1)
    ax.set_xticks(x, pivot.index)
    ax.set_ylabel("Spearman с ценой")
    ax.legend(loc="upper right")
    for idx, value in enumerate(pivot["new"]):
        ax.text(idx - width / 2, value + (0.02 if value >= 0 else -0.06), f"{value:.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=10, fontweight="bold")
    for idx, value in enumerate(pivot["used"]):
        ax.text(idx + width / 2, value + (0.02 if value >= 0 else -0.06), f"{value:.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=10, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    save(fig, FIG / "presentation_segmented_effects.png")


def make_homogeneity_overlay(df: pd.DataFrame) -> None:
    work = df[["condition", "price"]].dropna().copy()
    upper = float(work["price"].quantile(0.995))
    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    for cond, color, label in [
        ("new", ACCENT, "Новые"),
        ("used", ORANGE, "С пробегом"),
    ]:
        subset = work.loc[work["condition"] == cond, "price"].clip(upper=upper) / 1_000_000
        sns.histplot(subset, bins=30, stat="density", color=color, alpha=0.35, edgecolor=None, ax=ax, label=label)
        sns.kdeplot(subset, color=color, linewidth=2.2, ax=ax)
    ax.set_xlabel("Цена, млн руб.")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    add_source_note(ax, f"Ось ограничена по p99.5 = {upper/1_000_000:.2f} млн руб.")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    save(fig, FIG / "presentation_homogeneity_new_used_overlay.png")


def make_complete_case_models(tables: dict[str, pd.DataFrame]) -> None:
    metrics = tables["complete_case_models"].copy()
    label_map = {
        "xgboost_complete_case": "XGBoost",
        "random_forest_complete_case": "Random Forest",
        "ridge_complete_case": "Ridge",
        "linear_regression_complete_case": "Linear Regression",
    }
    metrics["model_ru"] = metrics["model"].map(label_map).fillna(metrics["model"])
    metrics = metrics.sort_values("r2", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9), gridspec_kw={"width_ratios": [1, 1]})

    ax = axes[0]
    bars = ax.barh(metrics["model_ru"], metrics["r2"], color=[ACCENT, TEAL, GREEN, ORANGE])
    ax.set_title("R² на complete-case подвыборке")
    ax.set_xlabel("R²")
    ax.set_xlim(0, 1.0)
    for bar, value in zip(bars, metrics["r2"]):
        ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=10, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    bars = ax.barh(metrics["model_ru"], metrics["mape"], color=[ACCENT, TEAL, GREEN, ORANGE])
    ax.set_title("MAPE на complete-case подвыборке")
    ax.set_xlabel("MAPE, %")
    for bar, value in zip(bars, metrics["mape"]):
        ax.text(value + 0.2, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=10, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.suptitle("Проверка моделей на полностью заполненных карточках", fontsize=19, fontweight="bold", color=TITLE, y=1.03)
    save(fig, FIG / "presentation_complete_case_models.png")


def make_compact_linear_tradeoff(tables: dict[str, pd.DataFrame]) -> None:
    metrics = tables["linear_reduced"].copy()
    metrics = metrics[(metrics["model"] == "ridge") & (metrics["status"] == "ok")].copy()
    order = [
        "dry13",
        "dry12_no_color",
        "dry11_no_color_body",
        "dry10_no_color_body_drive",
        "dry9_no_color_body_drive_fuel",
        "dry8_no_color_body_drive_fuel_volume",
    ]
    label_map = {
        "dry13": "13",
        "dry12_no_color": "12",
        "dry11_no_color_body": "11",
        "dry10_no_color_body_drive": "10",
        "dry9_no_color_body_drive_fuel": "9",
        "dry8_no_color_body_drive_fuel_volume": "8",
    }
    remove_map = {
        "dry13": "База",
        "dry12_no_color": "- color",
        "dry11_no_color_body": "- color, body_type",
        "dry10_no_color_body_drive": "- + drive_type",
        "dry9_no_color_body_drive_fuel": "- + fuel_type",
        "dry8_no_color_body_drive_fuel_volume": "- + engine_volume",
    }
    metrics["feature_set"] = pd.Categorical(metrics["feature_set"], categories=order, ordered=True)
    metrics = metrics.sort_values("feature_set")
    metrics["label"] = metrics["feature_set"].map(label_map)
    metrics["remove"] = metrics["feature_set"].map(remove_map)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), gridspec_kw={"width_ratios": [1, 1]})

    ax = axes[0]
    line = ax.plot(metrics["feature_count"], metrics["mape"], color=ACCENT, marker="o", linewidth=2.4)[0]
    ax.invert_xaxis()
    ax.set_title("MAPE при сжатии набора признаков")
    ax.set_xlabel("Число признаков")
    ax.set_ylabel("MAPE, %")
    for _, row in metrics.iterrows():
        ax.text(row["feature_count"], row["mape"] + 0.08, f"{row['mape']:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.axvspan(10.7, 12.3, color=LIGHT3, alpha=0.9)
    ax.text(11.5, metrics["mape"].min() + 0.15, "рабочая зона", ha="center", va="bottom", fontsize=10, color=TITLE, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    bars = ax.barh(metrics["label"], metrics["r2"], color=[ACCENT, GREEN, GREEN, ORANGE, ORANGE, RED])
    ax.set_title("R² по компактным наборам")
    ax.set_xlabel("R²")
    ax.set_xlim(0.86, 0.93)
    for bar, value, note in zip(bars, metrics["r2"], metrics["remove"], strict=False):
        ax.text(value + 0.001, bar.get_y() + bar.get_height() / 2, f"{value:.3f}   {note}", va="center", fontsize=10, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.suptitle("Насколько можно уменьшить набор без заметной потери качества", fontsize=19, fontweight="bold", color=TITLE, y=1.03)
    save(fig, FIG / "presentation_compact_linear_tradeoff.png")


def make_full_vs_complete_counts(tables: dict[str, pd.DataFrame]) -> None:
    metrics = tables["compact11_full_vs_complete"].copy()
    rows = (
        metrics[["dataset_variant", "rows"]]
        .drop_duplicates()
        .replace({"dataset_variant": {"full_imputed": "Весь датасет", "complete_case": "Только полные карточки"}})
    )
    stats_df = pd.DataFrame(
        {
            "dataset_variant": ["Весь датасет", "Только полные карточки"],
            "used_share": [45.02, 74.32],
            "median_price_mln": [2.91, 1.76],
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9), gridspec_kw={"width_ratios": [0.9, 1.1]})
    colors = [ACCENT, ORANGE]

    ax = axes[0]
    bars = ax.bar(rows["dataset_variant"], rows["rows"], color=colors, width=0.6)
    ax.set_title("Сколько строк остаётся")
    ax.set_ylabel("Количество карточек")
    for bar, value in zip(bars, rows["rows"], strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 60, f"{int(value)}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticklabels(rows["dataset_variant"], rotation=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    x = np.arange(len(stats_df))
    width = 0.34
    ax.bar(x - width / 2, stats_df["used_share"], width=width, color=ORANGE, label="Доля used, %")
    ax.bar(x + width / 2, stats_df["median_price_mln"], width=width, color=GREEN, label="Медианная цена, млн руб.")
    ax.set_title("Как меняется состав выборки")
    ax.set_xticks(x, stats_df["dataset_variant"])
    ax.set_ylabel("Значение показателя")
    ax.legend(loc="upper right")
    for idx, value in enumerate(stats_df["used_share"]):
        ax.text(idx - width / 2, value + 1.1, f"{value:.2f}", ha="center", fontsize=10, fontweight="bold")
    for idx, value in enumerate(stats_df["median_price_mln"]):
        ax.text(idx + width / 2, value + 1.1, f"{value:.2f}", ha="center", fontsize=10, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.suptitle("Complete-case уменьшает выборку и смещает её в сторону used-сегмента", fontsize=19, fontweight="bold", color=TITLE, y=1.03)
    save(fig, FIG / "presentation_full_vs_complete_counts.png")


def make_full_vs_complete_metrics(tables: dict[str, pd.DataFrame]) -> None:
    metrics = tables["compact11_full_vs_complete"].copy()
    metrics["dataset_label"] = metrics["dataset_variant"].replace({"full_imputed": "Весь датасет", "complete_case": "Только полные карточки"})
    metrics["model_label"] = metrics["model"].replace({"ridge": "Ridge", "linear_regression": "OLS"})
    order = ["Ridge", "OLS"]
    metrics["model_label"] = pd.Categorical(metrics["model_label"], categories=order, ordered=True)
    metrics = metrics.sort_values(["model_label", "dataset_label"])

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9))

    ax = axes[0]
    sns.barplot(data=metrics, x="model_label", y="r2", hue="dataset_label", palette=[ACCENT, ORANGE], ax=ax)
    ax.set_title("R²: весь датасет против complete-case")
    ax.set_xlabel("")
    ax.set_ylabel("R²")
    ax.set_ylim(0.65, 0.95)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=9, padding=3)
    ax.legend(title="")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    sns.barplot(data=metrics, x="model_label", y="mape", hue="dataset_label", palette=[ACCENT, ORANGE], ax=ax)
    ax.set_title("MAPE: весь датасет против complete-case")
    ax.set_xlabel("")
    ax.set_ylabel("MAPE, %")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=9, padding=3)
    ax.legend_.remove()
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.suptitle("На полных карточках качество не улучшается автоматически", fontsize=19, fontweight="bold", color=TITLE, y=1.03)
    save(fig, FIG / "presentation_full_vs_complete_metrics.png")


def main() -> None:
    setup_theme()
    df, tables = load_data()
    make_condition_counts(df)
    make_price_bands(tables)
    make_dataset_quality(tables)
    make_price_distribution(df)
    make_price_vs_log_distribution(df)
    make_price_qq(df)
    make_qq_raw_vs_log(df)
    make_condition_boxplot(df)
    make_condition_violin(df)
    make_condition_histograms(df)
    make_top_associations(tables)
    make_numeric_associations(tables)
    make_categorical_associations(tables)
    make_signal_types_summary(tables)
    make_body_type_boxplot(df)
    make_brand_boxplot(df)
    make_small_category_boxplots(df)
    make_smooth_vs_jump_example(df)
    make_dependency_plot(
        df,
        feature="age",
        out_name="presentation_dep_age_price.png",
        x_label="Возраст, лет",
    )
    make_dual_dependency_plot(
        df,
        feature="age",
        out_name="presentation_dep_age_dual.png",
        x_label="Возраст, лет",
    )
    make_dependency_plot(
        df,
        feature="mileage",
        out_name="presentation_dep_mileage_price.png",
        x_label="Пробег, тыс. км",
        x_transform=lambda s: s / 1000,
    )
    make_dual_dependency_plot(
        df,
        feature="mileage",
        out_name="presentation_dep_mileage_dual.png",
        x_label="Пробег, тыс. км",
        x_transform=lambda s: s / 1000,
    )
    make_dependency_plot(
        df,
        feature="engine_power_hp",
        out_name="presentation_dep_power_price.png",
        x_label="Мощность, л.с.",
    )
    make_dual_dependency_plot(
        df,
        feature="engine_power_hp",
        out_name="presentation_dep_power_dual.png",
        x_label="Мощность, л.с.",
    )
    make_dependency_plot(
        df,
        feature="engine_volume",
        out_name="presentation_dep_volume_price.png",
        x_label="Объём двигателя, л",
    )
    make_dual_dependency_plot(
        df,
        feature="engine_volume",
        out_name="presentation_dep_volume_dual.png",
        x_label="Объём двигателя, л",
    )
    make_corr_heatmap(tables)
    make_presentation_numeric_heatmap(df)
    make_presentation_categorical_heatmap(tables)
    make_feature_constraints(tables)
    make_feature_processing_counts()
    make_segmented_effects(tables)
    make_homogeneity_overlay(df)
    print("presentation figures regenerated")


if __name__ == "__main__":
    main()
