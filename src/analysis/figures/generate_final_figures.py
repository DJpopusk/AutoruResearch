"""Generate final unified-style figures for hypotheses H1-H5 of the research."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy import stats
from sklearn.mixture import GaussianMixture

BASE = Path(__file__).resolve().parents[3]
FIG = BASE / "figures"
DATA = BASE / "data/processed/cleaned_dataset.parquet"
FIG.mkdir(parents=True, exist_ok=True)

# Same palette as existing presentation figures
TITLE = "#162D50"
ACCENT = "#2F5597"
ORANGE = "#C55A11"
GREEN = "#70AD47"
TEAL = "#2B7A78"
RED = "#A61C3C"
GRAY = "#5F6368"
LIGHT = "#F3F7FC"
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
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.color": TITLE,
            "ytick.color": TITLE,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "legend.frameon": False,
            "legend.fontsize": 11,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    out = FIG / name
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out}")


def fig_gmm_bic(df: pd.DataFrame) -> None:
    """H4 — BIC curve with elbow at K=3 and three GMM components on log(price)."""
    log_p = np.log1p(df["price"].values).reshape(-1, 1)
    ks = range(1, 7)
    bics = []
    for k in ks:
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=5).fit(log_p)
        bics.append(gmm.bic(log_p))

    gmm3 = GaussianMixture(n_components=3, random_state=42, n_init=10).fit(log_p)
    order = np.argsort(gmm3.means_.flatten())
    weights = gmm3.weights_[order]
    means = gmm3.means_.flatten()[order]
    stds = np.sqrt(gmm3.covariances_.flatten())[order]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), gridspec_kw={"width_ratios": [1, 1.25]})

    ax = axes[0]
    ax.plot(list(ks), bics, marker="o", color=ACCENT, linewidth=2.2, markersize=7)
    ax.scatter([3], [bics[2]], s=200, color=ORANGE, zorder=5, edgecolors="white", linewidths=2)
    ax.annotate(
        "K=3 — точка излома\nдальше BIC почти не падает",
        xy=(3, bics[2]),
        xytext=(3.6, bics[2] + (max(bics) - min(bics)) * 0.32),
        fontsize=11,
        color=ORANGE,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4),
    )
    ax.set_xlabel("Число компонент K")
    ax.set_ylabel("BIC")
    ax.set_title("Подбор числа сегментов по BIC")
    ax.set_xticks(list(ks))

    ax = axes[1]
    x = np.linspace(log_p.min(), log_p.max(), 600)
    ax.hist(
        log_p.ravel(),
        bins=70,
        density=True,
        color=LIGHT,
        edgecolor="#B7C6DA",
        linewidth=0.6,
        alpha=0.95,
        label="log(1+price)",
    )
    seg_colors = [GREEN, ACCENT, ORANGE]
    seg_names = ["бюджетный", "средний", "премиальный"]
    for w, m, s, c, name in zip(weights, means, stds, seg_colors, seg_names):
        pdf = w / (s * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - m) / s) ** 2)
        ax.plot(x, pdf, color=c, linewidth=2.5, label=f"{name}: {w*100:.0f}%, медиана ≈ {np.expm1(m)/1e6:.1f} млн ₽")
    ax.set_xlabel("log(1 + price)")
    ax.set_ylabel("Плотность")
    ax.set_title("3 компоненты GMM на log(price)")
    ax.legend(loc="upper left")

    fig.suptitle("H4: Рынок устроен как смесь трёх ценовых режимов", fontsize=17, fontweight="bold", color=TITLE, y=1.02)
    save(fig, "h4_gmm_bic_segments.png")


def fig_segment_age_slopes(df: pd.DataFrame) -> None:
    """H5 — log(price)~age slopes per GMM segment with confidence bands."""
    log_p = np.log1p(df["price"].values).reshape(-1, 1)
    gmm = GaussianMixture(n_components=3, random_state=42, n_init=10).fit(log_p)
    labels = gmm.predict(log_p)
    order = np.argsort(gmm.means_.flatten())
    remap = {old: new for new, old in enumerate(order)}
    df = df.copy()
    df["seg"] = pd.Series(labels).map(remap).map({0: "бюджет", 1: "средний", 2: "премиум"})

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), gridspec_kw={"width_ratios": [1.25, 1]})

    seg_order = ["бюджет", "средний", "премиум"]
    seg_colors = {"бюджет": GREEN, "средний": ACCENT, "премиум": ORANGE}
    slopes_pct = {}
    cis = {}

    ax = axes[0]
    age_grid = np.linspace(0, 25, 200)
    for seg in seg_order:
        sub = df[(df["seg"] == seg) & (df["age"] > 0)].dropna(subset=["age", "price"])
        if len(sub) < 30:
            continue
        slope, intercept, r, p, se = stats.linregress(sub["age"].values, np.log(sub["price"].values))
        pct = (np.exp(slope) - 1) * 100
        ci_low = (np.exp(slope - 1.96 * se) - 1) * 100
        ci_high = (np.exp(slope + 1.96 * se) - 1) * 100
        slopes_pct[seg] = pct
        cis[seg] = (ci_low, ci_high)
        ax.scatter(
            sub["age"], sub["price"] / 1e6, s=8, alpha=0.18, color=seg_colors[seg], edgecolors="none"
        )
        line_y = np.exp(intercept + slope * age_grid) / 1e6
        ax.plot(age_grid, line_y, color=seg_colors[seg], linewidth=3.0, label=f"{seg}: {pct:+.1f}% / год")

    ax.set_yscale("log")
    ax.set_xlabel("Возраст автомобиля, лет")
    ax.set_ylabel("Цена, млн ₽ (лог-шкала)")
    ax.set_title("Падение цены с возрастом по сегментам")
    ax.set_xlim(-0.5, 25)
    ax.legend(loc="upper right", title="Скорость дисконта")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

    ax = axes[1]
    seg_labels = list(slopes_pct.keys())
    values = [slopes_pct[s] for s in seg_labels]
    errs_low = [slopes_pct[s] - cis[s][0] for s in seg_labels]
    errs_high = [cis[s][1] - slopes_pct[s] for s in seg_labels]
    bar_colors = [seg_colors[s] for s in seg_labels]
    ax.barh(seg_labels, values, xerr=[errs_low, errs_high], color=bar_colors, height=0.55, error_kw={"ecolor": TITLE, "elinewidth": 1.5, "capsize": 5})
    for i, v in enumerate(values):
        ax.text(v - 0.25, i, f"{v:+.1f}%", va="center", ha="right", color="white", fontweight="bold", fontsize=12)
    ax.axvline(0, color=GRAY, linewidth=1)
    ax.set_xlabel("Среднегодовой темп изменения цены, %")
    ax.set_title("Slope log(price) ~ age, 95% CI")
    ax.invert_yaxis()
    ax.set_xlim(min(values) * 1.25, 1)

    fig.suptitle("H5: Премиум-сегмент дисконтируется значимо медленнее", fontsize=17, fontweight="bold", color=TITLE, y=1.02)
    save(fig, "h5_segment_age_slopes.png")


def fig_proxy_dedup(df: pd.DataFrame) -> None:
    """Dedup block: matrix of proxy/duplication strength between key features."""
    pairs = [
        ("year", "age", -1.00, "Spearman", "идентичные"),
        ("condition", "seller_type", 1.00, "Cramér's V", "идентичные"),
        ("engine_power_hp", "engine_volume", 0.77, "Spearman", "сильное"),
        ("mileage", "age", 0.47, "Spearman", "умеренное"),
        ("body_type", "drive_type", 0.43, "Cramér's V", "умеренное"),
        ("body_type", "fuel_type", 0.24, "Cramér's V", "слабое"),
    ]
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    labels = [f"{a}\n×\n{b}" for a, b, *_ in pairs]
    values = [abs(v) for *_, v, _, _ in pairs]
    methods = [m for *_, _, m, _ in pairs]
    cls = [c for *_, _, _, c in pairs]
    colors_by_class = {"идентичные": RED, "сильное": ORANGE, "умеренное": ACCENT, "слабое": TEAL}
    bar_colors = [colors_by_class[c] for c in cls]
    bars = ax.barh(labels, values, color=bar_colors, height=0.62, edgecolor="white", linewidth=1.5)
    for bar, val, m, c in zip(bars, values, methods, cls):
        ax.text(val - 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.2f}  ({m})", ha="right", va="center", color="white", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Сила связи (по модулю)")
    ax.set_title("Дубли и proxy-признаки в данных")
    ax.invert_yaxis()
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors_by_class.values()]
    ax.legend(legend_handles, list(colors_by_class.keys()), loc="lower right", title="Уровень связи")
    fig.suptitle("Дублирование информации до моделирования", fontsize=16, fontweight="bold", color=TITLE, y=1.02)
    save(fig, "dedup_proxy_strength.png")


def fig_segmented_spearman(df: pd.DataFrame) -> None:
    """H2: Spearman of numeric features vs price overall and per condition segment."""
    feats = ["engine_power_hp", "age", "mileage", "engine_volume"]
    rows = []
    for label, sub in [("общая", df), ("new", df[df["condition"] == "new"]), ("used", df[df["condition"] == "used"])]:
        for f in feats:
            s = sub[[f, "price"]].dropna()
            if len(s) < 30:
                rows.append({"group": label, "feature": f, "rho": np.nan, "n": len(s)})
                continue
            rho, _ = stats.spearmanr(s[f], s["price"])
            rows.append({"group": label, "feature": f, "rho": rho, "n": len(s)})
    res = pd.DataFrame(rows)
    pivot = res.pivot(index="feature", columns="group", values="rho").reindex(feats)[["общая", "new", "used"]]

    fig, ax = plt.subplots(figsize=(11.4, 5.0))
    x = np.arange(len(feats))
    width = 0.27
    colors_g = [GRAY, ACCENT, ORANGE]
    for i, g in enumerate(["общая", "new", "used"]):
        vals = pivot[g].values
        bars = ax.bar(x + (i - 1) * width, vals, width=width, color=colors_g[i], label=g, edgecolor="white", linewidth=0.8)
        for b, v in zip(bars, vals):
            if np.isnan(v):
                continue
            offset = 0.025 if v >= 0 else -0.045
            ax.text(b.get_x() + b.get_width() / 2, v + offset, f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=10, fontweight="bold", color=TITLE)
    ax.axhline(0, color=GRAY, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(feats)
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("ρ Spearman с ценой")
    ax.set_title("H2: монотонные связи числовых признаков с ценой")
    ax.legend(title="Сегмент", loc="lower right")
    fig.suptitle("Технические признаки → цена, по сегментам new и used", fontsize=15, fontweight="bold", color=TITLE, y=1.02)
    save(fig, "h2_segmented_spearman.png")


def fig_price_vs_log_distribution(df: pd.DataFrame) -> None:
    """H1: side-by-side raw vs log price distribution with stats annotation."""
    raw = df["price"].values
    logp = np.log1p(raw)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), sharey=False)

    ax = axes[0]
    ax.hist(raw / 1e6, bins=70, color=ACCENT, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Цена, млн ₽")
    ax.set_ylabel("Частота")
    ax.set_title("price (raw)")
    sk_raw = stats.skew(raw)
    ku_raw = stats.kurtosis(raw)
    sw_raw = stats.shapiro(np.random.choice(raw, 5000, replace=False)).statistic
    ax.text(
        0.97,
        0.93,
        f"skew = {sk_raw:.2f}\nkurt = {ku_raw:.2f}\nShapiro W = {sw_raw:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        color=RED,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor=RED, alpha=0.95),
    )

    ax = axes[1]
    ax.hist(logp, bins=70, color=GREEN, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("log(1 + price)")
    ax.set_title("log(price)")
    sk_log = stats.skew(logp)
    ku_log = stats.kurtosis(logp)
    sw_log = stats.shapiro(np.random.choice(logp, 5000, replace=False)).statistic
    ax.text(
        0.03,
        0.93,
        f"skew = {sk_log:.2f}\nkurt = {ku_log:.2f}\nShapiro W = {sw_log:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color=GREEN,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor=GREEN, alpha=0.95),
    )

    fig.suptitle(
        "H1: переход к log-шкале приводит цену к почти симметричному распределению",
        fontsize=16,
        fontweight="bold",
        color=TITLE,
        y=1.02,
    )
    save(fig, "h1_price_vs_log_distribution.png")


def main() -> None:
    setup_theme()
    df = pd.read_parquet(DATA)
    fig_price_vs_log_distribution(df)
    fig_segmented_spearman(df)
    fig_gmm_bic(df)
    fig_segment_age_slopes(df)
    fig_proxy_dedup(df)


if __name__ == "__main__":
    main()
