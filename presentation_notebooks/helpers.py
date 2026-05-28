"""Общий код для презентационных ноутбуков.

Изолирован, чтобы каждый ноутбук оставался компактным.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from functions.constants import PROCESSED_DATA_DIR  # noqa: E402

RANDOM_STATE = 42

NUM_COLS_FULL = [
    "age", "mileage", "owners_count",
    "engine_power_hp", "engine_volume", "max_torque_nm",
    "cylinders_count", "valves_per_cylinder", "gears_count",
    "length", "width", "height", "wheelbase", "clearance",
    "front_track_width", "rear_track_width",
    "weight_curb", "max_speed", "acceleration_0_100",
    "fuel_tank_volume", "trunk_volume", "tax_amount",
    "fuel_consumption_mixed",
]

FEATURE_GROUPS = {
    "Время и пробег":   ["age", "mileage", "owners_count"],
    "Двигатель":        ["engine_power_hp", "engine_volume", "max_torque_nm",
                          "cylinders_count", "valves_per_cylinder", "gears_count"],
    "Габариты":         ["length", "width", "height", "wheelbase", "clearance",
                          "front_track_width", "rear_track_width"],
    "Масса и динамика": ["weight_curb", "max_speed", "acceleration_0_100"],
    "Объёмы и налог":   ["fuel_tank_volume", "trunk_volume", "tax_amount"],
    "Расход":           ["fuel_consumption_mixed"],
}

FEATURE_DESCRIPTIONS = {
    "age":                    ("лет", "Возраст автомобиля (год парсинга − год выпуска)"),
    "mileage":                ("км",  "Пробег"),
    "owners_count":           ("шт",  "Число владельцев по ПТС"),
    "engine_power_hp":        ("л.с.","Мощность двигателя"),
    "engine_volume":          ("л",   "Рабочий объём двигателя"),
    "max_torque_nm":          ("Н·м", "Максимальный крутящий момент"),
    "cylinders_count":        ("шт",  "Число цилиндров"),
    "valves_per_cylinder":    ("шт",  "Клапанов на цилиндр"),
    "gears_count":            ("шт",  "Число передач КПП"),
    "length":                 ("мм",  "Длина кузова"),
    "width":                  ("мм",  "Ширина кузова"),
    "height":                 ("мм",  "Высота кузова"),
    "wheelbase":              ("мм",  "Колёсная база"),
    "clearance":              ("мм",  "Дорожный просвет (клиренс)"),
    "front_track_width":      ("мм",  "Колея передняя"),
    "rear_track_width":       ("мм",  "Колея задняя"),
    "weight_curb":            ("кг",  "Снаряжённая масса"),
    "max_speed":              ("км/ч","Максимальная скорость"),
    "acceleration_0_100":     ("с",   "Разгон 0-100 км/ч"),
    "fuel_tank_volume":       ("л",   "Объём топливного бака"),
    "trunk_volume":           ("л",   "Объём багажника"),
    "tax_amount":             ("₽/год","Транспортный налог"),
    "fuel_consumption_mixed": ("л/100км","Расход топлива (смешанный цикл)"),
}

CAT_FEATURES = ["fuel_type", "transmission", "drive_type", "body_type"]


def set_plot_style() -> None:
    """Единый стиль графиков для презентации: белый фон, чёрный текст, сетка."""
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "savefig.facecolor": "white",
        "axes.edgecolor":    "black",
        "axes.labelcolor":   "black",
        "axes.titlecolor":   "black",
        "xtick.color":       "black",
        "ytick.color":       "black",
        "text.color":        "black",
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        "font.size":         10,
        "axes.titlesize":    12,
        "axes.titleweight":  "bold",
        "legend.frameon":    True,
        "legend.framealpha": 0.95,
    })


def load_data():
    """Загружает датасет, добавляет age и ln_price, делает train/test split."""
    df = pd.read_parquet(PROCESSED_DATA_DIR / "cleaned_dataset_all_runs.parquet")
    parsed_year = pd.to_datetime(df["parsed_at"], errors="coerce").dt.year
    df["age"] = parsed_year - df["year"]
    df.loc[df["age"] < 0, "age"] = np.nan
    df["ln_price"] = np.log(df["price"])

    key = ["price", "price_log1p", "ln_price", "age", "mileage",
           "engine_power_hp", "engine_volume",
           "fuel_type", "transmission", "drive_type", "body_type"]
    df_clean = df.dropna(subset=key).reset_index(drop=True)
    df_tr, df_te = train_test_split(df_clean, test_size=0.2, random_state=RANDOM_STATE)
    return df_clean, df_tr.reset_index(drop=True), df_te.reset_index(drop=True)


def impute_median(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Замена пропусков медианой + приведение к float64 (statsmodels не любит Int64)."""
    df = df.copy()
    for c in cols:
        df[c] = df[c].fillna(df[c].median()).astype(float)
    return df


def make_design(df: pd.DataFrame, num: list[str], cat: list[str]):
    """One-hot encode категориальные, конкатенация с числовыми."""
    cats = pd.get_dummies(df[cat], drop_first=True, dtype=float)
    X = pd.concat([df[num].reset_index(drop=True),
                   cats.reset_index(drop=True)], axis=1)
    return X, list(cats.columns)


def backward_elimination(X, y, alpha=0.05, must_keep=("const",)):
    """Backward elimination по p-value."""
    X = X.copy()
    history = []
    while True:
        m = sm.OLS(y, X).fit()
        pv = m.pvalues.drop(list(must_keep), errors="ignore")
        if pv.max() < alpha or len(pv) == 0:
            return m, X, history
        worst = pv.idxmax()
        history.append({"feature": worst, "pvalue": float(pv.max())})
        X = X.drop(columns=[worst])


def fit_fgls_wls(model, X_const, y):
    """Двухшаговая FGLS: оценка σ²(x) через вспомогательную регрессию, WLS."""
    e2 = np.maximum(model.resid ** 2, 1e-10)
    aux = sm.OLS(np.log(e2), X_const).fit()
    weights = 1.0 / np.exp(aux.fittedvalues)
    wls = sm.WLS(y, X_const, weights=weights).fit()
    return wls, weights


def white_test(resid, exog, num_cols=None):
    """Тест Уайта вручную (sm.het_white падает при наличии dummy из-за коллинеарности).

    Вспомогательная регрессия: e² = θ₀ + Σ θⱼ xⱼ + Σ θⱼⱼ xⱼ² + Σ_{j<k} θⱼₖ xⱼxₖ
    Возвращает: LM-статистику = n·R²_aux, p-value по χ²(df), df, R²_aux.
    Квадраты и взаимодействия строятся только для числовых (без dummy — у них x² = x).
    """
    from scipy import stats as _sp
    e2 = np.asarray(resid) ** 2
    X = exog.copy()
    if "const" in X.columns:
        X = X.drop(columns=["const"])
    n = len(X)

    if num_cols is None:
        num_cols = [c for c in X.columns if X[c].nunique() > 2]

    terms = [X]
    sq = pd.DataFrame({f"{c}^2": X[c] ** 2 for c in num_cols}, index=X.index)
    terms.append(sq)
    inter = {}
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            inter[f"{num_cols[i]}*{num_cols[j]}"] = X[num_cols[i]].values * X[num_cols[j]].values
    if inter:
        terms.append(pd.DataFrame(inter, index=X.index))

    aux = pd.concat(terms, axis=1)
    aux = aux.loc[:, aux.std() > 1e-10]  # убираем коллинеарные
    aux_c = sm.add_constant(aux)
    m = sm.OLS(e2, aux_c).fit()
    LM = n * m.rsquared
    df = aux.shape[1]
    p = float(_sp.chi2.sf(LM, df))
    return LM, p, df, float(m.rsquared)


def format_test_row(name: str, h0: str, stat_name: str, stat: float,
                    pval: float, alpha: float = 0.05) -> dict:
    """Универсальная строка таблицы тестов: H₀, статистика, p-value, вердикт."""
    decision = "отвергаем H₀" if pval < alpha else "не отвергаем H₀"
    return {
        "тест":         name,
        "H₀":           h0,
        stat_name:      round(stat, 4) if abs(stat) < 1e4 else f"{stat:.4g}",
        "p-value":      f"{pval:.3g}",
        f"α={alpha}":   decision,
    }


def derived_features(df: pd.DataFrame, center: bool = True) -> pd.DataFrame:
    """Добавляет ratio-признаки и центрированные age/mileage."""
    df = df.copy()
    df["hp_per_kg"]      = df["engine_power_hp"] / df["weight_curb"]
    df["hp_per_liter"]   = df["engine_power_hp"] / df["engine_volume"]
    df["ln_weight_curb"] = np.log(df["weight_curb"].clip(lower=1))
    df["ln_mileage"]     = np.log1p(df["mileage"])
    if center:
        df["age_c"]         = df["age"] - df["age"].mean()
        df["mileage_c"]     = df["mileage"] - df["mileage"].mean()
        df["age_c_sq"]      = df["age_c"] ** 2
        df["mileage_c_sq"]  = (df["mileage_c"] / 1e5) ** 2
        df["age_x_mileage"] = df["age_c"] * df["mileage_c"] / 1e6
        df["engine_x_age"]  = df["engine_volume"] * df["age_c"]
    return df
