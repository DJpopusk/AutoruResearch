"""Data cleaning and feature preparation for Auto.ru dataset."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.constants import AUTORU_SCHEMA, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.utils.io_utils import ensure_dir, load_tabular, save_dataframe, write_json

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PreprocessConfig:
    """Configuration for preprocessing stage."""

    input_path: Path
    output_dir: Path
    min_year: int = 1970
    max_reasonable_mileage: int = 1_500_000
    min_reasonable_price: int = 10_000
    max_reasonable_price: int = 500_000_000
    price_low_quantile: float = 0.005
    price_high_quantile: float = 0.995


CATEGORY_MAPS: dict[str, dict[str, str]] = {
    "seller_type": {
        "дилер": "dealer",
        "dealer": "dealer",
        "част": "private",
        "private": "private",
    },
    "fuel_type": {
        "бенз": "petrol",
        "диз": "diesel",
        "элект": "electric",
        "гибр": "hybrid",
        "газ": "gas",
    },
    "transmission": {
        "автомат": "automatic",
        "вариатор": "cvt",
        "робот": "robot",
        "механ": "manual",
    },
    "drive_type": {
        "перед": "fwd",
        "зад": "rwd",
        "пол": "awd",
    },
    "steering_wheel": {
        "лев": "left",
        "прав": "right",
    },
    "condition": {
        "нов": "new",
        "не бит": "used",
        "б/у": "used",
        "used": "used",
    },
}


def _to_numeric(series: pd.Series) -> pd.Series:
    """Extract numeric part from free-form strings and convert to float."""
    cleaned = (
        series.astype(str)
        .str.replace(r"[^0-9,\.\-]", "", regex=True)
        .str.replace(",", ".", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _normalize_text(value: object) -> str | None:
    """Normalize text cells for category mapping."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in {"nan", "none"}:
        return None
    return text


def _map_category(value: object, mapping: dict[str, str]) -> str | None:
    """Map category based on partial-key dictionary."""
    text = _normalize_text(value)
    if text is None:
        return None
    for key, mapped in mapping.items():
        if key in text:
            return mapped
    return text


def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all required columns are present in dataframe."""
    for column in AUTORU_SCHEMA:
        if column not in df.columns:
            df[column] = None
    return df[AUTORU_SCHEMA].copy()


def _filter_outliers(df: pd.DataFrame, config: PreprocessConfig) -> pd.DataFrame:
    """Apply straightforward filters for obviously invalid rows."""
    if df.empty:
        return df

    year_max = datetime.now().year + 1
    mask = df["year"].isna() | df["year"].between(config.min_year, year_max, inclusive="both")

    if df["price"].notna().sum() >= 20:
        low = df["price"].quantile(config.price_low_quantile)
        high = df["price"].quantile(config.price_high_quantile)
    else:
        low, high = config.min_reasonable_price, config.max_reasonable_price

    low = max(float(low), float(config.min_reasonable_price))
    high = min(float(high), float(config.max_reasonable_price))

    mask &= df["price"].between(low, high, inclusive="both")
    mask &= df["mileage"].fillna(0).between(0, config.max_reasonable_mileage, inclusive="both")
    mask &= df["engine_power_hp"].fillna(100).between(20, 2500, inclusive="both")
    mask &= df["engine_volume"].fillna(2.0).between(0.5, 12.0, inclusive="both")

    return df.loc[mask].copy()


def preprocess_dataset(config: PreprocessConfig) -> pd.DataFrame:
    """Load raw dataset, clean and normalize it, then save artifacts."""
    LOGGER.info("Loading raw dataset: %s", config.input_path)
    raw_df = load_tabular(config.input_path)
    raw_count = len(raw_df)

    df = _ensure_schema(raw_df)

    df = df.drop_duplicates(subset=["url"], keep="last")

    for column in NUMERIC_FEATURES:
        df[column] = _to_numeric(df[column])

    df["year"] = df["year"].round().astype("Int64")
    df["owners_count"] = df["owners_count"].round().astype("Int64")

    for column, mapping in CATEGORY_MAPS.items():
        df[column] = df[column].map(lambda x: _map_category(x, mapping))

    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].map(_normalize_text)
        df[column] = df[column].fillna("unknown")

    df["description_text"] = df["description_text"].astype(str).replace({"nan": "", "None": ""})
    df["parsed_at"] = pd.to_datetime(df["parsed_at"], errors="coerce", utc=True)

    before_outliers = len(df)
    df = _filter_outliers(df, config=config)

    df = df[df["price"].notna()]
    df["price"] = df["price"].astype(float)

    current_year = datetime.now().year
    df["age"] = current_year - df["year"].astype(float)
    df["price_log1p"] = np.log1p(df["price"])

    df = df.reset_index(drop=True)

    ensure_dir(config.output_dir)
    csv_path = config.output_dir / "cleaned_dataset.csv"
    parquet_path = config.output_dir / "cleaned_dataset.parquet"
    save_dataframe(df, csv_path=csv_path, parquet_path=parquet_path)

    summary = {
        "raw_rows": int(raw_count),
        "after_schema_rows": int(before_outliers),
        "cleaned_rows": int(len(df)),
        "dropped_rows": int(raw_count - len(df)),
        "missing_price_after_cleaning": int(df["price"].isna().sum()),
        "generated_at": datetime.now().isoformat(),
    }
    write_json(config.output_dir / "preprocessing_summary.json", summary)
    LOGGER.info("Preprocessing complete. Cleaned rows: %s", len(df))

    return df
