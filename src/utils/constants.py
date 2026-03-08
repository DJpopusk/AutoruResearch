"""Project-wide constants."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"

AUTORU_SCHEMA = [
    "brand",
    "model",
    "generation",
    "year",
    "price",
    "mileage",
    "body_type",
    "color",
    "engine_volume",
    "engine_power_hp",
    "fuel_type",
    "transmission",
    "drive_type",
    "steering_wheel",
    "condition",
    "owners_count",
    "pts_type",
    "customs",
    "region",
    "seller_type",
    "description_text",
    "url",
    "parsed_at",
]

NUMERIC_FEATURES = [
    "year",
    "price",
    "mileage",
    "engine_volume",
    "engine_power_hp",
    "owners_count",
]

CATEGORICAL_FEATURES = [
    "brand",
    "model",
    "generation",
    "body_type",
    "color",
    "fuel_type",
    "transmission",
    "drive_type",
    "steering_wheel",
    "condition",
    "pts_type",
    "customs",
    "region",
    "seller_type",
]

DEFAULT_USER_AGENT = "Mozilla/5.0"
