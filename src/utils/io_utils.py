"""Filesystem and dataframe persistence utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(df: pd.DataFrame, csv_path: Path, parquet_path: Path) -> None:
    """Save dataframe to CSV and Parquet."""
    ensure_dir(csv_path.parent)
    ensure_dir(parquet_path.parent)
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)


def load_tabular(path: Path) -> pd.DataFrame:
    """Load CSV or Parquet into a dataframe."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format: {path}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist dictionary as JSON."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read JSON if file exists; return default otherwise."""
    if not path.exists():
        return default or {}
    return json.loads(path.read_text(encoding="utf-8"))
