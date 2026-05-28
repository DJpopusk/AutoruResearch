"""Filesystem and dataframe persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(df: pd.DataFrame, csv_path: Path, parquet_path: Path) -> None:
    ensure_dir(csv_path.parent)
    ensure_dir(parquet_path.parent)
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)


def append_to_aggregate(
    df: pd.DataFrame,
    aggregate_path: Path | str,
    run_tag: str | None = None,
    dedup_key: str = "url",
) -> pd.DataFrame:
    """Дописывает df прогона в общий parquet с дедупликацией.

    - Помечает строки колонкой ``source_run`` (= run_tag), чтобы знать, из какого прогона запись.
    - Если общий parquet уже есть — конкатенирует и дедуплицирует по ``dedup_key`` (keep=last).
    - Возвращает объединённый датафрейм.
    """
    aggregate_path = Path(aggregate_path)
    ensure_dir(aggregate_path.parent)

    df = df.copy()
    if run_tag is not None and "source_run" not in df.columns:
        df["source_run"] = run_tag

    if aggregate_path.exists():
        existing = pd.read_parquet(aggregate_path)
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df

    if dedup_key in combined.columns:
        combined = combined.drop_duplicates(subset=[dedup_key], keep="last").reset_index(drop=True)

    combined.to_parquet(aggregate_path, index=False)
    return combined


def load_tabular(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format: {path}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return default or {}
    return json.loads(path.read_text(encoding="utf-8"))
