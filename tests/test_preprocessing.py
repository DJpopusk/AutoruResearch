from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.preprocessing.cleaner import PreprocessConfig, preprocess_dataset


def test_preprocess_dataset_normalizes_and_filters(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        [
            {
                "brand": "toyota",
                "model": "camry",
                "year": "2019",
                "price": "2 300 000 ₽",
                "mileage": "80 000 км",
                "seller_type": "Дилер",
                "url": "u1",
                "parsed_at": "2026-01-01T10:00:00Z",
            },
            {
                "brand": "toyota",
                "model": "camry",
                "year": "2019",
                "price": "2 300 000 ₽",
                "mileage": "80 000 км",
                "seller_type": "Дилер",
                "url": "u1",
                "parsed_at": "2026-01-01T10:00:00Z",
            },
            {
                "brand": "bmw",
                "model": "x5",
                "year": "2005",
                "price": "999999999 ₽",
                "mileage": "50 000 км",
                "seller_type": "Частник",
                "url": "u2",
                "parsed_at": "2026-01-02T10:00:00Z",
            },
        ]
    )

    input_path = tmp_path / "raw.csv"
    raw.to_csv(input_path, index=False)

    output_dir = tmp_path / "processed"
    config = PreprocessConfig(input_path=input_path, output_dir=output_dir)
    cleaned = preprocess_dataset(config)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["url"] == "u1"
    assert cleaned.iloc[0]["seller_type"] == "dealer"
    assert float(cleaned.iloc[0]["price"]) == 2300000.0

    assert (output_dir / "cleaned_dataset.csv").exists()
    assert (output_dir / "cleaned_dataset.parquet").exists()
    assert (output_dir / "preprocessing_summary.json").exists()


def test_preprocess_keeps_rows_with_missing_year_if_other_values_are_valid(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        [
            {
                "brand": "audi",
                "model": "a6",
                "year": None,
                "price": "3 200 000 ₽",
                "mileage": "25 000 км",
                "url": "u3",
                "parsed_at": "2026-01-01T10:00:00Z",
            }
        ]
    )

    input_path = tmp_path / "raw.csv"
    raw.to_csv(input_path, index=False)

    cleaned = preprocess_dataset(PreprocessConfig(input_path=input_path, output_dir=tmp_path / "processed"))

    assert len(cleaned) == 1
    assert float(cleaned.iloc[0]["price"]) == 3200000.0
    assert pd.isna(cleaned.iloc[0]["year"])


def test_preprocess_marks_and_optionally_excludes_commercial_like_rows(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        [
            {
                "brand": "audi",
                "model": "q5",
                "year": "2024",
                "price": "5 500 000 ₽",
                "mileage": "0 км",
                "seller_type": "Дилер",
                "condition": "новый",
                "description_text": "Кредит, лизинг, trade-in, скидка, звоните!",
                "url": "u1",
                "parsed_at": "2026-01-01T10:00:00Z",
            },
            {
                "brand": "toyota",
                "model": "camry",
                "year": "2019",
                "price": "2 300 000 ₽",
                "mileage": "80 000 км",
                "seller_type": "Частник",
                "condition": "б/у",
                "description_text": "Один владелец, обслуживалась вовремя.",
                "url": "u2",
                "parsed_at": "2026-01-01T10:00:00Z",
            },
        ]
    )

    input_path = tmp_path / "raw.csv"
    raw.to_csv(input_path, index=False)

    cleaned = preprocess_dataset(PreprocessConfig(input_path=input_path, output_dir=tmp_path / "processed"))
    row_map = {row["url"]: row for _, row in cleaned.iterrows()}

    assert bool(row_map["u1"]["is_commercial_like"]) is True
    assert int(row_map["u1"]["commercial_signal_count"]) >= 2
    assert bool(row_map["u2"]["is_commercial_like"]) is False

    cleaned_excluded = preprocess_dataset(
        PreprocessConfig(
            input_path=input_path,
            output_dir=tmp_path / "processed_excluded",
            exclude_commercial_like=True,
        )
    )

    assert cleaned_excluded["url"].tolist() == ["u2"]
