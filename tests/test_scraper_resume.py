from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.parser.scraper import AutoRuScraper, ScrapeConfig


def test_scrape_resume_reuses_checkpoint_when_no_new_listing_urls(tmp_path: Path, monkeypatch) -> None:
    checkpoint_path = tmp_path / "autoru_checkpoint.jsonl"
    state_path = tmp_path / "autoru_state.json"
    output_csv = tmp_path / "autoru_raw.csv"
    output_parquet = tmp_path / "autoru_raw.parquet"

    record = {
        "brand": "toyota",
        "model": "camry",
        "price": 2300000,
        "mileage": 80000,
        "url": "https://auto.ru/cars/used/sale/toyota/camry/1/",
        "parsed_at": "2026-03-08T00:00:00Z",
    }
    checkpoint_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    state_path.write_text(
        json.dumps(
            {
                "processed_urls": [record["url"]],
                "failed_urls": [],
                "last_catalog_page": 20,
                "records_in_memory": 1,
            }
        ),
        encoding="utf-8",
    )

    config = ScrapeConfig(
        catalog_url="https://auto.ru/cars/all/",
        pages=20,
        output_csv=output_csv,
        output_parquet=output_parquet,
        state_file=state_path,
        checkpoint_jsonl=checkpoint_path,
    )

    monkeypatch.setattr(AutoRuScraper, "collect_listing_urls", lambda self: [])

    with AutoRuScraper(config) as scraper:
        df = scraper.run()

    assert len(df) == 1
    assert df.iloc[0]["url"] == record["url"]

    restored = pd.read_parquet(output_parquet)
    assert len(restored) == 1
    assert restored.iloc[0]["url"] == record["url"]
