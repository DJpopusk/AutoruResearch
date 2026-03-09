from __future__ import annotations

import json
from pathlib import Path

from src.parser.scraper import AutoRuScraper, ScrapeConfig


def _make_config(
    tmp_path: Path,
    *,
    condition_filter: str = "all",
    seller_type_filter: str = "all",
    min_mileage: int | None = None,
) -> ScrapeConfig:
    return ScrapeConfig(
        catalog_url="https://auto.ru/cars/all/",
        pages=10,
        output_csv=tmp_path / "autoru_raw.csv",
        output_parquet=tmp_path / "autoru_raw.parquet",
        state_file=tmp_path / "autoru_state.json",
        checkpoint_jsonl=tmp_path / "autoru_checkpoint.jsonl",
        condition_filter=condition_filter,
        seller_type_filter=seller_type_filter,
        min_mileage=min_mileage,
    )


def test_url_condition_filter_keeps_only_used_urls(tmp_path: Path) -> None:
    scraper = AutoRuScraper(_make_config(tmp_path, condition_filter="used"))

    assert scraper._url_matches_condition_filter("https://auto.ru/cars/used/sale/toyota/camry/1/")
    assert not scraper._url_matches_condition_filter("https://auto.ru/cars/new/group/audi/q5/1/")


def test_record_filters_require_private_used_when_requested(tmp_path: Path) -> None:
    scraper = AutoRuScraper(_make_config(tmp_path, condition_filter="used", seller_type_filter="private"))

    assert scraper._record_matches_filters({"condition": "used", "seller_type": "private"})
    assert not scraper._record_matches_filters({"condition": "new", "seller_type": "dealer"})
    assert not scraper._record_matches_filters({"condition": "used", "seller_type": "dealer"})


def test_record_filters_apply_min_mileage(tmp_path: Path) -> None:
    scraper = AutoRuScraper(_make_config(tmp_path, condition_filter="used", seller_type_filter="private", min_mileage=50_000))

    assert scraper._record_matches_filters({"condition": "used", "seller_type": "private", "mileage": 60_000})
    assert not scraper._record_matches_filters({"condition": "used", "seller_type": "private", "mileage": 49_999})
    assert not scraper._record_matches_filters({"condition": "used", "seller_type": "private", "mileage": None})


def test_state_signature_change_resets_last_page(tmp_path: Path) -> None:
    state_path = tmp_path / "autoru_state.json"
    state_path.write_text(
        json.dumps(
            {
                "signature": {
                    "catalog_url": "https://auto.ru/cars/all/",
                    "condition_filter": "all",
                    "seller_type_filter": "all",
                    "use_playwright": False,
                },
                "processed_urls": ["https://auto.ru/cars/new/group/audi/q5/1/"],
                "failed_urls": [],
                "last_catalog_page": 200,
                "records_in_memory": 1,
            }
        ),
        encoding="utf-8",
    )

    scraper = AutoRuScraper(_make_config(tmp_path, condition_filter="used", seller_type_filter="private"))
    scraper._load_resume_state()

    assert scraper.last_catalog_page == 0


def test_challenge_detection_flags_antibot_html(tmp_path: Path) -> None:
    scraper = AutoRuScraper(_make_config(tmp_path))
    html = """
    <html>
      <body>
        <h1>Подтвердите, что вы не робот</h1>
        <div>Проверка безопасности</div>
        <script>window.smartCaptcha = true;</script>
      </body>
    </html>
    """

    assert scraper._looks_like_challenge_page(html)


def test_empty_catalog_detection_distinguishes_stub_from_real_listing_page(tmp_path: Path) -> None:
    scraper = AutoRuScraper(_make_config(tmp_path))
    stub_html = "<html><body><h1>Авто.ру</h1><div>Пустой ответ</div></body></html>"
    listing_html = "<html><body><a href='/cars/used/sale/toyota/camry/1/'>car</a></body></html>"

    assert scraper._looks_like_empty_catalog_page(stub_html)
    assert not scraper._looks_like_empty_catalog_page(listing_html)


def test_start_page_override_ignores_resume_last_page(tmp_path: Path, monkeypatch) -> None:
    state_path = tmp_path / "autoru_state.json"
    state_path.write_text(
        json.dumps(
            {
                "signature": {
                    "catalog_url": "https://auto.ru/cars/all/",
                    "condition_filter": "all",
                    "seller_type_filter": "all",
                    "min_mileage": None,
                    "use_playwright": False,
                },
                "processed_urls": [],
                "failed_urls": [],
                "last_catalog_page": 200,
                "records_in_memory": 0,
            }
        ),
        encoding="utf-8",
    )

    scraper = AutoRuScraper(_make_config(tmp_path))
    scraper.config.pages = 160
    scraper.config.start_page = 150
    scraper._load_resume_state()
    visited_urls: list[str] = []

    class _DummyFetcher:
        def fetch_html(self, url: str) -> str:
            visited_urls.append(url)
            return "<html><body><a href='/cars/used/sale/toyota/camry/1/'>car</a></body></html>"

    scraper.fetcher = _DummyFetcher()
    monkeypatch.setattr(
        "src.parser.scraper.parse_listing_links",
        lambda html, page_url: ["https://auto.ru/cars/used/sale/toyota/camry/1/"],
    )

    scraper.collect_listing_urls()

    assert visited_urls[0].endswith("?page=150")
