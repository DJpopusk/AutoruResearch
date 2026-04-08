from __future__ import annotations

import sys
import types
from pathlib import Path

from src.parser.fetchers import FetcherConfig, PlaywrightFetcher
from src.parser.scraper import AutoRuScraper, ScrapeConfig


def _make_scrape_config(tmp_path: Path, **overrides: object) -> ScrapeConfig:
    config = ScrapeConfig(
        catalog_url="https://auto.ru/cars/used/",
        pages=1,
        output_csv=tmp_path / "autoru_raw.csv",
        output_parquet=tmp_path / "autoru_raw.parquet",
        state_file=tmp_path / "autoru_state.json",
        checkpoint_jsonl=tmp_path / "autoru_checkpoint.jsonl",
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_scraper_passes_debug_browser_flags_to_fetcher(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _DummyPlaywrightFetcher:
        def __init__(self, config: FetcherConfig) -> None:
            captured["config"] = config

        def close(self) -> None:
            return None

    monkeypatch.setattr("src.parser.scraper.PlaywrightFetcher", _DummyPlaywrightFetcher)

    scraper = AutoRuScraper(
        _make_scrape_config(
            tmp_path,
            use_playwright=True,
            debug_browser=True,
            slow_mo_ms=150,
            pause_on_page=True,
            highlight_selectors=True,
        )
    )

    fetcher_config = captured["config"]
    assert isinstance(fetcher_config, FetcherConfig)
    assert fetcher_config.debug_browser is True
    assert fetcher_config.slow_mo_ms == 150
    assert fetcher_config.pause_on_page is True
    assert fetcher_config.highlight_selectors is True
    scraper.fetcher.close()


def test_playwright_fetcher_uses_headed_launch_args_in_debug_mode(monkeypatch) -> None:
    launch_calls: list[dict[str, object]] = []

    class _FakeContext:
        def close(self) -> None:
            return None

    class _FakeBrowser:
        def new_context(self, **kwargs):  # type: ignore[no-untyped-def]
            return _FakeContext()

        def close(self) -> None:
            return None

    class _FakeChromium:
        def launch(self, **kwargs):  # type: ignore[no-untyped-def]
            launch_calls.append(kwargs)
            return _FakeBrowser()

    class _FakePlaywright:
        def __init__(self) -> None:
            self.chromium = _FakeChromium()

        def stop(self) -> None:
            return None

    class _FakeManager:
        def start(self) -> _FakePlaywright:
            return _FakePlaywright()

    fake_playwright_package = types.ModuleType("playwright")
    fake_playwright_module = types.ModuleType("playwright.sync_api")
    fake_playwright_module.sync_playwright = lambda: _FakeManager()

    monkeypatch.setitem(sys.modules, "playwright", fake_playwright_package)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_playwright_module)

    fetcher = PlaywrightFetcher(FetcherConfig(debug_browser=True, slow_mo_ms=125))
    fetcher._ensure_started()

    assert launch_calls == [
        {
            "headless": False,
            "slow_mo": 125,
            "args": ["--no-proxy-server", "--disable-gpu"],
        }
    ]
    fetcher.close()
