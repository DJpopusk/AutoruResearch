"""HTML fetching backends for parser."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

from src.utils.network import HttpClient, RequestConfig
from src.utils.constants import DEFAULT_USER_AGENT

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class FetcherConfig:
    """Configuration shared across fetchers."""

    timeout_seconds: int = 30
    min_delay_seconds: float = 1.2
    max_delay_seconds: float = 3.0


class BaseFetcher:
    """Abstract fetcher interface."""

    def fetch_html(self, url: str) -> str:
        """Fetch page HTML."""
        raise NotImplementedError

    def close(self) -> None:
        """Release allocated resources."""


class RequestsFetcher(BaseFetcher):
    """requests-based fetcher with retry and rate limiting."""

    def __init__(self, config: FetcherConfig) -> None:
        self.config = config
        self.client = HttpClient(
            RequestConfig(
                timeout_seconds=config.timeout_seconds,
                min_delay_seconds=config.min_delay_seconds,
                max_delay_seconds=config.max_delay_seconds,
            )
        )

    @staticmethod
    def _looks_like_stub_page(html: str) -> bool:
        """Detect compact landing/anti-bot pages that do not contain listing data."""
        if len(html) > 20_000:
            return False

        markers = (
            "Авто.ру: купить, продать и обменять машину",
            "body {",
            "logo {",
        )
        return all(marker in html for marker in markers)

    def fetch_html(self, url: str) -> str:
        response = self.client.get(url)
        html = response.text
        if not self._looks_like_stub_page(html):
            return html

        LOGGER.warning("Received compact stub page for %s, retrying with fallback user-agent.", url)
        fallback_response = requests.get(
            url,
            headers={"User-Agent": DEFAULT_USER_AGENT},
            timeout=self.config.timeout_seconds,
        )
        fallback_response.raise_for_status()
        return fallback_response.text


class PlaywrightFetcher(BaseFetcher):
    """Playwright backend for dynamic pages."""

    def __init__(self, config: FetcherConfig) -> None:
        self.config = config
        self._playwright = None
        self._browser = None
        self._context = None

    def _ensure_started(self) -> None:
        if self._browser is not None:
            return

        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is not installed. Install dependencies and run `playwright install chromium`."
            ) from exc

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        self._context = self._browser.new_context(
            user_agent=DEFAULT_USER_AGENT,
            viewport={"width": 1440, "height": 900},
            locale="ru-RU",
        )

    def fetch_html(self, url: str) -> str:
        self._ensure_started()
        assert self._context is not None

        page = self._context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=self.config.timeout_seconds * 1000)
            page.wait_for_timeout(1200)
            return page.content()
        finally:
            page.close()

    def close(self) -> None:
        if self._context:
            self._context.close()
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()

        self._context = None
        self._browser = None
        self._playwright = None
