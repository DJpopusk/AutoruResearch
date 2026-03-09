"""Auto.ru scraping orchestration with checkpointing and resume support."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.parser.extractors import canonicalize_columns, extract_listing_record, parse_listing_links
from src.parser.fetchers import FetcherConfig, PlaywrightFetcher, RequestsFetcher
from src.utils.constants import AUTORU_SCHEMA
from src.utils.io_utils import ensure_dir, load_tabular, save_dataframe, write_json
from src.utils.text_utils import update_query_param

LOGGER = logging.getLogger(__name__)

CHALLENGE_MARKERS = (
    "подтвердите, что запросы отправляли вы",
    "проверка безопасности",
    "подтвердите, что вы не робот",
    "нажмите в центр",
    "smartcaptcha",
    "captcha",
    "access denied",
    "robot",
)

LISTING_HTML_MARKERS = (
    "/cars/used/sale/",
    "/cars/new/group/",
    "listingitem",
    "bulls-list",
)


@dataclass(slots=True)
class ScrapeConfig:
    """Configuration for scraping execution."""

    catalog_url: str
    pages: int
    output_csv: Path
    output_parquet: Path
    state_file: Path
    checkpoint_jsonl: Path
    start_page: int | None = None
    use_playwright: bool = False
    timeout_seconds: int = 30
    min_delay_seconds: float = 1.2
    max_delay_seconds: float = 3.0
    condition_filter: str = "all"
    seller_type_filter: str = "all"
    min_mileage: int | None = None
    challenge_cooldown_seconds: int = 90
    max_consecutive_challenge_pages: int = 2
    challenge_retries_per_page: int = 1


class AutoRuScraper:
    """Scraper for Auto.ru listing pages and car cards."""

    def __init__(self, config: ScrapeConfig) -> None:
        self.config = config
        fetcher_config = FetcherConfig(
            timeout_seconds=config.timeout_seconds,
            min_delay_seconds=config.min_delay_seconds,
            max_delay_seconds=config.max_delay_seconds,
        )

        self.fetcher = PlaywrightFetcher(fetcher_config) if config.use_playwright else RequestsFetcher(fetcher_config)

        ensure_dir(config.output_csv.parent)
        ensure_dir(config.state_file.parent)

        self.records: list[dict[str, object | None]] = []
        self.processed_urls: set[str] = set()
        self.failed_urls: set[str] = set()
        self.last_catalog_page: int = 0

    def _state_signature(self) -> dict[str, object]:
        """Return parameters that define resume-compatibility of the current crawl."""
        return {
            "catalog_url": self.config.catalog_url,
            "condition_filter": self.config.condition_filter,
            "seller_type_filter": self.config.seller_type_filter,
            "min_mileage": self.config.min_mileage,
            "use_playwright": self.config.use_playwright,
        }

    @staticmethod
    def _looks_like_challenge_page(html: str) -> bool:
        """Detect anti-bot / challenge pages so scraper can back off safely."""
        lower = html.lower()
        marker_hits = sum(marker in lower for marker in CHALLENGE_MARKERS)
        has_listings = any(marker in lower for marker in LISTING_HTML_MARKERS)
        return marker_hits >= 1 and not has_listings

    @staticmethod
    def _looks_like_empty_catalog_page(html: str) -> bool:
        """Detect pages without listing markup and without explicit challenge markers."""
        lower = html.lower()
        has_listings = any(marker in lower for marker in LISTING_HTML_MARKERS)
        if has_listings:
            return False
        return "listingitem" not in lower and "/cars/" not in lower

    def _handle_challenge_backoff(self, page_number: int, url: str, attempt: int) -> None:
        """Sleep before retrying or aborting after suspected anti-bot response."""
        cooldown = max(1, self.config.challenge_cooldown_seconds)
        LOGGER.warning(
            "Possible anti-bot/challenge page detected on catalog page %s (attempt %s). "
            "Sleeping %s seconds before retry.",
            page_number,
            attempt,
            cooldown,
        )
        time.sleep(cooldown)

    def __enter__(self) -> "AutoRuScraper":
        self._load_resume_state()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        self._save_state()
        self.fetcher.close()

    def _load_resume_state(self) -> None:
        if self.config.output_csv.exists():
            try:
                existing_df = load_tabular(self.config.output_csv)
                if "url" in existing_df.columns:
                    existing_urls = existing_df["url"].dropna().astype(str).unique().tolist()
                    self.processed_urls.update(existing_urls)
                    LOGGER.info("Loaded %s processed URLs from existing output.", len(existing_urls))
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed to load existing output for resume: %s", exc)

        if self.config.checkpoint_jsonl.exists():
            for line in self.config.checkpoint_jsonl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("url"):
                    self.records.append(canonicalize_columns(row))
                    self.processed_urls.add(str(row["url"]))
            LOGGER.info("Loaded %s records from checkpoint.", len(self.records))

        if self.config.state_file.exists():
            try:
                state = json.loads(self.config.state_file.read_text(encoding="utf-8"))
                stored_signature = state.get("signature")
                if stored_signature == self._state_signature():
                    self.processed_urls.update(state.get("processed_urls", []))
                    self.failed_urls.update(state.get("failed_urls", []))
                    self.last_catalog_page = int(state.get("last_catalog_page", 0))
                    LOGGER.info(
                        "Loaded state: processed=%s failed=%s last_page=%s",
                        len(self.processed_urls),
                        len(self.failed_urls),
                        self.last_catalog_page,
                    )
                else:
                    LOGGER.info("State signature changed. Restarting catalog scan from page 1 for new filters/URL.")
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed to load state file: %s", exc)

    def _save_state(self) -> None:
        payload = {
            "signature": self._state_signature(),
            "processed_urls": sorted(self.processed_urls),
            "failed_urls": sorted(self.failed_urls),
            "last_catalog_page": self.last_catalog_page,
            "records_in_memory": len(self.records),
        }
        write_json(self.config.state_file, payload)

    def _url_matches_condition_filter(self, url: str) -> bool:
        """Check URL-level filter for new/used listings."""
        if self.config.condition_filter == "all":
            return True
        if self.config.condition_filter == "used":
            return "/cars/used/" in url
        if self.config.condition_filter == "new":
            return "/cars/new/" in url
        return True

    def _record_matches_filters(self, record: dict[str, object | None]) -> bool:
        """Check post-parse filters that require listing metadata."""
        if self.config.condition_filter != "all":
            condition = str(record.get("condition") or "").lower()
            if condition != self.config.condition_filter:
                return False

        if self.config.seller_type_filter != "all":
            seller_type = str(record.get("seller_type") or "").lower()
            if seller_type != self.config.seller_type_filter:
                return False

        if self.config.min_mileage is not None:
            mileage_value = record.get("mileage")
            mileage_numeric: float | None = None
            if mileage_value is not None:
                try:
                    mileage_numeric = float(mileage_value)
                except (TypeError, ValueError):
                    mileage_numeric = None
            if mileage_numeric is None or mileage_numeric < self.config.min_mileage:
                return False

        return True

    def _append_checkpoint(self, record: dict[str, object | None]) -> None:
        ensure_dir(self.config.checkpoint_jsonl.parent)
        with self.config.checkpoint_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _catalog_page_url(self, page_number: int) -> str:
        if page_number <= 1:
            return self.config.catalog_url
        return update_query_param(self.config.catalog_url, "page", page_number)

    def collect_listing_urls(self) -> list[str]:
        """Collect listing URLs from catalog pages."""
        unique_urls: set[str] = set()
        start_page = max(1, self.last_catalog_page + 1)
        if self.config.start_page is not None:
            start_page = max(1, self.config.start_page)
            LOGGER.info("Using explicit start page override: %s", start_page)
        consecutive_challenge_pages = 0

        # If the previous run collected nothing, resume state should not block a fresh retry.
        if (
            self.config.start_page is None
            and not self.processed_urls
            and not self.records
            and start_page > self.config.pages
        ):
            LOGGER.info("Resume state contains no parsed listings. Restarting catalog scan from page 1.")
            start_page = 1

        for page_number in range(start_page, self.config.pages + 1):
            page_url = self._catalog_page_url(page_number)
            LOGGER.info("Collecting listing links from page %s: %s", page_number, page_url)
            html = None
            links: list[str] = []
            challenge_detected = False

            for attempt in range(1, self.config.challenge_retries_per_page + 2):
                try:
                    html = self.fetcher.fetch_html(page_url)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.error("Failed to fetch catalog page %s: %s", page_url, exc)
                    break

                links = parse_listing_links(html, page_url)
                links = [url for url in links if self._url_matches_condition_filter(url)]
                if links:
                    challenge_detected = False
                    break

                challenge_detected = self._looks_like_challenge_page(html)
                if challenge_detected and attempt <= self.config.challenge_retries_per_page:
                    self._handle_challenge_backoff(page_number=page_number, url=page_url, attempt=attempt)
                    continue
                break

            if challenge_detected:
                consecutive_challenge_pages += 1
                LOGGER.warning(
                    "Catalog page %s looks like an anti-bot/challenge response. "
                    "Consecutive challenge pages: %s.",
                    page_number,
                    consecutive_challenge_pages,
                )
                if consecutive_challenge_pages >= self.config.max_consecutive_challenge_pages:
                    LOGGER.error(
                        "Stopping catalog scan after %s consecutive challenge pages. "
                        "Resume later or increase delays.",
                        consecutive_challenge_pages,
                    )
                    self._save_state()
                    break
            else:
                consecutive_challenge_pages = 0

            if not links:
                if html is not None and self._looks_like_empty_catalog_page(html):
                    LOGGER.warning(
                        "Catalog page %s returned no listing markup. This is likely a stub/empty response, "
                        "not a selector breakage.",
                        page_number,
                    )
                else:
                    LOGGER.warning("No listing links found on page %s. Selectors may need update.", page_number)

            unique_urls.update(links)
            self.last_catalog_page = page_number
            self._save_state()
            LOGGER.info("Page %s yielded %s links (unique total=%s).", page_number, len(links), len(unique_urls))

        return sorted(unique_urls)

    def scrape_listings(self, listing_urls: Iterable[str]) -> pd.DataFrame:
        """Scrape detail pages and return parsed dataframe."""
        scraped_now = 0
        consecutive_challenge_pages = 0
        listing_urls = list(listing_urls)
        total_candidates = len(listing_urls)
        for idx, url in enumerate(listing_urls, start=1):
            if url in self.processed_urls:
                continue

            LOGGER.info("[%s/%s] Parsing listing: %s", idx, total_candidates, url)
            try:
                html = self.fetcher.fetch_html(url)
                if self._looks_like_challenge_page(html):
                    consecutive_challenge_pages += 1
                    LOGGER.warning(
                        "Detail page looks like anti-bot/challenge response: %s (consecutive=%s)",
                        url,
                        consecutive_challenge_pages,
                    )
                    self.failed_urls.add(url)
                    if consecutive_challenge_pages >= self.config.max_consecutive_challenge_pages:
                        LOGGER.error(
                            "Stopping detail parsing after %s consecutive challenge pages. "
                            "Resume later or increase delays.",
                            consecutive_challenge_pages,
                        )
                        self._save_state()
                        break
                    time.sleep(max(1, self.config.challenge_cooldown_seconds))
                    continue

                consecutive_challenge_pages = 0
                record = canonicalize_columns(extract_listing_record(html, url=url))
                if not self._record_matches_filters(record):
                    self.processed_urls.add(url)
                    LOGGER.info(
                        "Skipping listing due to filters: condition=%s seller_type=%s url=%s",
                        record.get("condition"),
                        record.get("seller_type"),
                        url,
                    )
                    continue
                self.records.append(record)
                self.processed_urls.add(url)
                scraped_now += 1
                self._append_checkpoint(record)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Failed to parse listing %s: %s", url, exc)
                self.failed_urls.add(url)

            if idx % 20 == 0:
                self._save_state()

        LOGGER.info("Parsed %s new records in this run.", scraped_now)
        df = pd.DataFrame(self.records)
        if df.empty:
            return pd.DataFrame(columns=AUTORU_SCHEMA)

        for column in AUTORU_SCHEMA:
            if column not in df.columns:
                df[column] = None

        df = df[AUTORU_SCHEMA]
        df = df.drop_duplicates(subset=["url"], keep="last").reset_index(drop=True)
        return df

    def _records_dataframe(self) -> pd.DataFrame:
        """Build normalized dataframe from in-memory records."""
        df = pd.DataFrame(self.records)
        if df.empty:
            return pd.DataFrame(columns=AUTORU_SCHEMA)

        for column in AUTORU_SCHEMA:
            if column not in df.columns:
                df[column] = None

        df = df[AUTORU_SCHEMA]
        return df.drop_duplicates(subset=["url"], keep="last").reset_index(drop=True)

    def run(self) -> pd.DataFrame:
        """Execute complete scraping workflow and persist outputs."""
        listing_urls = self.collect_listing_urls()
        if not listing_urls:
            output = self._records_dataframe()
            if not output.empty:
                LOGGER.info(
                    "No new listing URLs collected. Reusing %s records from checkpoint/output state.",
                    len(output),
                )
                save_dataframe(output, self.config.output_csv, self.config.output_parquet)
                self._save_state()
                return output

            LOGGER.warning("No listing URLs collected and no existing records were available.")
            output = pd.DataFrame(columns=AUTORU_SCHEMA)
            save_dataframe(output, self.config.output_csv, self.config.output_parquet)
            return output

        parsed_df = self.scrape_listings(listing_urls)

        if self.config.output_csv.exists():
            existing = load_tabular(self.config.output_csv)
            for column in AUTORU_SCHEMA:
                if column not in existing.columns:
                    existing[column] = None
            existing = existing[AUTORU_SCHEMA]
            parsed_df = pd.concat([existing, parsed_df], ignore_index=True)
            parsed_df = parsed_df.drop_duplicates(subset=["url"], keep="last").reset_index(drop=True)

        parsed_df = parsed_df[AUTORU_SCHEMA]
        save_dataframe(parsed_df, self.config.output_csv, self.config.output_parquet)
        self._save_state()

        LOGGER.info(
            "Saved dataset: %s rows, %s columns. CSV=%s PARQUET=%s",
            len(parsed_df),
            len(parsed_df.columns),
            self.config.output_csv,
            self.config.output_parquet,
        )
        return parsed_df


def run_scrape(config: ScrapeConfig) -> pd.DataFrame:
    """Convenience function to run scraper with context management."""
    with AutoRuScraper(config) as scraper:
        return scraper.run()
