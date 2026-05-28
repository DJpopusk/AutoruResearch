"""Auto.ru scraping orchestration with checkpointing and resume support."""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from functions.constants import AUTORU_SCHEMA
from functions.extractors import canonicalize_columns, extract_listing_record, parse_listing_links
from functions.fetchers import FetcherConfig, PlaywrightFetcher, RequestsFetcher
from functions.io import ensure_dir, load_tabular, save_dataframe, write_json
from functions.text import update_query_param

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
    debug_browser: bool = False
    slow_mo_ms: int = 0
    pause_on_page: bool = False
    highlight_selectors: bool = False
    auto_advance_seconds: float = 0.0
    interactive_confirm: bool = False  # видимый браузер + пауза Enter после каждой карточки


class AutoRuScraper:
    """Scraper for Auto.ru listing pages and car cards."""

    def __init__(self, config: ScrapeConfig) -> None:
        self.config = config

        # interactive_confirm требует видимого браузера, но не тяжёлого debug-монитора
        use_playwright = config.use_playwright or config.interactive_confirm
        headed = config.interactive_confirm
        auto_advance = 0.0 if config.interactive_confirm else config.auto_advance_seconds

        fetcher_config = FetcherConfig(
            timeout_seconds=config.timeout_seconds,
            min_delay_seconds=config.min_delay_seconds,
            max_delay_seconds=config.max_delay_seconds,
            debug_browser=config.debug_browser,
            headed=headed,
            slow_mo_ms=config.slow_mo_ms,
            pause_on_page=config.pause_on_page,
            highlight_selectors=config.highlight_selectors,
            auto_advance_seconds=auto_advance,
        )

        self.fetcher = PlaywrightFetcher(fetcher_config) if use_playwright else RequestsFetcher(fetcher_config)

        ensure_dir(config.output_csv.parent)
        ensure_dir(config.state_file.parent)

        self.records: list[dict[str, object | None]] = []
        self.processed_urls: set[str] = set()
        self.failed_urls: set[str] = set()
        self.last_catalog_page: int = 0

    def _state_signature(self) -> dict[str, object]:
        return {
            "catalog_url": self.config.catalog_url,
            "condition_filter": self.config.condition_filter,
            "seller_type_filter": self.config.seller_type_filter,
            "min_mileage": self.config.min_mileage,
            "use_playwright": self.config.use_playwright,
        }

    @staticmethod
    def _looks_like_challenge_page(html: str) -> bool:
        lower = html.lower()
        marker_hits = sum(marker in lower for marker in CHALLENGE_MARKERS)
        has_listings = any(marker in lower for marker in LISTING_HTML_MARKERS)
        return marker_hits >= 1 and not has_listings

    @staticmethod
    def _looks_like_empty_catalog_page(html: str) -> bool:
        lower = html.lower()
        has_listings = any(marker in lower for marker in LISTING_HTML_MARKERS)
        if has_listings:
            return False
        return "listingitem" not in lower and "/cars/" not in lower

    def _handle_challenge_backoff(self, page_number: int, url: str, attempt: int) -> None:
        cooldown = max(1, self.config.challenge_cooldown_seconds)
        LOGGER.warning(
            "Possible anti-bot page detected on catalog page %s (attempt %s). Sleeping %s seconds.",
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
                else:
                    LOGGER.info("State signature changed. Restarting catalog scan from page 1.")
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
        if self.config.condition_filter == "all":
            return True
        if self.config.condition_filter == "used":
            return "/cars/used/" in url
        if self.config.condition_filter == "new":
            return "/cars/new/" in url
        return True

    def _record_matches_filters(self, record: dict[str, object | None]) -> bool:
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

    def _wait_for_manual_captcha(self, url: str, max_attempts: int = 5) -> str | None:
        """Пауза: пользователь руками проходит капчу в окне, потом жмёт Enter.

        Поднимает окно в фокус, ждёт догрузки страницы (чтобы виджет капчи стал
        кликабельным), читает свежий HTML без повторной навигации.
        Возвращает свежий HTML после решения или None, если пользователь сдался.
        """
        playwright_fetcher = getattr(self.fetcher, "bring_to_front", None) is not None

        for attempt in range(1, max_attempts + 1):
            # Поднимаем окно и даём странице полностью прогрузиться, иначе клики
            # по капче не регистрируются (виджет Яндекса грузится JS-ом).
            if playwright_fetcher:
                self.fetcher.bring_to_front()
                self.fetcher.wait_until_idle(timeout_ms=8000)

            print(f"\n  ⚠️  Обнаружена капча на {url}")
            print("  → Окно браузера сейчас в фокусе, страница догружена.")
            print("  → Нажми 'Я не робот' в окне, дождись редиректа на нужную страницу.")
            print("  → Потом возвращайся в терминал и жми Enter.")
            try:
                answer = input(
                    f"  [Enter] капча пройдена · q — пропустить страницу "
                    f"(попытка {attempt}/{max_attempts}): "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                return None
            if answer in ("q", "quit", "skip"):
                return None

            # Не делаем goto заново — Яндекс снова покажет капчу.
            # Берём текущий HTML из окна, в котором пользователь уже всё решил.
            if playwright_fetcher:
                self.fetcher.wait_until_idle(timeout_ms=5000)
                new_html = self.fetcher.current_html()
            else:
                try:
                    new_html = self.fetcher.fetch_html(url)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.error("Перезапрос после капчи упал: %s", exc)
                    continue

            if not self._looks_like_challenge_page(new_html):
                LOGGER.info("Капча пройдена, страница загружена нормально.")
                return new_html

            print("  Похоже, капча всё ещё на месте. Попробуй ещё раз.")

        print("  Превышено число попыток для этой страницы — пропускаю.")
        return None

    def _resolve_captcha_if_any(self, url: str, html: str) -> str:
        """В интерактивном режиме даёт пользователю руками решить капчу.

        Если режим не интерактивный или капчи нет — возвращает html как есть.
        """
        if not self.config.interactive_confirm:
            return html
        if not self._looks_like_challenge_page(html):
            return html
        new_html = self._wait_for_manual_captcha(url)
        return new_html if new_html is not None else html

    @staticmethod
    def _missing_fields(record: dict[str, object | None]) -> list[str]:
        """Schema-поля, которые после парсинга пустые/None/NaN."""
        missing: list[str] = []
        for col in AUTORU_SCHEMA:
            value = record.get(col)
            if value is None:
                missing.append(col)
            elif isinstance(value, float) and value != value:  # NaN
                missing.append(col)
            elif isinstance(value, str) and not value.strip():
                missing.append(col)
        return missing

    def _dump_page_html_for_debug(self, url: str, idx: int) -> Path | None:
        """В интерактивном режиме сохраняем HTML каждой карточки рядом с датасетом.

        По этим дампам потом точно настраивать селекторы под текущую вёрстку Auto.ru.
        """
        saver = getattr(self.fetcher, "save_page_html", None)
        if saver is None:
            return None
        slug = url.rstrip("/").split("/")[-1] or f"page_{idx}"
        path = self.config.output_csv.parent / "page_dumps" / f"{idx:04d}_{slug}.html"
        try:
            saver(path)
            return path
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Не удалось сохранить HTML-дамп: %s", exc)
            return None

    def _interactive_confirm_record(
        self,
        url: str,
        record: dict[str, object | None],
        idx: int,
        total: int,
    ) -> str:
        """Печатает извлечённые/непрочитанные поля и ждёт ответа.

        Возвращает:
        - 'keep' — сохранить и спросить на следующей карточке;
        - 'skip' — не сохранять, спросить на следующей;
        - 'stop' — остановить парсер;
        - 'auto' — сохранить и дальше идти автоматически без вопросов.
        """
        missing = self._missing_fields(record)
        filled = [c for c in AUTORU_SCHEMA if c not in missing]

        print(f"\n[{idx}/{total}] {url}")
        print(f"  Извлечено  ({len(filled):2d}/{len(AUTORU_SCHEMA)}): " + ", ".join(filled))
        if missing:
            print(f"  НЕ извлечены ({len(missing):2d}): " + ", ".join(missing))
        else:
            print("  НЕ извлечены: —")

        dump_path = self._dump_page_html_for_debug(url, idx)
        if dump_path is not None:
            print(f"  HTML карточки: {dump_path}")

        while True:
            try:
                answer = input(
                    "  [Enter] продолжить · s — пропустить · a — автопилот · q — выход: "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                return "stop"
            if answer in ("", "y", "yes"):
                return "keep"
            if answer in ("s", "skip"):
                return "skip"
            if answer in ("q", "quit", "exit"):
                return "stop"
            if answer in ("a", "auto", "autopilot"):
                return "auto"
            print("  Не понял ответ. Введи Enter / s / a / q.")

    def _append_checkpoint(self, record: dict[str, object | None]) -> None:
        ensure_dir(self.config.checkpoint_jsonl.parent)
        with self.config.checkpoint_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _catalog_page_url(self, page_number: int) -> str:
        if page_number <= 1:
            return self.config.catalog_url
        return update_query_param(self.config.catalog_url, "page", page_number)

    def collect_listing_urls(self) -> list[str]:
        unique_urls: set[str] = set()
        start_page = max(1, self.last_catalog_page + 1)
        if self.config.start_page is not None:
            start_page = max(1, self.config.start_page)
        consecutive_challenge_pages = 0

        if (
            self.config.start_page is None
            and not self.processed_urls
            and not self.records
            and start_page > self.config.pages
        ):
            start_page = 1

        if start_page > self.config.pages:
            LOGGER.warning(
                "No catalog pages to scan: resolved start_page=%s while pages=%s.",
                start_page,
                self.config.pages,
            )
            return []

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

                html = self._resolve_captcha_if_any(page_url, html)

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
                if consecutive_challenge_pages >= self.config.max_consecutive_challenge_pages:
                    LOGGER.error("Stopping after %s consecutive challenge pages.", consecutive_challenge_pages)
                    self._save_state()
                    break
            else:
                consecutive_challenge_pages = 0

            unique_urls.update(links)
            self.last_catalog_page = page_number
            self._save_state()
            LOGGER.info("Page %s yielded %s links (unique total=%s).", page_number, len(links), len(unique_urls))

        # URL'ы содержат марку в пути (/cars/used/sale/<brand>/...).
        # При сортировке по алфавиту обход на ограниченном времени всегда
        # выгребал A→G и не доходил до Toyota/VW. Перемешиваем — детерминированно
        # внутри одного прогона (set ordering нестабилен).
        shuffled = sorted(unique_urls)
        random.Random(42).shuffle(shuffled)
        return shuffled

    # Опорные поля для проверки качества парса в автопилоте — у электрокаров
    # и ДВС разный набор «обязательных» характеристик. Если заполнено меньше
    # AUTOPILOT_MIN_RATIO от соответствующего списка — автопилот ставится на
    # паузу (вероятно анти-бот / капча / неполный рендер).
    AUTOPILOT_EV_CORE_FIELDS = (
        "price", "brand", "model",
        "battery_capacity_kwh", "electric_range_km", "max_charging_power_kw",
        "length", "width", "height", "wheelbase",
        "max_speed", "acceleration_0_100",
        "transmission", "drive_type", "fuel_type",
    )
    AUTOPILOT_ICE_CORE_FIELDS = (
        "price", "brand", "model",
        "engine_volume", "engine_power_hp", "fuel_type",
        "fuel_consumption_mixed", "fuel_tank_volume",
        "length", "width", "height", "wheelbase",
        "max_speed", "acceleration_0_100",
        "transmission", "drive_type",
    )
    AUTOPILOT_MIN_RATIO = 0.7  # 70% опорных полей должно быть заполнено

    @staticmethod
    def _is_electric(record: dict[str, object | None]) -> bool:
        """Эвристика «это электрокар»: по типу топлива или наличию батареи."""
        fuel_type = (record.get("fuel_type") or "")
        if isinstance(fuel_type, str) and "электро" in fuel_type.lower():
            return True
        if record.get("battery_capacity_kwh") is not None:
            return True
        return False

    def _autopilot_quality(
        self, record: dict[str, object | None]
    ) -> tuple[int, int, list[str], bool]:
        """Возвращает (filled, total, missing, is_ev) для опорных полей."""
        is_ev = self._is_electric(record)
        core = self.AUTOPILOT_EV_CORE_FIELDS if is_ev else self.AUTOPILOT_ICE_CORE_FIELDS
        missing = [f for f in core if record.get(f) is None]
        return len(core) - len(missing), len(core), missing, is_ev

    def scrape_listings(self, listing_urls: Iterable[str]) -> pd.DataFrame:
        scraped_now = 0
        consecutive_challenge_pages = 0
        # Когда пользователь нажал `a` в интерактивном режиме — переключаемся
        # в автопилот: всё, что дальше, сохраняется без вопросов. Парсер сам
        # выпадает из автопилота, если карточка выглядит подозрительно (мало
        # полей / анти-бот), и спрашивает пользователя.
        autopilot = False
        listing_urls = list(listing_urls)
        total_candidates = len(listing_urls)
        for idx, url in enumerate(listing_urls, start=1):
            if url in self.processed_urls:
                continue

            LOGGER.info("[%s/%s] Parsing listing: %s", idx, total_candidates, url)
            try:
                html = self.fetcher.fetch_html(url)
                html = self._resolve_captcha_if_any(url, html)
                if self._looks_like_challenge_page(html):
                    consecutive_challenge_pages += 1
                    self.failed_urls.add(url)
                    if consecutive_challenge_pages >= self.config.max_consecutive_challenge_pages:
                        LOGGER.error("Stopping detail parsing after %s consecutive challenge pages.", consecutive_challenge_pages)
                        self._save_state()
                        break
                    time.sleep(max(1, self.config.challenge_cooldown_seconds))
                    continue

                consecutive_challenge_pages = 0
                record = canonicalize_columns(extract_listing_record(html, url=url))
                if not self._record_matches_filters(record):
                    self.processed_urls.add(url)
                    continue

                # В автопилоте: для электрокара и ДВС используются разные опорные
                # наборы (у EV нет engine_volume/fuel_consumption_mixed, у ДВС
                # нет battery_capacity_kwh). Если заполнено <70% от соответствующего
                # списка — почти наверняка анти-бот / неполный рендер.
                if autopilot:
                    filled, total, missing, is_ev = self._autopilot_quality(record)
                    if filled / total < self.AUTOPILOT_MIN_RATIO:
                        kind = "электрокар" if is_ev else "ДВС"
                        print(
                            f"\n⚠️  Автопилот приостановлен ({kind}): заполнено "
                            f"{filled}/{total} опорных полей "
                            f"({filled / total:.0%}, порог {self.AUTOPILOT_MIN_RATIO:.0%})."
                        )
                        print(f"  Не извлечены: {', '.join(missing)}")
                        print(
                            "  Похоже на анти-бот / капчу. Реши её в окне браузера,"
                            " затем нажми Enter тут (или s/a/q)."
                        )
                        autopilot = False  # сбрасываем — пусть юзер решит явно

                if self.config.interactive_confirm and not autopilot:
                    action = self._interactive_confirm_record(url, record, idx, total_candidates)
                    if action == "stop":
                        LOGGER.info("Останов по запросу пользователя на %s", url)
                        self._save_state()
                        break
                    if action == "skip":
                        self.processed_urls.add(url)
                        continue
                    if action == "auto":
                        autopilot = True
                        LOGGER.info("Автопилот включён — дальше без подтверждений.")

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
        return self._records_dataframe()

    def _records_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.records)
        if df.empty:
            return pd.DataFrame(columns=AUTORU_SCHEMA)

        for column in AUTORU_SCHEMA:
            if column not in df.columns:
                df[column] = None

        df = df[AUTORU_SCHEMA]
        return df.drop_duplicates(subset=["url"], keep="last").reset_index(drop=True)

    def run(self) -> pd.DataFrame:
        listing_urls = self.collect_listing_urls()
        if not listing_urls:
            output = self._records_dataframe()
            if not output.empty:
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
    """Run the scraper end-to-end with context management."""
    with AutoRuScraper(config) as scraper:
        return scraper.run()
