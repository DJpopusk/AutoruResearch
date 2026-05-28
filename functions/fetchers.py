"""HTML fetching backends for the scraper."""

from __future__ import annotations

import json
import logging
import platform
import re
import shlex
import subprocess
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path

import requests

from functions.constants import AUTORU_SCHEMA, DEFAULT_USER_AGENT
from functions.extractors import extract_listing_record
from functions.network import HttpClient, RequestConfig
from functions.selectors import (
    DETAIL_TEXT_SELECTORS,
    EXPAND_BUTTON_SELECTORS,
    EXPAND_BUTTON_TEXTS,
    LISTING_LINK_SELECTORS,
    SPEC_KEY_SELECTORS,
    SPEC_ROW_SELECTORS,
    SPEC_VALUE_SELECTORS,
)

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class FetcherConfig:
    timeout_seconds: int = 30
    min_delay_seconds: float = 1.2
    max_delay_seconds: float = 3.0
    debug_browser: bool = False
    headed: bool = False  # видимое окно без тяжёлого debug-монитора
    slow_mo_ms: int = 0
    pause_on_page: bool = False
    highlight_selectors: bool = False
    auto_advance_seconds: float = 0.0


class BaseFetcher:
    def fetch_html(self, url: str) -> str:
        raise NotImplementedError

    def close(self) -> None:
        return None


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
    """Playwright backend for dynamic pages (with optional debug monitor)."""

    def __init__(self, config: FetcherConfig) -> None:
        self.config = config
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._debug_logs: deque[str] = deque(maxlen=120)
        self._debug_status = "idle"
        self._monitor_opened = False
        self._monitor_snapshot_path = Path("logs") / "playwright_debug_monitor.txt"
        self._monitor_run_flag = Path("logs") / "playwright_debug_monitor.run"

    @staticmethod
    def _selector_groups() -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {"listing_links": list(LISTING_LINK_SELECTORS)}
        for field, selectors in DETAIL_TEXT_SELECTORS.items():
            groups[f"detail_{field}"] = list(selectors)
        groups["spec_rows"] = list(SPEC_ROW_SELECTORS)
        groups["spec_keys"] = list(SPEC_KEY_SELECTORS)
        groups["spec_values"] = list(SPEC_VALUE_SELECTORS)
        return groups

    @staticmethod
    def _highlight_color(group_name: str) -> str:
        palette = {
            "listing_links": "#6c5ce7",
            "detail_title": "#0057ff",
            "detail_price": "#ff5a5f",
            "detail_description": "#00a86b",
            "detail_region": "#ff8c00",
            "detail_seller_badge": "#9b59b6",
            "spec_rows": "#16a085",
            "spec_keys": "#f39c12",
            "spec_values": "#e84393",
        }
        return palette.get(group_name, "#2d3436")

    def _append_debug_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._debug_logs.append(f"[{timestamp}] {message}")

    @staticmethod
    def _build_debug_text(
        *,
        url: str,
        status: str,
        record: dict[str, object | None],
        selector_matches: list[dict[str, object]],
        logs: list[str],
    ) -> str:
        filled_fields = sum(value is not None for value in record.values())
        matched_groups = sum(1 for row in selector_matches if int(row["count"]) > 0)

        lines = [
            "AUTORU PARSER DEBUG",
            "=" * 72,
            f"URL: {url}",
            f"STATUS: {status}",
            f"FILLED_FIELDS: {filled_fields} / {len(AUTORU_SCHEMA)}",
            f"MATCHED_SELECTOR_GROUPS: {matched_groups} / {len(selector_matches)}",
            "",
            "EXTRACTED FIELDS",
            "-" * 72,
        ]

        for field in AUTORU_SCHEMA:
            value = record.get(field)
            value_text = "null" if value is None else str(value)
            lines.append(f"{field:24} {value_text}")

        lines.extend(["", "SELECTOR MATCHES", "-" * 72])
        for row in selector_matches:
            lines.append(f"[{row['group']}] count={row['count']:>3} selector={row['selector']}")

        lines.extend(["", "LIVE LOG", "-" * 72])
        lines.extend(logs or ["(no logs yet)"])
        return "\n".join(lines)

    def _render_debug_page(
        self,
        *,
        url: str,
        status: str,
        record: dict[str, object | None] | None = None,
        selector_matches: list[dict[str, object]] | None = None,
    ) -> None:
        if not self.config.debug_browser:
            return
        self._ensure_debug_page()
        record_to_render = record or {field: None for field in AUTORU_SCHEMA}
        record_to_render.setdefault("url", url)
        selector_matches_to_render = selector_matches or []
        self._debug_status = status
        self._monitor_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self._monitor_snapshot_path.write_text(
            self._build_debug_text(
                url=url,
                status=status,
                record=record_to_render,
                selector_matches=selector_matches_to_render,
                logs=list(self._debug_logs),
            ),
            encoding="utf-8",
        )

    def _ensure_debug_page(self) -> None:
        if self._monitor_opened:
            return
        self._monitor_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self._monitor_run_flag.write_text("running\n", encoding="utf-8")
        self._append_debug_log("Debug monitor started")
        self._monitor_snapshot_path.write_text(
            self._build_debug_text(
                url="about:blank",
                status="monitor_ready",
                record={field: None for field in AUTORU_SCHEMA},
                selector_matches=[],
                logs=list(self._debug_logs),
            ),
            encoding="utf-8",
        )

        if platform.system() == "Darwin":
            monitor_file = shlex.quote(str(self._monitor_snapshot_path.resolve()))
            run_flag = shlex.quote(str(self._monitor_run_flag.resolve()))
            command = (
                "bash -lc 'printf \"\\033]0;AUTORU DEBUG\\007\"; "
                f"while [ -f {run_flag} ]; do clear; "
                f"cat {monitor_file} 2>/dev/null || echo \"waiting for parser snapshot...\"; "
                "sleep 0.35; done'"
            )
            script = (
                'tell application "Terminal"\n'
                "activate\n"
                f"do script {json.dumps(command)}\n"
                "delay 0.2\n"
                "try\n"
                "set bounds of front window to {1120, 640, 1860, 1190}\n"
                "end try\n"
                "end tell\n"
            )
            try:
                subprocess.run(["osascript", "-e", script], check=False)
            except Exception:  # noqa: BLE001
                LOGGER.debug("Failed to open Terminal debug monitor.", exc_info=True)

        self._monitor_opened = True

    def _highlight_selectors(self, page) -> list[dict[str, object]]:  # type: ignore[no-untyped-def]
        matches: list[dict[str, object]] = []
        for group_name, selectors in self._selector_groups().items():
            color = self._highlight_color(group_name)
            for selector in selectors:
                count = 0
                try:
                    locator = page.locator(selector)
                    count = locator.count()
                    if count > 0:
                        locator.evaluate_all(
                            """
                            (elements, payload) => {
                                for (const element of elements) {
                                    if (!(element instanceof HTMLElement)) continue;
                                    element.style.outline = `3px solid ${payload.color}`;
                                    element.style.outlineOffset = '2px';
                                    element.style.boxShadow = `0 0 0 2px rgba(255,255,255,0.9), 0 0 0 5px ${payload.color}33`;
                                }
                            }
                            """,
                            {"color": color, "label": f"{group_name}: {selector}"},
                        )
                except Exception:  # noqa: BLE001
                    LOGGER.debug("Failed to highlight selector %s", selector, exc_info=True)
                matches.append({"group": group_name, "selector": selector, "count": int(count)})
        return matches

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
        launch_args: list[str] = []
        if self.config.debug_browser:
            launch_args.extend(["--no-proxy-server", "--disable-gpu"])
            self._append_debug_log("Launching headed Chromium")

        self._browser = self._playwright.chromium.launch(
            headless=not (self.config.debug_browser or self.config.headed),
            slow_mo=self.config.slow_mo_ms,
            args=launch_args,
        )
        self._context = self._browser.new_context(
            user_agent=DEFAULT_USER_AGENT,
            viewport={"width": 1440, "height": 900},
            locale="ru-RU",
        )
        self._page = self._context.new_page()
        if self.config.debug_browser:
            self._ensure_debug_page()

    def fetch_html(self, url: str) -> str:
        self._ensure_started()
        assert self._page is not None
        page = self._page
        LOGGER.info("Opening page in Chromium: %s", url)
        self._append_debug_log(f"Open page: {url}")
        self._render_debug_page(url=url, status="opening_page")

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=self.config.timeout_seconds * 1000)
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if "ERR_SOCKET_NOT_CONNECTED" in str(exc) and attempt == 0:
                    self._append_debug_log("Socket error in goto; retrying once")
                    page.wait_for_timeout(900)
                    continue
                raise

        if last_error is not None:
            raise last_error

        # В headed-режиме даём странице догрузиться: иначе виджеты (в том числе
        # Яндекс-капча) ещё не отрисованы и не реагируют на клики мышью.
        if self.config.headed or self.config.debug_browser:
            try:
                page.wait_for_load_state("networkidle", timeout=8000)
            except Exception:  # noqa: BLE001
                pass  # фоновая аналитика мешает networkidle — это ок
        else:
            page.wait_for_timeout(1200)

        # На детальных страницах: сначала фиксируем HTML карточки (там offer-only
        # поля — VIN, владельцы, регион, цвет, год…), потом кликаем
        # «Все характеристики». Если клик увёл на /specifications/ — склеиваем
        # оба HTML, чтобы парсеру достались поля сразу с обеих страниц.
        if "/sale/" in url:
            offer_html = page.content()
            url_before = page.url

            expanded = self._try_expand_sections(page)

            if expanded:
                LOGGER.info("Развёрнуто секций на карточке: %s", expanded)
                try:
                    page.wait_for_load_state("networkidle", timeout=6000)
                except Exception:  # noqa: BLE001
                    pass
                url_after = page.url
                second_html = page.content()

                if url_after != url_before:
                    LOGGER.info("Доп. страница характеристик: %s", url_after)
                    html = self._merge_html(offer_html, second_html)
                else:
                    # Разворот случился inline — берём обновлённый HTML карточки
                    html = second_html
            else:
                html = offer_html
        else:
            html = page.content()

        selector_matches: list[dict[str, object]] = []
        if self.config.highlight_selectors or self.config.debug_browser:
            selector_matches = self._highlight_selectors(page)
            self._render_debug_page(url=url, status="selectors_highlighted", selector_matches=selector_matches)

        record: dict[str, object | None] | None = None
        if self.config.debug_browser:
            try:
                record = extract_listing_record(html, url=url)
            except Exception as exc:  # noqa: BLE001
                record = {field: None for field in AUTORU_SCHEMA}
                record["url"] = url
                record["parsed_at"] = f"debug_error: {exc}"
            self._render_debug_page(
                url=url,
                status="record_extracted",
                record=record,
                selector_matches=selector_matches,
            )

        if self.config.auto_advance_seconds > 0:
            page.wait_for_timeout(int(self.config.auto_advance_seconds * 1000))
        elif self.config.pause_on_page:
            page.pause()

        self._render_debug_page(
            url=url,
            status="page_done",
            record=record,
            selector_matches=selector_matches,
        )
        return html

    def _try_expand_sections(self, page) -> int:  # type: ignore[no-untyped-def]
        """Прокликать «Все характеристики» / «Показать ещё» и т.п.

        Возвращает количество успешных кликов. Все ошибки глотаем — на каталоге
        и на нестандартных карточках кнопок может не быть, и это норма.
        """
        clicked = 0

        for text in EXPAND_BUTTON_TEXTS:
            try:
                locator = page.get_by_text(text, exact=False)
                count = locator.count()
            except Exception:  # noqa: BLE001
                continue
            for i in range(min(count, 5)):
                try:
                    element = locator.nth(i)
                    if not element.is_visible(timeout=400):
                        continue
                    element.scroll_into_view_if_needed(timeout=600)
                    element.click(timeout=1500)
                    clicked += 1
                    page.wait_for_timeout(300)
                except Exception:  # noqa: BLE001
                    continue

        for selector in EXPAND_BUTTON_SELECTORS:
            try:
                locator = page.locator(selector)
                count = locator.count()
            except Exception:  # noqa: BLE001
                continue
            for i in range(min(count, 5)):
                try:
                    element = locator.nth(i)
                    if not element.is_visible(timeout=400):
                        continue
                    element.scroll_into_view_if_needed(timeout=600)
                    element.click(timeout=1500)
                    clicked += 1
                    page.wait_for_timeout(300)
                except Exception:  # noqa: BLE001
                    continue

        if clicked:
            try:
                page.wait_for_load_state("networkidle", timeout=4000)
            except Exception:  # noqa: BLE001
                pass

        return clicked

    @staticmethod
    def _merge_html(primary_html: str, secondary_html: str) -> str:
        """Склеить тела двух HTML-документов в один валидный документ.

        Берётся `<body>` каждого и оба содержимых вставляются последовательно
        внутрь одного `<body>`. Так BeautifulSoup пройдётся по всем spec-строкам
        и offer-страницы, и страницы /specifications/.
        """
        def extract_body(html: str) -> tuple[str, str]:
            head_match = re.search(r"<head[^>]*>(.*?)</head>", html, re.DOTALL | re.IGNORECASE)
            body_match = re.search(r"<body[^>]*>(.*?)</body>", html, re.DOTALL | re.IGNORECASE)
            head = head_match.group(1) if head_match else ""
            body = body_match.group(1) if body_match else html
            return head, body

        head1, body1 = extract_body(primary_html)
        head2, body2 = extract_body(secondary_html)
        return (
            "<!doctype html><html><head>"
            f"{head1}{head2}"
            "</head><body>"
            f"{body1}"
            "<!-- ===== specs page below ===== -->"
            f"{body2}"
            "</body></html>"
        )

    def bring_to_front(self) -> None:
        """Поднять окно браузера в фокус, чтобы пользователь мог по нему кликнуть."""
        if self._page is None:
            return
        try:
            self._page.bring_to_front()
        except Exception:  # noqa: BLE001
            LOGGER.debug("bring_to_front не сработал", exc_info=True)

    def wait_until_idle(self, timeout_ms: int = 8000) -> None:
        """Дать странице догрузиться (сеть успокоилась), чтобы капча/виджеты стали кликабельны."""
        if self._page is None:
            return
        try:
            self._page.wait_for_load_state("networkidle", timeout=timeout_ms)
        except Exception:  # noqa: BLE001
            # networkidle может не наступить из-за фоновой аналитики — это нормально
            pass

    def current_html(self) -> str:
        """Свежий HTML из открытой страницы без повторной навигации."""
        if self._page is None:
            return ""
        return self._page.content()

    def save_page_html(self, path: Path) -> None:
        """Сохранить текущий HTML страницы — нужно, чтобы потом подкручивать селекторы."""
        if self._page is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._page.content(), encoding="utf-8")

    def close(self) -> None:
        if self._monitor_run_flag.exists():
            self._monitor_run_flag.unlink()
        if self._page:
            self._page.close()
        if self._context:
            self._context.close()
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()

        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
        self._monitor_opened = False
