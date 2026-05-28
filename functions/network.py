"""Networking helpers with retry and rate limiting."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Any

import requests
from requests import Response
from tenacity import RetryCallState, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from functions.constants import DEFAULT_USER_AGENT

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RequestConfig:
    timeout_seconds: int = 30
    min_delay_seconds: float = 1.0
    max_delay_seconds: float = 2.5
    max_retries: int = 4
    user_agent: str = DEFAULT_USER_AGENT


class RateLimiter:
    def __init__(self, min_delay_seconds: float, max_delay_seconds: float) -> None:
        self.min_delay_seconds = min_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        self._last_call = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_call
        target = random.uniform(self.min_delay_seconds, self.max_delay_seconds)
        sleep_for = max(0.0, target - elapsed)
        if sleep_for > 0:
            time.sleep(sleep_for)
        self._last_call = time.monotonic()


def _before_retry(state: RetryCallState) -> None:
    if state.outcome:
        LOGGER.warning("Retrying after error: %s", state.outcome.exception())


class HttpClient:
    def __init__(self, config: RequestConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": config.user_agent,
                "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
            }
        )
        self.rate_limiter = RateLimiter(
            min_delay_seconds=config.min_delay_seconds,
            max_delay_seconds=config.max_delay_seconds,
        )

    @retry(
        reraise=True,
        retry=retry_if_exception_type((requests.RequestException, ValueError)),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        stop=stop_after_attempt(5),
        before_sleep=_before_retry,
    )
    def get(self, url: str, **kwargs: Any) -> Response:
        self.rate_limiter.wait()
        response = self.session.get(url, timeout=self.config.timeout_seconds, **kwargs)
        if response.status_code >= 500:
            raise ValueError(f"Server error {response.status_code} for {url}")
        if response.status_code in (403, 429):
            raise ValueError(f"Rate limited ({response.status_code}) for {url}")
        response.raise_for_status()
        return response
