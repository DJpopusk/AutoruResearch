"""Text parsing and normalization helpers."""

from __future__ import annotations

import re
import unicodedata
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


_SPACE_RE = re.compile(r"\s+")
_INT_RE = re.compile(r"(-?\d[\d\s]*)")
_FLOAT_RE = re.compile(r"(-?\d+[\.,]?\d*)")


def normalize_whitespace(text: str | None) -> str | None:
    """Collapse repeated whitespace and trim text."""
    if text is None:
        return None
    cleaned = _SPACE_RE.sub(" ", text).strip()
    return cleaned or None


def slugify_label(label: str) -> str:
    """Convert arbitrary label to normalized key."""
    text = unicodedata.normalize("NFKD", label.lower())
    text = re.sub(r"[^a-zа-я0-9]+", "_", text)
    return text.strip("_")


def extract_int(text: str | None) -> int | None:
    """Extract integer from text; return None if unavailable."""
    if not text:
        return None
    match = _INT_RE.search(text.replace("\xa0", " "))
    if not match:
        return None
    return int(match.group(1).replace(" ", ""))


def extract_float(text: str | None) -> float | None:
    """Extract float from text; return None if unavailable."""
    if not text:
        return None
    match = _FLOAT_RE.search(text.replace("\xa0", " "))
    if not match:
        return None
    return float(match.group(1).replace(",", "."))


def update_query_param(url: str, key: str, value: Any) -> str:
    """Set or replace query parameter in URL."""
    parsed = urlparse(url)
    params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    params[key] = str(value)
    return urlunparse(parsed._replace(query=urlencode(params)))


def parse_engine_text(engine_text: str | None) -> dict[str, str | float | int | None]:
    """Extract engine volume, power, and fuel type from a compact engine string."""
    if not engine_text:
        return {"engine_volume": None, "engine_power_hp": None, "fuel_type": None}

    lower = engine_text.lower()
    volume = None
    power = None

    volume_match = re.search(r"(\d+[\.,]?\d*)\s*л(?![\.\w])", lower)
    if volume_match:
        volume = float(volume_match.group(1).replace(",", "."))

    power_match = re.search(r"(\d+)\s*л\.?с", lower)
    if power_match:
        power = int(power_match.group(1))

    fuel = None
    for candidate in ("бензин", "дизель", "электро", "гибрид", "газ"):
        if candidate in lower:
            fuel = candidate
            break

    return {"engine_volume": volume, "engine_power_hp": power, "fuel_type": fuel}


def to_bool_or_none(value: str | None) -> bool | None:
    """Map common ru values to bool, otherwise None."""
    if not value:
        return None
    lower = value.lower()
    truthy = ("да", "есть", "растаможен", "в наличии")
    falsy = ("нет", "не", "не растаможен")
    if any(token in lower for token in truthy):
        return True
    if any(token in lower for token in falsy):
        return False
    return None
