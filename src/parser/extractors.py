"""Data extraction helpers for Auto.ru listing and detail pages."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Iterable
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from src.parser.selectors import (
    DETAIL_TEXT_SELECTORS,
    LISTING_LINK_SELECTORS,
    SPEC_KEY_SELECTORS,
    SPEC_ROW_SELECTORS,
    SPEC_VALUE_SELECTORS,
)
from src.utils.constants import AUTORU_SCHEMA
from src.utils.text_utils import (
    extract_int,
    extract_float,
    normalize_whitespace,
    parse_engine_text,
    slugify_label,
)

LOGGER = logging.getLogger(__name__)


SPEC_KEYWORDS = {
    "brand": ["марка"],
    "model": ["модель"],
    "generation": ["поколение"],
    "year": ["год", "год выпуска"],
    "mileage": ["пробег"],
    "body_type": ["кузов"],
    "color": ["цвет"],
    "engine": ["двигатель"],
    "transmission": ["коробка передач", "кпп"],
    "drive_type": ["привод"],
    "steering_wheel": ["руль"],
    "condition": ["состояние"],
    "owners_count": ["владельц"],
    "pts_type": ["птс"],
    "customs": ["растамож"],
    "region": ["регион"],
}

BODY_TYPES = (
    "внедорожник",
    "седан",
    "хэтчбек",
    "универсал",
    "лифтбек",
    "купе",
    "минивэн",
    "пикап",
    "кабриолет",
    "фургон",
    "кроссовер",
)

COLOR_NAMES = (
    "чёрный",
    "черный",
    "белый",
    "серый",
    "синий",
    "красный",
    "зеленый",
    "зелёный",
    "серебристый",
    "коричневый",
    "бежевый",
    "оранжевый",
    "жёлтый",
    "желтый",
    "голубой",
    "фиолетовый",
)


def _empty_record(url: str) -> dict[str, object | None]:
    record = {column: None for column in AUTORU_SCHEMA}
    record["url"] = url
    record["parsed_at"] = datetime.now(timezone.utc).isoformat()
    return record


def _first_text(soup: BeautifulSoup, selectors: Iterable[str]) -> str | None:
    for selector in selectors:
        element = soup.select_one(selector)
        if element is None:
            continue

        if element.name == "meta":
            text = element.get("content")
        else:
            text = element.get_text(" ", strip=True)

        cleaned = normalize_whitespace(text)
        if cleaned:
            return cleaned
    return None


def _first_meta_content(soup: BeautifulSoup, names: Iterable[str]) -> str | None:
    """Get content from meta tags by name/property."""
    for name in names:
        element = soup.find("meta", attrs={"name": name}) or soup.find("meta", attrs={"property": name})
        if element is None:
            continue
        content = normalize_whitespace(element.get("content"))
        if content:
            return content
    return None


def _extract_price_from_text(text: str | None) -> int | None:
    if not text:
        return None
    normalized = text.replace("\xa0", " ")
    match = re.search(r"(\d[\d\s]{3,})\s*₽", normalized)
    if not match:
        match = re.search(r"(\d[\d\s]{3,})", normalized)
    return int(match.group(1).replace(" ", "")) if match else None


def _extract_specs_map(soup: BeautifulSoup) -> dict[str, str]:
    specs: dict[str, str] = {}

    for row_selector in SPEC_ROW_SELECTORS:
        for row in soup.select(row_selector):
            key = None
            value = None

            for key_selector in SPEC_KEY_SELECTORS:
                key_el = row.select_one(key_selector)
                if key_el:
                    key = normalize_whitespace(key_el.get_text(" ", strip=True))
                    if key:
                        break

            for value_selector in SPEC_VALUE_SELECTORS:
                val_el = row.select_one(value_selector)
                if val_el:
                    value = normalize_whitespace(val_el.get_text(" ", strip=True))
                    if value:
                        break

            if key and value:
                specs[key.lower()] = value

    return specs


def _pick_spec(specs: dict[str, str], key: str) -> str | None:
    keywords = SPEC_KEYWORDS.get(key, [])
    for label, value in specs.items():
        if any(keyword in label for keyword in keywords):
            return value
    return None


def _extract_from_jsonld(soup: BeautifulSoup) -> dict[str, str | int | float | None]:
    output: dict[str, str | int | float | None] = {}
    for script in soup.select("script[type='application/ld+json']"):
        if not script.string:
            continue
        try:
            payload = json.loads(script.string)
        except json.JSONDecodeError:
            continue

        if isinstance(payload, list):
            items = payload
        else:
            items = [payload]

        for item in items:
            if not isinstance(item, dict):
                continue

            offers = item.get("offers", {})
            price = offers.get("price") if isinstance(offers, dict) else None
            if price is not None:
                output["price"] = extract_int(str(price))

            brand = item.get("brand")
            if isinstance(brand, dict):
                brand = brand.get("name")
            if isinstance(brand, str) and brand:
                output["brand"] = brand

            model = item.get("model")
            if isinstance(model, str) and model:
                output["model"] = model

    return output


def _extract_summary_fields(text: str | None, source_url: str) -> dict[str, object | None]:
    """Extract common listing attributes from title/meta summary text."""
    result: dict[str, object | None] = {}
    if not text:
        return result

    normalized = normalize_whitespace(text)
    if not normalized:
        return result

    lower = normalized.lower()

    if "новый" in lower or "/cars/new/" in source_url:
        result["condition"] = "new"
    elif "подержан" in lower or "б/у" in lower or "/cars/used/" in source_url:
        result["condition"] = "used"

    years = re.findall(r"(19\d{2}|20\d{2})(?=\s*года)", lower)
    if years:
        result["year"] = int(years[-1])

    mileage_match = re.search(r"пробег\s+(\d[\d\s\xa0]*)\s*км", lower)
    if mileage_match:
        result["mileage"] = extract_int(mileage_match.group(1))

    engine_text = normalized
    engine_parts = parse_engine_text(engine_text)
    result.update({k: v for k, v in engine_parts.items() if v is not None})

    if result.get("engine_volume") is None:
        volume_match = re.search(r"\b(\d+(?:[.,]\d+)?)\s*(?:at|mt|cvt|dct)\b", lower)
        if volume_match:
            result["engine_volume"] = extract_float(volume_match.group(1))

    if result.get("engine_power_hp") is None:
        kw_match = re.search(r"(\d+(?:[.,]\d+)?)\s*квт", lower)
        if kw_match:
            kw_value = extract_float(kw_match.group(1))
            if kw_value is not None:
                result["engine_power_hp"] = int(round(kw_value * 1.35962))

    if "автомат" in lower:
        result["transmission"] = "автомат"
    elif "механ" in lower:
        result["transmission"] = "механика"
    elif "вариатор" in lower:
        result["transmission"] = "вариатор"
    elif "робот" in lower:
        result["transmission"] = "робот"

    if "4wd" in lower or "awd" in lower or "полный" in lower:
        result["drive_type"] = "полный"
    elif "fwd" in lower or "передний" in lower:
        result["drive_type"] = "передний"
    elif "rwd" in lower or "задний" in lower:
        result["drive_type"] = "задний"

    for body_type in BODY_TYPES:
        if body_type in lower:
            result["body_type"] = body_type
            break

    for color in COLOR_NAMES:
        if color in lower:
            result["color"] = color
            break

    if "от дилера" in lower:
        result["seller_type"] = "dealer"
    elif "частных лиц" in lower or "частного лица" in lower:
        result["seller_type"] = "private"

    return result


def _is_invalid_record(record: dict[str, object | None], title: str | None) -> bool:
    """Detect anti-bot/garbled responses and obviously broken parsed records."""
    title_text = title or ""
    brand = str(record.get("brand") or "")
    price = record.get("price")

    if "Авто.ру: купить, продать и обменять машину" in title_text:
        return True
    if "Ð" in title_text or "Ñ" in title_text or "Ð" in brand or "Ñ" in brand:
        return True
    if isinstance(price, (int, float)) and price > 1_000_000_000:
        return True
    return False


def _extract_listing_links_from_jsonld(soup: BeautifulSoup) -> list[str]:
    """Extract listing URLs from schema.org JSON-LD blocks on catalog pages."""
    links: set[str] = set()

    for script in soup.select("script[type='application/ld+json']"):
        if not script.string:
            continue

        try:
            payload = json.loads(script.string)
        except json.JSONDecodeError:
            continue

        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue

            offers = item.get("offers")
            if not isinstance(offers, dict):
                continue

            nested_offers = offers.get("offers")
            if not isinstance(nested_offers, list):
                continue

            for offer in nested_offers:
                if not isinstance(offer, dict):
                    continue
                url = offer.get("url")
                if isinstance(url, str) and "/cars/" in url and ("/sale/" in url or "/cars/new/" in url):
                    links.add(url.split("?")[0])

    return sorted(links)


def parse_listing_links(html: str, page_url: str) -> list[str]:
    """Extract unique listing URLs from search page HTML."""
    soup = BeautifulSoup(html, "lxml")
    parsed_page = urlparse(page_url)
    base_domain = f"{parsed_page.scheme}://{parsed_page.netloc}"

    links: set[str] = set()

    for selector in LISTING_LINK_SELECTORS:
        for anchor in soup.select(selector):
            href = anchor.get("href")
            if not href:
                continue
            absolute = urljoin(base_domain, href)
            if "/cars/" in absolute and ("/sale/" in absolute or "/new/" in absolute):
                links.add(absolute.split("?")[0])

    if not links:
        links.update(_extract_listing_links_from_jsonld(soup))

    return sorted(links)


def extract_listing_record(html: str, url: str) -> dict[str, object | None]:
    """Parse single car detail page into unified schema record."""
    soup = BeautifulSoup(html, "lxml")
    record = _empty_record(url)

    jsonld_data = _extract_from_jsonld(soup)
    record.update({k: v for k, v in jsonld_data.items() if k in record and v is not None})

    title = _first_text(soup, DETAIL_TEXT_SELECTORS["title"])
    if title and (record["brand"] is None or record["model"] is None):
        title_parts = title.split()
        if len(title_parts) >= 2:
            record["brand"] = record["brand"] or title_parts[0].strip(",")
            record["model"] = record["model"] or title_parts[1].strip(",")
        if len(title_parts) >= 3 and record["generation"] is None:
            record["generation"] = " ".join(title_parts[2:5])

    meta_description = _first_meta_content(soup, ("description", "og:description"))
    if meta_description:
        summary_fields = _extract_summary_fields(meta_description, source_url=url)
        for key, value in summary_fields.items():
            if record.get(key) is None and value is not None:
                record[key] = value

    if title:
        title_fields = _extract_summary_fields(title, source_url=url)
        for key, value in title_fields.items():
            if record.get(key) is None and value is not None:
                record[key] = value

    price_text = _first_text(soup, DETAIL_TEXT_SELECTORS["price"])
    record["price"] = record["price"] or _extract_price_from_text(price_text)

    description = _first_text(soup, DETAIL_TEXT_SELECTORS["description"])
    if description:
        record["description_text"] = description

    region = _first_text(soup, DETAIL_TEXT_SELECTORS["region"])
    if region:
        record["region"] = region

    seller_badge = _first_text(soup, DETAIL_TEXT_SELECTORS["seller_badge"])
    if seller_badge:
        badge_lower = seller_badge.lower()
        if "дилер" in badge_lower:
            record["seller_type"] = "dealer"
        elif "част" in badge_lower:
            record["seller_type"] = "private"
        else:
            record["seller_type"] = seller_badge

    specs = _extract_specs_map(soup)

    for field in [
        "brand",
        "model",
        "generation",
        "body_type",
        "color",
        "transmission",
        "drive_type",
        "steering_wheel",
        "condition",
        "pts_type",
        "customs",
        "region",
    ]:
        if record[field] is None:
            record[field] = _pick_spec(specs, field)

    year_text = _pick_spec(specs, "year")
    mileage_text = _pick_spec(specs, "mileage")
    owners_text = _pick_spec(specs, "owners_count")
    engine_text = _pick_spec(specs, "engine")

    if year_text:
        record["year"] = extract_int(year_text)
    if mileage_text:
        record["mileage"] = extract_int(mileage_text)
    if owners_text:
        record["owners_count"] = extract_int(owners_text)
    if engine_text:
        engine_parts = parse_engine_text(engine_text)
        for key, value in engine_parts.items():
            if record.get(key) is None:
                record[key] = value

    # Fallback extraction from generic page text when price or mileage is missing.
    page_text = soup.get_text(" ", strip=True)
    if record["price"] is None:
        record["price"] = _extract_price_from_text(page_text)
    if record["mileage"] is None and "км" in page_text.lower():
        mileage_match = re.search(r"(\d[\d\s]{2,})\s*км", page_text.lower())
        if mileage_match:
            mileage_text = mileage_match.group(1).replace("\xa0", " ").replace(" ", "")
            record["mileage"] = int(mileage_text)

    for field in ("brand", "model", "generation", "body_type", "fuel_type", "transmission", "drive_type"):
        value = record.get(field)
        if isinstance(value, str):
            record[field] = normalize_whitespace(value.strip(","))

    if _is_invalid_record(record, title=title):
        raise ValueError("Received invalid or anti-bot detail page instead of listing")

    return record


def canonicalize_columns(record: dict[str, object | None]) -> dict[str, object | None]:
    """Return record containing only schema keys in fixed order."""
    output: dict[str, object | None] = {}
    for col in AUTORU_SCHEMA:
        output[col] = record.get(col)
    return output


def extract_spec_pairs_from_html(html: str) -> dict[str, str]:
    """Public helper for tests and debugging parser selectors."""
    soup = BeautifulSoup(html, "lxml")
    raw_specs = _extract_specs_map(soup)
    return {slugify_label(k): v for k, v in raw_specs.items()}
