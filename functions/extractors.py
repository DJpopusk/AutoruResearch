"""Data extraction helpers for Auto.ru listing and detail pages."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Iterable
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from functions.constants import AUTORU_SCHEMA
from functions.selectors import (
    DETAIL_TEXT_SELECTORS,
    LISTING_LINK_SELECTORS,
    SPEC_KEY_SELECTORS,
    SPEC_ROW_SELECTORS,
    SPEC_VALUE_SELECTORS,
)
from functions.text import (
    extract_float,
    extract_int,
    normalize_whitespace,
    parse_engine_text,
    slugify_label,
)

LOGGER = logging.getLogger(__name__)


SPEC_KEYWORDS = {
    # базовые
    "brand": ["марка"],
    "model": ["модель"],
    "generation": ["поколение"],
    "year": ["год выпуска", "год"],
    "mileage": ["пробег"],
    "body_type": ["кузов"],
    "color": ["цвет"],
    "engine": ["двигатель"],
    "transmission": ["коробка передач", "кпп"],
    "drive_type": ["тип привода", "привод"],
    "steering_wheel": ["расположение руля", "рул"],  # "рул" ловит и "руль" и "руля"
    # NB: condition хранит секцию рынка (new/used) и заполняется из URL/мета —
    # см. _extract_summary_fields. Спец-строка "Состояние" — это физическое
    # состояние машины ("Битый / не на ходу", "Не битый"), отдельное поле.
    "vehicle_state": ["состояние"],
    "owners_count": ["владельц"],
    "pts_type": ["птс"],
    "customs": ["таможн", "растамож"],  # label «Таможня», value «Растаможен»
    "region": ["регион"],
    # offer-only
    "vin": ["vin", "вин"],
    "complectation": ["комплектация"],
    "modification": ["модификация"],
    "tax_amount": ["транспортный налог", "налог"],
    "service_book": ["сервисная книжка", "сервисная"],
    "warranty_until": ["гарантия"],
    # геометрия и динамика
    "doors_count": ["количество дверей", "дверей"],
    "seats_count": ["количество мест", "мест"],
    "acceleration_0_100": ["разгон до 100", "разгон 0", "разгон"],
    "max_speed": ["максимальная скорость", "макс. скорость", "макс скорость"],
    "fuel_consumption_combo": [
        "расход топлива, город",  # "Расход топлива, город/трасса/смешанный"
        "расход топлива",
    ],
    "fuel_consumption_mixed": ["расход в смешан", "смешанный цикл", "смешан"],
    "eco_class": ["экологический класс", "эко класс", "евро"],
    "trunk_volume": ["объём багажника", "объем багажника", "багажник"],
    "fuel_tank_volume": ["объём топливного бака", "объем топливного бака", "топливн"],
    "clearance": ["клиренс", "дорожный просвет"],
    "wheelbase": ["колёсная база", "колесная база"],
    "length": ["длина"],
    "width": ["ширина"],
    "height": ["высота"],
    "weight_curb": ["снаряжённая масса", "снаряженная масса"],
    "weight_gross": ["полная масса"],
    "front_track_width": ["ширина передней колеи", "передн"],
    "rear_track_width": ["ширина задней колеи", "задн"],
    "wheel_size": ["размер колёс", "размер колес"],
    "bolt_pattern": ["сверловка"],
    "disc_size": ["размеры дисков", "размер дисков"],
    # двигатель и трансмиссия
    "engine_type_text": ["тип двигателя"],
    "engine_volume_cm3": ["объём двигателя", "объем двигателя"],
    "engine_power_max": ["максимальная мощность", "мощность"],
    "boost_type": ["тип наддува", "наддув"],
    "max_torque_nm": ["максимальный крутящий момент", "крутящий момент"],
    "cylinder_layout": ["расположение цилиндров"],
    "cylinders_count": ["количество цилиндров"],
    "valves_per_cylinder": ["число клапанов на цилиндр", "клапан"],
    "gears_count": ["количество передач"],
    "engine_position": ["расположение двигателя"],
    "fuel_brand": ["марка топлива"],
    "co2_emissions": ["выбросы co2", "выбросы со2", "co2", "со2"],
    # подвеска и тормоза
    "front_suspension": ["тип передней подвески"],
    "rear_suspension": ["тип задней подвески"],
    "front_brakes": ["передние тормоза"],
    "rear_brakes": ["задние тормоза"],
    # электрокары
    "battery_capacity_kwh": ["ёмкость батареи", "емкость батареи"],
    "battery_type": ["тип батареи"],
    "electric_range_km": ["запас хода на электричестве", "запас хода"],
    "max_charging_power_kw": ["максимальная мощность зарядки"],
    "fast_charging_time_min": ["время быстрой зарядки"],
    "fast_charging_description": ["описание быстрой зарядки"],
    "charging_connector": ["тип разъема для зарядки", "тип разъёма для зарядки", "разъем для зарядки"],
    "power_consumption_kwh_100": ["расход"],  # "13.7 кВт⋅ч/100 км" — пометка ниже
    "consumption_method": ["методика расчета расхода", "методика расчёта расхода"],
    # маркетинговые
    "country_brand": ["страна марки", "страна"],
    "car_class": ["класс автомобиля"],
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
    """Собрать содержимое всех meta-тегов с подходящими именами в одну строку.

    Auto.ru держит и `name="description"`, и `property="og:description"` — у них
    разный текст. Один может быть без пробега, другой с ним. Объединяем оба
    через перенос, чтобы regex'ы в `_extract_summary_fields` могли что-то найти.
    """
    parts: list[str] = []
    for name in names:
        for element in soup.find_all("meta", attrs={"name": name}):
            content = normalize_whitespace(element.get("content"))
            if content and content not in parts:
                parts.append(content)
        for element in soup.find_all("meta", attrs={"property": name}):
            content = normalize_whitespace(element.get("content"))
            if content and content not in parts:
                parts.append(content)
    return "\n".join(parts) if parts else None


def _extract_price_from_text(text: str | None) -> int | None:
    """Цена с ₽, но НЕ месячный платёж в кредите и НЕ транспортный налог.

    На проданных объявлениях Auto.ru вместо цены показывает «Транспортный
    налог: 3565 ₽/год» — это и попадало раньше в price. Игнорируем такие
    паттерны и числа меньше 50 000 ₽ (любая реальная машина дороже).
    """
    if not text:
        return None
    normalized = text.replace("\xa0", " ")

    # Перебираем все числа с ₽, отбрасываем «месяц/год/в кредит» и слишком мелкие.
    for match in re.finditer(r"(\d[\d\s]{3,})\s*₽", normalized):
        suffix = normalized[match.end():match.end() + 15].lower()
        if any(bad in suffix for bad in ("/мес", "/год", "в мес", "в год", "ежемес")):
            continue
        try:
            value = int(match.group(1).replace(" ", ""))
        except ValueError:
            continue
        if value < 50_000:
            continue  # это явно не цена машины
        return value

    return None


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


SPEC_BLACKLIST: dict[str, list[str]] = {
    # Для pts_type не брать "Владельцы по ПТС" (это owners_count).
    "pts_type": ["владельц"],
    # "Тип двигателя" — fuel_type, не engine.
    "engine": ["тип двигателя", "объём", "объем", "расположен", "максимальн"],
    # "Длина" — кузов. "Длина выноса" / прочее — исключаем чтобы не путалось.
    "length": ["выноса", "база"],
    # "Ширина" — кузов. "Ширина передней колеи" → отдельное поле.
    "width": ["колеи"],
    # "Расход" электрокара — не "Методика расчёта расхода" и не "Расход топлива".
    "power_consumption_kwh_100": ["методика", "топлива"],
    # "Расход топлива" комбо-строка — не "Методика расчёта расхода".
    "fuel_consumption_combo": ["методика"],
}


def _pick_spec(specs: dict[str, str], key: str) -> str | None:
    keywords = SPEC_KEYWORDS.get(key, [])
    blacklist = SPEC_BLACKLIST.get(key, [])
    for label, value in specs.items():
        if any(blocked in label for blocked in blacklist):
            continue
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
    """Extract unique listing URLs from a search page HTML."""
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


SOLD_MARKERS = (
    "автомобиль уже продан",
    "уже продан",
    "снято с продажи",
    "снят с продажи",
    "продано",
    "объявление снято",
)


def _detect_sold(soup: BeautifulSoup) -> bool:
    """Заглянуть в видимый текст и понять, продано ли объявление.

    Auto.ru на проданных объявлениях добавляет блок с классом `CardSold__*`
    и текстом «Этот автомобиль уже продан».
    """
    # Самый надёжный сигнал — DOM-класс CardSold
    if soup.select_one("[class*='CardSold']"):
        return True
    page_text = soup.get_text(" ", strip=True).lower()
    return any(marker in page_text for marker in SOLD_MARKERS)


def extract_listing_record(html: str, url: str) -> dict[str, object | None]:
    """Parse a single car detail page into a unified schema record."""
    soup = BeautifulSoup(html, "lxml")
    record = _empty_record(url)
    record["is_sold"] = _detect_sold(soup)

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

    text_fields = [
        # базовые
        "brand", "model", "generation",
        "body_type", "color", "transmission", "drive_type",
        "steering_wheel", "condition", "vehicle_state", "pts_type", "customs", "region",
        # offer-only
        "vin", "complectation", "modification", "eco_class",
        "service_book", "warranty_until",
        # catalog-specs
        "wheel_size", "bolt_pattern", "disc_size",
        "boost_type", "cylinder_layout", "engine_position",
        "fuel_brand", "front_suspension", "rear_suspension",
        "front_brakes", "rear_brakes",
        "country_brand", "car_class",
        # электрокары — текст
        "battery_type", "charging_connector",
        "fast_charging_description", "consumption_method",
    ]
    for field in text_fields:
        if record.get(field) is None:
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

    # На catalog/specs страницах двигатель раскидан по отдельным строкам:
    # «Тип двигателя», «Объём двигателя», «Максимальная мощность».
    if record.get("fuel_type") is None:
        engine_type_text = _pick_spec(specs, "engine_type_text")
        if engine_type_text:
            record["fuel_type"] = engine_type_text

    if record.get("engine_volume") is None:
        volume_text = _pick_spec(specs, "engine_volume_cm3")
        if volume_text:
            cm3 = extract_int(volume_text)
            if cm3:
                record["engine_volume"] = round(cm3 / 1000.0, 1)

    if record.get("engine_power_hp") is None:
        power_text = _pick_spec(specs, "engine_power_max")
        if power_text:
            # "625 л.с. (460 кВт) при 6000 об/мин" → берём первое число до "л.с."
            hp_match = re.search(r"(\d+)\s*л\.?\s*с", power_text.lower())
            if hp_match:
                record["engine_power_hp"] = int(hp_match.group(1))
            else:
                record["engine_power_hp"] = extract_int(power_text)

    # Целочисленные поля.
    int_fields = [
        "doors_count", "seats_count", "max_speed", "trunk_volume",
        "clearance", "wheelbase", "length", "width", "height",
        "weight_curb", "weight_gross", "fuel_tank_volume",
        "front_track_width", "rear_track_width",
        "max_torque_nm", "cylinders_count", "valves_per_cylinder",
        "gears_count", "co2_emissions", "tax_amount",
        # электро
        "electric_range_km", "fast_charging_time_min",
    ]
    for field in int_fields:
        if record.get(field) is None:
            text_value = _pick_spec(specs, field)
            if text_value:
                record[field] = extract_int(text_value)

    # Дробные поля (одно число).
    float_fields = [
        "acceleration_0_100",
        # электро
        "battery_capacity_kwh", "max_charging_power_kw",
    ]
    for field in float_fields:
        if record.get(field) is None:
            text_value = _pick_spec(specs, field)
            if text_value:
                record[field] = extract_float(text_value)

    # Электрокар: «Расход» = "13.7 кВт⋅ч/100 км" → power_consumption_kwh_100.
    # Для ДВС значение приходит как "22.4/10.2/14.7 л/100км" и парсится выше.
    if record.get("power_consumption_kwh_100") is None:
        consumption_text = _pick_spec(specs, "power_consumption_kwh_100")
        if consumption_text and "квт" in consumption_text.lower():
            record["power_consumption_kwh_100"] = extract_float(consumption_text)

    # Расход топлива: либо строка "22.4/10.2/14.7" (город/трасса/смешанный),
    # либо одно число (когда лейбл "Расход в смешанном цикле").
    consumption_combo = _pick_spec(specs, "fuel_consumption_combo")
    if consumption_combo:
        numbers = re.findall(r"(\d+(?:[.,]\d+)?)", consumption_combo.replace("\xa0", " "))
        nums = [float(n.replace(",", ".")) for n in numbers]
        if len(nums) >= 3:
            record["fuel_consumption_city"] = nums[0]
            record["fuel_consumption_highway"] = nums[1]
            record["fuel_consumption_mixed"] = nums[2]
        elif len(nums) == 1 and record.get("fuel_consumption_mixed") is None:
            record["fuel_consumption_mixed"] = nums[0]
    if record.get("fuel_consumption_mixed") is None:
        mixed_only = _pick_spec(specs, "fuel_consumption_mixed")
        if mixed_only:
            record["fuel_consumption_mixed"] = extract_float(mixed_only)

    page_text = soup.get_text(" ", strip=True)
    if record["price"] is None:
        record["price"] = _extract_price_from_text(page_text)
    if record["mileage"] is None and "км" in page_text.lower():
        mileage_match = re.search(r"(\d[\d\s]{2,})\s*км", page_text.lower())
        if mileage_match:
            mileage_text = mileage_match.group(1).replace("\xa0", " ").replace(" ", "")
            record["mileage"] = int(mileage_text)

    for field in ("brand", "model", "generation"):
        value = record.get(field)
        if isinstance(value, str):
            record[field] = normalize_whitespace(value.strip(","))
    # Категории, которые мы анализируем как уровни — приводим к нижнему регистру,
    # иначе из spec-map приходит "Передний", а из карточки "передний", и они
    # склеиваются в EDA/регрессии как разные уровни.
    for field in ("body_type", "fuel_type", "transmission", "drive_type"):
        value = record.get(field)
        if isinstance(value, str):
            record[field] = normalize_whitespace(value.strip(",")).lower()

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
