from __future__ import annotations

from src.parser.extractors import extract_listing_record, parse_listing_links
from src.utils.text_utils import parse_engine_text


def test_parse_listing_links_deduplicates_urls() -> None:
    html = """
    <html>
      <body>
        <a href="/cars/used/sale/bmw/12345/">BMW</a>
        <a href="https://auto.ru/cars/used/sale/bmw/12345/?from=search">BMW duplicate</a>
        <a href="/cars/used/sale/audi/99999/">Audi</a>
      </body>
    </html>
    """

    links = parse_listing_links(html, "https://auto.ru/cars/used/")

    assert links == [
        "https://auto.ru/cars/used/sale/audi/99999/",
        "https://auto.ru/cars/used/sale/bmw/12345/",
    ]


def test_parse_listing_links_falls_back_to_jsonld() -> None:
    html = """
    <html>
      <head>
        <script type="application/ld+json">
          {
            "@context": "https://schema.org",
            "@type": "Product",
            "offers": {
              "@type": "AggregateOffer",
              "offers": [
                {"@type": "Offer", "url": "https://auto.ru/cars/used/sale/skoda/octavia/12345-a/"},
                {"@type": "Offer", "url": "https://auto.ru/cars/used/sale/skoda/octavia/12345-a/?from=feed"},
                {"@type": "Offer", "url": "https://auto.ru/cars/new/group/hongqi/hs3/99999-b/"}
              ]
            }
          }
        </script>
      </head>
      <body></body>
    </html>
    """

    links = parse_listing_links(html, "https://auto.ru/cars/all/")

    assert links == [
        "https://auto.ru/cars/new/group/hongqi/hs3/99999-b/",
        "https://auto.ru/cars/used/sale/skoda/octavia/12345-a/",
    ]


def test_extract_listing_record_fills_core_fields() -> None:
    html = """
    <html>
      <body>
        <h1>Toyota Camry VII</h1>
        <span data-ftid="bull_price">2 150 000 ₽</span>
        <ul>
          <li class="CardInfoRow"><span class="CardInfoRow__label">Год выпуска</span><span class="CardInfoRow__cell">2018</span></li>
          <li class="CardInfoRow"><span class="CardInfoRow__label">Пробег</span><span class="CardInfoRow__cell">85 000 км</span></li>
          <li class="CardInfoRow"><span class="CardInfoRow__label">Двигатель</span><span class="CardInfoRow__cell">2.5 л / 181 л.с. / бензин</span></li>
          <li class="CardInfoRow"><span class="CardInfoRow__label">Привод</span><span class="CardInfoRow__cell">передний</span></li>
        </ul>
        <div data-ftid="bull_description">Хорошее состояние</div>
      </body>
    </html>
    """

    row = extract_listing_record(html, url="https://auto.ru/cars/used/sale/toyota/camry/1/")

    assert row["brand"] == "Toyota"
    assert row["model"] == "Camry"
    assert row["price"] == 2150000
    assert row["year"] == 2018
    assert row["mileage"] == 85000
    assert row["engine_volume"] == 2.5
    assert row["engine_power_hp"] == 181
    assert row["fuel_type"] == "бензин"
    assert row["description_text"] == "Хорошее состояние"


def test_extract_listing_record_handles_nbsp_mileage_fallback() -> None:
    html = """
    <html>
      <body>
        <h1>Jetta VS5 I</h1>
        <div>Пробег 20&nbsp;100 км</div>
        <div>Цена 2 050 000 ₽</div>
      </body>
    </html>
    """

    row = extract_listing_record(html, url="https://auto.ru/cars/used/sale/jetta/vs5/1/")

    assert row["price"] == 2050000
    assert row["mileage"] == 20100


def test_extract_listing_record_uses_meta_summary_fields() -> None:
    html = """
    <html>
      <head>
        <title>Купить новый 212 T01 2024-2026 2.0 AT (238 л.с.) 4WD бензин автомат в Благовещенске: чёрный 212 T01 2026 внедорожник 5-дверный 2026 года по цене 3 799 000 рублей на Авто.ру</title>
        <meta name="description" content="Новый 212 T01 2024-2026 2.0 AT (238 л.с.) 4WD бензин автомат, чёрный внедорожник 5-дверный 2026 года от дилера за 3 799 000 рублей." />
        <meta property="og:description" content="Внедорожник 212 T01 2024-2026 2026 года, новый, двигатель 2.0 AT (238 л.с.) 4WD, цвет чёрный за 3 799 000 рублей." />
      </head>
      <body></body>
    </html>
    """

    row = extract_listing_record(html, url="https://auto.ru/cars/new/group/212/212_t01/1/")

    assert row["year"] == 2026
    assert row["engine_volume"] == 2.0
    assert row["engine_power_hp"] == 238
    assert row["fuel_type"] == "бензин"
    assert row["transmission"] == "автомат"
    assert row["drive_type"] == "полный"
    assert row["body_type"] == "внедорожник"
    assert row["color"] in {"чёрный", "черный"}
    assert row["condition"] == "new"
    assert row["seller_type"] == "dealer"


def test_parse_engine_text() -> None:
    parsed = parse_engine_text("1.4 л / 125 л.с. / бензин")
    assert parsed["engine_volume"] == 1.4
    assert parsed["engine_power_hp"] == 125
    assert parsed["fuel_type"] == "бензин"
