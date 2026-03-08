"""CSS selector definitions for Auto.ru parsing.

Selectors are isolated here because Auto.ru markup can change over time.
To adapt quickly, update these lists without touching parser logic.
"""

from __future__ import annotations

LISTING_LINK_SELECTORS = [
    "a.ListingItemTitle__link",
    "a[data-ftid='bulls-list_bull']",
    "a.Link.ListingItemTitle__link",
    "a[href*='/cars/used/sale/']",
    "a[href*='/cars/new/sale/']",
]

DETAIL_TEXT_SELECTORS = {
    "price": [
        "span.PriceListingItem__price",
        "span[data-ftid='bull_price']",
        "div[data-testid='price']",
        "span[class*='OfferPriceCaption']",
        "meta[itemprop='price']",
    ],
    "description": [
        "div[data-ftid='bull_description']",
        "div.CardDescription__textInner",
        "div.CardDescription",
    ],
    "title": [
        "h1",
        "div.CardHead__title",
    ],
    "region": [
        "span.CardSellerLocation__regionName",
        "div[data-ftid='seller_place']",
        "span[class*='CardSellerLocation']",
    ],
    "seller_badge": [
        "div.SellerStatus",
        "span[data-ftid='seller_type']",
        "div[data-testid='seller-type']",
    ],
}

SPEC_ROW_SELECTORS = [
    "li.CardInfoRow",
    "li[class*='CardInfoRow']",
    "div.CardInfo__row",
    "tr",
]

SPEC_KEY_SELECTORS = [
    "span.CardInfoRow__label",
    "span[class*='CardInfoRow__label']",
    "th",
    "div[class*='CardInfo__cell']:first-child",
]

SPEC_VALUE_SELECTORS = [
    "span.CardInfoRow__cell",
    "span[class*='CardInfoRow__cell']",
    "td",
    "div[class*='CardInfo__cell']:last-child",
]
