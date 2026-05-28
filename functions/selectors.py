"""CSS selector definitions for Auto.ru parsing.

Selectors live here so they can be tweaked when markup changes
without touching parser logic.
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
        # Современные классы Auto.ru
        "[class*='OfferPriceCaption']",
        "[class*='OfferPriceBadge']",
        "[class*='PriceListingItem__price']",
        # Древние/data-ftid
        "[data-ftid='bull_price']",
        "[data-testid='price']",
        "meta[itemprop='price']",
    ],
    "description": [
        "[data-ftid='bull_description']",
        "[class*='CardDescription__textInner']",
        "[class*='CardDescription']",
        "[class*='OfferDescription']",
    ],
    "title": [
        "h1",
        "[class*='CardHead__title']",
        "[class*='OfferTitle']",
    ],
    "region": [
        # Старые
        "[class*='CardSellerLocation__regionName']",
        "[class*='CardSellerLocation']",
        "[data-ftid='seller_place']",
        # Современные (из CSS: SellerPopupFooter, MetroListPlace)
        "[class*='SellerPopupFooter__address']",
        "[class*='MetroListPlace__regionName']",
        "[class*='SellerLocation']",
    ],
    "seller_badge": [
        "[class*='SellerStatus']",
        "[class*='SellerPopupHeader__sellerOfficialDealer']",
        "[class*='SellerPopupHeader__sellerDescription']",
        "[data-ftid='seller_type']",
        "[data-testid='seller-type']",
    ],
}

SPEC_ROW_SELECTORS = [
    # Современный Auto.ru (CSS Modules, BEM + hash) — каждая «строка» характеристики
    "[class*='ModificationInfoRenovation__option']:not([class*='Name']):not([class*='Value'])",
    "[class*='CardTechInfoDimensionsItem']:not([class*='__name']):not([class*='__value'])",
    "[class*='CatalogCardComplectationFilter__infoItem']:not([class*='Title']):not([class*='Value'])",
    "[class*='OfferCardCharacteristics'] li",
    "[class*='OfferCardCharacteristics'] [class*='Row']",
    "[class*='SpecificationContent__tableBodyRow']",
    "[class*='ReglamentToContent__tableBodyRow']",
    "[class*='SpecificationsItem']",
    "[class*='CardInfoRow']",
    "[class*='CardInfo__row']",
    # дополнительные fallback'и под data-testid и древние варианты
    "[data-testid*='char'] > div",
    "[data-testid*='spec'] > div",
    "ul[class*='CardInfoSummary__list'] > li[class*='CardInfoSummarySimpleRow']",
    "ul[class*='CardInfoSummary__list'] > li[class*='CardInfoSummaryComplexRow']",
    "li.CardInfoRow",
    "dl > div",
    "tr",
]

SPEC_KEY_SELECTORS = [
    # Современные блоки Auto.ru
    "[class*='ModificationInfoRenovation__optionName']",
    "[class*='CardTechInfoDimensionsItem__name']",
    "[class*='CatalogCardComplectationFilter__infoItemTitle']",
    "[class*='SpecificationContent__tableHeadCell']",
    "[class*='ReglamentToContent__tableHeadCell']",
    "[class*='ListContent2__title']",
    # Card layout — title идёт ПЕРВЫМ, иначе cellLabel-обёртка съест value целиком.
    "[class*='CardInfoSummaryComplexRow__cellTitle']",
    "[class*='CardInfoSummarySimpleRow__label']",
    "[class*='CardInfoRow__label']",
    "[class*='CardInfo__cell']:first-child",
    "[data-testid*='label']",
    # Обобщённые
    "dt",
    "th",
]

SPEC_VALUE_SELECTORS = [
    # Современные блоки Auto.ru
    "[class*='ModificationInfoRenovation__optionValue']",
    "[class*='ModificationInfoRenovation__optionValueLink']",
    "[class*='CardTechInfoDimensionsItem__value']",
    "[class*='CatalogCardComplectationFilter__infoItemValue']",
    "[class*='SpecificationContent__tableBodyCell']:not(:first-child)",
    "[class*='ReglamentToContent__tableBodyCell']:not(:first-child)",
    "[class*='CatalogComplectationList__itemInfo']",
    "[class*='ListContent2__description']",
    "[class*='ListContent2__subtitle']",
    # Старый дизайн
    "[class*='CardInfoRow__cell']",
    "[class*='CardInfoSummarySimpleRow__content']",
    "[class*='CardInfoSummaryComplexRow__cellValue']",
    "[class*='CardInfo__cell']:last-child",
    "[data-testid*='value']",
    # Обобщённые
    "dd",
    "td",
]

# Кнопки/ссылки на детальной странице, которые надо «прокликать» в headed-режиме,
# чтобы выдать экстрактору полную таблицу характеристик.
EXPAND_BUTTON_TEXTS = (
    "Все характеристики",
    "Все характеристики и опции",
    "Полные характеристики",
    "Все параметры",
    "Показать все характеристики",
    "Показать всё",
    "Показать ещё",
    "Развернуть",
    "Подробнее",
    "Характеристики",  # часто это ссылка на вкладку спецификации
    "Опции",
)

EXPAND_BUTTON_SELECTORS = [
    "button[class*='CardSpoiler']",
    "button[class*='Spoiler']",
    "button[class*='ShowMore']",
    "div[class*='SpoilerLink']",
    "a[class*='SpoilerLink']",
    "a[href*='/specifications']",
    "a[href*='/spec/']",
    "[data-testid*='spoiler']",
    "[data-testid*='show-more']",
    "[data-testid*='expand']",
    "[data-testid*='characteristics-link']",
]
