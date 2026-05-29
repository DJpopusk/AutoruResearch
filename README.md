# Auto.ru Market Research

Регрессионный анализ цен подержанных автомобилей по данным Auto.ru: от сбора данных
парсером до интерпретируемой линейной модели с полной диагностикой.

Python `3.13`.

## Структура

```text
.
├── scrape_cli.py                       # CLI-обёртка парсера для интерактивного режима из терминала
├── presentation_notebooks/             # основной пайплайн (запускать по порядку 00 → 09)
│   ├── 00_scraping.ipynb               # запуск парсера + агрегация прогонов в общий parquet
│   ├── 01_feature_analysis.ipynb       # описание признаков, корреляции (Pearson/Spearman/MI), PCA
│   ├── 02_feature_selection.ipynb      # кластерный отбор признаков (corr_y + communality) → KEEP/DROP
│   ├── 03_baseline_models.ipynb        # OLS: Full / Best / Min (backward elimination), VIF
│   ├── 04_diagnostics_baseline.ipynb   # диагностика остатков: JB, Shapiro, BP, White, GQ, DW, RESET
│   ├── 05_feature_engineering.ipynb    # ratio-признаки, центрирование, нелинейности
│   ├── 06_wls.ipynb                    # гетероскедастичность → WLS (двухшаговая FGLS)
│   ├── 07_segmentation.ipynb           # сегментированная регрессия (vehicle_state, возраст)
│   ├── 08_final_summary.ipynb          # сводка всех моделей и тестов
│   ├── 09_regularization.ipynb         # Ridge / Lasso / ElasticNet (sanity-check отбора)
│   ├── helpers.py                      # общий код: загрузка, дизайн-матрица, тесты, FGLS, стиль графиков
│   ├── selected_features.json          # артефакт отбора (пишет 02, читают 03–09)
│   └── *.pkl                           # промежуточные модели между ноутбуками
├── functions/                          # модули парсера (цепочка скрапинга)
│   ├── constants.py                    # пути, схема AUTORU_SCHEMA, AGGREGATE_PARQUET
│   ├── io.py                           # load/save, json, append_to_aggregate
│   ├── text.py                         # текстовые утилиты, нормализация URL
│   ├── network.py                      # HttpClient с rate-limit и retry
│   ├── selectors.py                    # CSS-селекторы Auto.ru
│   ├── extractors.py                   # парсинг карточек и каталога
│   ├── fetchers.py                     # requests / Playwright backends
│   └── scraper.py                      # ScrapeConfig + AutoRuScraper + run_scrape
└── data/                               # данные (в .gitignore, локально)
    ├── raw/
    │   ├── autoru_raw_all.parquet      # общий parquet всех прогонов (дедуп по url)
    │   └── notebook_run/<run_id>/      # сырые выгрузки каждого прогона
    └── processed/
        └── cleaned_dataset_all_runs.parquet   # очищенный датасет для регрессии
```

## Пайплайн анализа

Цель — построить интерпретируемую регрессионную модель цены автомобиля, оценить её
метрики и выявить ключевые факторы стоимости (включая неочевидные зависимости).
Каждый ноутбук — отдельный логический шаг, вытекающий из предыдущего:

1. **00 — Скрапинг.** Парсер каталога и карточек (requests / Playwright). Каждый прогон
   пишет свой parquet и дописывается в общий `autoru_raw_all.parquet` с дедупликацией по `url`.
2. **01 — Анализ признаков.** 23 числовых + 4 категориальных. Корреляции с `ln(price)`
   (Pearson, Spearman, Mutual Information), частные корреляции, матрицы по группам, PCA-биплот.
3. **02 — Отбор признаков.** Иерархическая кластеризация по `1 − |corr|`; в каждом кластере
   оставляем «чемпиона» по `corr_y`, с проверкой communality. Результат → `selected_features.json`.
4. **03 — Базовые модели.** OLS на трёх наборах: Full (все), Best (отобранные), Min
   (backward elimination). Сравнение R²adj / AIC / BIC / VIF.
5. **04 — Диагностика остатков.** Полная батарея тестов с H₀ и p-value: нормальность
   (Jarque–Bera, Shapiro–Wilk), гомоскедастичность (Breusch–Pagan, White, Goldfeld–Quandt),
   автокорреляция (Durbin–Watson), линейность формы (RESET).
6. **05 — Feature engineering.** Ratio-признаки (`hp_per_kg`, `hp_per_liter`), центрирование
   `age`/`mileage` (против artificial VIF), нелинейности (`age²`, `mileage²`, взаимодействия).
7. **06 — WLS.** Двухшаговая FGLS против гетероскедастичности: оценка `σ²(x)`, веса `1/σ̂²`.
8. **07 — Сегментация.** Отдельные модели по `vehicle_state` (на ходу / битые) и возрасту;
   честная out-of-sample оценка на test.
9. **08 — Итоговая сводка.** Все модели и тесты в одной таблице, проверка критериев успеха.
10. **09 — Регуляризация.** Ridge / Lasso / ElasticNet с CV — подтверждение, что отбор не
    оставил избыточной мультиколлинеарности.

## Как пользоваться

```bash
pip install -r requirements.txt
playwright install chromium        # для парсера в режиме Playwright
jupyter lab
```

Ноутбуки запускать по порядку из `presentation_notebooks/` (нумерация 00 → 09).
Артефакты (`selected_features.json`, `*.pkl`) передаются между ноутбуками автоматически.

### Скрапинг

В `00_scraping.ipynb` по умолчанию `RUN_SCRAPE=True` пишет в изолированный
`data/raw/notebook_run/<run_id>/` и дописывает результат в общий `autoru_raw_all.parquet`.

- **Одиночный прогон**: `ScrapeConfig(aggregate_parquet=AGGREGATE_PARQUET, ...)` — общий
  parquet пополняется автоматически после `run_scrape`.
- **Параллельные батчи**: каждый батч пишет свой parquet; слияние в общий — отдельной
  ячейкой после завершения всех терминалов (чтобы избежать гонки записи).
- **Интерактивный режим** (`INTERACTIVE_CONFIRM=True`): Playwright sync API не работает в
  Jupyter, поэтому ноутбук сохраняет JSON-конфиг и печатает команду
  `python scrape_cli.py <config>` для запуска в терминале с видимым окном Chromium.

## Данные

`data/` в `.gitignore` — сырые и очищенные parquet хранятся локально, в репозиторий не
попадают. Общий `autoru_raw_all.parquet` накапливает все прогоны с дедупликацией по `url`
и колонкой `source_run` (метка прогона).
