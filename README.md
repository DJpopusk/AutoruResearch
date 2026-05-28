# Auto.ru Market Research

Ноутбучный проект для сбора, анализа и моделирования цен на Auto.ru.

Python `3.13`.

## Структура

```text
.
├── scrape_cli.py                # CLI-обёртка парсера для интерактивного режима из терминала
├── notebooks/
│   ├── 01_scraping.ipynb        # запуск парсера (по умолчанию RUN_SCRAPE=False)
│   ├── 02_eda.ipynb             # нормальность, однородность, корреляции
│   ├── 03_preprocessing.ipynb   # пошаговая чистка с пояснениями
│   ├── 04_regression.ipynb      # OLS + Box-Cox + Ridge/Lasso/ENet + FE + backward
│   └── 05_residuals.ipynb       # адекватность + диагностика остатков OLS
├── functions/                   # переиспользуемые функции
│   ├── constants.py             # схемы, фичи, пути
│   ├── io.py                    # load/save, read/write json
│   ├── text.py                  # text utilities
│   ├── network.py               # HttpClient с rate-limit и retry
│   ├── selectors.py             # CSS-селекторы Auto.ru
│   ├── extractors.py            # парсинг карточек и каталога
│   ├── fetchers.py              # requests / Playwright backends
│   ├── scraper.py               # ScrapeConfig + AutoRuScraper + run_scrape
│   ├── preprocessing.py         # композируемые шаги чистки
│   ├── eda.py                   # normality, group tests, VIF, Cramer's V и т.д.
│   ├── regression.py            # пайплайны, оценка, SegmentedRegressor
│   └── diagnostics.py           # OLS design, adequacy, residual checks, F-tests
└── data/
    ├── raw/                     # сырые выгрузки парсера
    │   └── notebook_run/        # сюда пишет парсер из ноутбука
    └── processed/               # cleaned_dataset.{csv,parquet}
```

## Как пользоваться

Запусти Jupyter в корне проекта:

```bash
jupyter lab
```

Ноутбуки идут по порядку:

1. `01_scraping.ipynb` — конфигурирует и (при желании) запускает парсер.
   По умолчанию `RUN_SCRAPE=False`, чтобы не трогать существующий `data/raw/autoru_raw.parquet`.
   Свежие парсинги пишутся в `data/raw/notebook_run/<run_id>/`.
   Для интерактивного режима с видимым окном Chromium — флаг `INTERACTIVE_CONFIRM=True`:
   ноутбук сохранит JSON-конфиг и напечатает команду `python scrape_cli.py <config>`,
   которую надо выполнить в терминале (Playwright sync API не работает в Jupyter).
2. `02_eda.ipynb` — статистический анализ: Shapiro-Wilk, ANOVA/Kruskal,
   Spearman, Cramer's V, MI, VIF.
3. `03_preprocessing.ipynb` — пошаговая чистка датасета.
   `WRITE_OUTPUT=False` по умолчанию (не переписывает `data/processed/cleaned_dataset.*`).
4. `04_regression.ipynb` — OLS через statsmodels (R², R²_adj, F, AIC, BIC, t-тесты),
   Box-Cox для таргета, Ridge/Lasso/ElasticNet, feature engineering (степени,
   произведения, kink), backward elimination, плюс RF/GB/XGBoost для сравнения и
   сегментированная по `condition` модель.
5. `05_residuals.ipynb` — анализ остатков и адекватность OLS из ноутбука 4:
   стандартизация, QQ-plot, Jarque-Bera, Shapiro-Wilk, Breusch-Pagan, White,
   Goldfeld-Quandt, Durbin-Watson, Breusch-Godfrey, RESET-тест Рамсея,
   групповой F-тест для вложенных моделей.

## Зависимости

```bash
pip install -r requirements.txt
# для парсера в режиме playwright:
playwright install chromium
```

## Защита данных

- Исходный `data/raw/autoru_raw.parquet` и `data/processed/cleaned_dataset.parquet`
  никогда не переписываются автоматически.
- Парсер в ноутбуке пишет в изолированный `data/raw/notebook_run/...`.
- Сохранение очищенного датасета в ноутбуке 3 включается вручную флагом `WRITE_OUTPUT`.
