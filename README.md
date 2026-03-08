# Auto.ru Market Research

Полноценный Python-проект для:

- сбора объявлений с Auto.ru;
- очистки и нормализации датасета;
- статистического анализа факторов цены;
- обучения моделей регрессии для прогноза цены автомобиля.

Проект рассчитан на Python `3.11+`. В текущей рабочей среде использовался Python `3.13`.

## Что делает проект

Пайплайн состоит из 4 этапов:

1. `scrape`
   - проходит по страницам выдачи Auto.ru;
   - собирает ссылки на объявления;
   - открывает карточки;
   - извлекает цену и характеристики;
   - сохраняет сырые данные.

2. `preprocess`
   - приводит типы;
   - нормализует категориальные признаки;
   - удаляет дубли;
   - фильтрует очевидные выбросы;
   - формирует аналитический датасет.

3. `analyze1`
   - проверяет нормальность;
   - проверяет различия цены по группам;
   - оценивает влияние признаков на цену;
   - проверяет зависимости между признаками.

4. `analyze2`
   - обучает несколько моделей регрессии;
   - сравнивает их по метрикам;
   - сохраняет лучшую модель и отчеты.

## Структура проекта

```text
.
├── main.py
├── requirements.txt
├── .env.example
├── src/
│   ├── parser/
│   │   ├── selectors.py
│   │   ├── fetchers.py
│   │   ├── extractors.py
│   │   └── scraper.py
│   ├── preprocessing/
│   │   └── cleaner.py
│   ├── analysis/
│   │   ├── analysis_stage1.py
│   │   └── analysis_stage2_regression.py
│   ├── modeling/
│   │   └── regression.py
│   └── utils/
│       ├── constants.py
│       ├── io_utils.py
│       ├── logging_utils.py
│       ├── network.py
│       └── text_utils.py
└── tests/
```

Во время работы также создаются директории:

- `data/raw/`
- `data/processed/`
- `reports/`
- `figures/`
- `models/`
- `logs/`

## Установка

### 1. Клонировать репозиторий

```bash
git clone <REPOSITORY_URL>
cd autoru_market_research
```

### 2. Создать виртуальное окружение

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

Если у тебя `python3` уже указывает на Python `3.13`, можно так:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Установить зависимости

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
playwright install chromium
```

## Быстрый старт

Полный пайплайн:

```bash
python main.py full-pipeline --url "https://auto.ru/cars/all/" --pages 20
```

Если хочешь запускать по шагам:

```bash
python main.py scrape --url "https://auto.ru/cars/all/" --pages 20
python main.py preprocess --input data/raw/autoru_raw.parquet --output-dir data/processed
python main.py analyze1 --input data/processed/cleaned_dataset.parquet
python main.py analyze2 --input data/processed/cleaned_dataset.parquet
```

## Что означает каждая команда

### `scrape`

Команда:

```bash
python main.py scrape --url "https://auto.ru/cars/all/" --pages 20
```

Что делает:

- проходит по указанным страницам каталога;
- собирает объявления;
- сохраняет сырые данные в `CSV` и `Parquet`;
- пишет состояние обхода в checkpoint/state;
- поддерживает безопаский повторный запуск.

Что создаёт:

- `data/raw/autoru_raw.csv`
- `data/raw/autoru_raw.parquet`
- `data/raw/autoru_state.json`
- `data/raw/autoru_checkpoint.jsonl`

Когда запускать:

- когда хочешь собрать данные впервые;
- когда хочешь расширить датасет, увеличив `--pages`;
- когда хочешь повторно использовать уже сохраненный checkpoint без потери данных.

Пример:

```bash
python main.py scrape --url "https://auto.ru/cars/all/" --pages 50
```

### `preprocess`

Команда:

```bash
python main.py preprocess --input data/raw/autoru_raw.parquet --output-dir data/processed
```

Что делает:

- читает сырой датасет;
- удаляет дубли;
- вытаскивает числа из строк;
- приводит признаки к нужным типам;
- нормализует категории;
- фильтрует явный мусор;
- добавляет вычисляемые признаки.

Что создаёт:

- `data/processed/cleaned_dataset.csv`
- `data/processed/cleaned_dataset.parquet`
- `data/processed/preprocessing_summary.json`

Когда запускать:

- после каждого нового `scrape`;
- перед любым анализом или обучением модели.

### `analyze1`

Команда:

```bash
python main.py analyze1 --input data/processed/cleaned_dataset.parquet
```

Что делает:

- строит гистограммы, boxplot и QQ-plot;
- проверяет нормальность числовых признаков;
- сравнивает цену по группам;
- считает корреляции, Cramer's V, mutual information;
- оценивает мультиколлинеарность и зависимости признаков.

Что создаёт:

- таблицы в `reports/`
- графики в `figures/`
- итоговый отчёт `reports/stage1_report.md`

Когда запускать:

- когда нужен исследовательский анализ данных;
- перед построением финальной модели;
- для подготовки отчёта по статистике.

### `analyze2`

Команда:

```bash
python main.py analyze2 --input data/processed/cleaned_dataset.parquet
```

Что делает:

- делит данные на `train/test`;
- кодирует категориальные признаки;
- обучает модели:
  - `Linear Regression`
  - `Ridge`
  - `Lasso`
  - `Random Forest`
  - `Gradient Boosting`
  - `XGBoost`, если доступен
- считает метрики:
  - `MAE`
  - `RMSE`
  - `R²`
  - `MAPE`
- сохраняет лучшую модель на диск.

Что создаёт:

- `reports/stage2_model_metrics.csv`
- `reports/stage2_cv_metrics.csv`
- `reports/stage2_feature_importance.csv`
- `reports/stage2_summary.json`
- `reports/stage2_report.md`
- `models/best_price_model.joblib`
- диагностические графики в `figures/`

Когда запускать:

- после `preprocess`;
- когда нужна оценка качества прогноза цены;
- когда нужно сохранить лучшую модель.

### `full-pipeline`

Команда:

```bash
python main.py full-pipeline --url "https://auto.ru/cars/all/" --pages 20
```

Что делает:

- последовательно выполняет `scrape -> preprocess -> analyze1 -> analyze2`.

Когда запускать:

- если нужен полный цикл одной командой;
- если не требуется вручную разбирать промежуточные этапы.

## Полезные сценарии запуска

### Собрать данные только один раз

```bash
python main.py scrape --url "https://auto.ru/cars/all/" --pages 20
```

### Повторно прогнать аналитику на уже собранных данных

```bash
python main.py preprocess --input data/raw/autoru_raw.parquet --output-dir data/processed
python main.py analyze1 --input data/processed/cleaned_dataset.parquet
python main.py analyze2 --input data/processed/cleaned_dataset.parquet
```

### Расширить выборку

```bash
python main.py scrape --url "https://auto.ru/cars/all/" --pages 100
python main.py preprocess --input data/raw/autoru_raw.parquet --output-dir data/processed
```

### Запустить только моделирование

```bash
python main.py analyze2 --input data/processed/cleaned_dataset.parquet
```

## Какие поля собираются

Схема единая. Если поле не найдено, сохраняется `None/NaN`.

- `brand`
- `model`
- `generation`
- `year`
- `price`
- `mileage`
- `body_type`
- `color`
- `engine_volume`
- `engine_power_hp`
- `fuel_type`
- `transmission`
- `drive_type`
- `steering_wheel`
- `condition`
- `owners_count`
- `pts_type`
- `customs`
- `region`
- `seller_type`
- `description_text`
- `url`
- `parsed_at`

После `preprocess` появляются также вычисляемые признаки:

- `age`
- `price_log1p`

## Ограничения парсинга

- Структура Auto.ru меняется, поэтому селекторы вынесены отдельно в `src/parser/selectors.py`.
- Парсер не использует агрессивный scraping и не пытается обходить капчу/антибот.
- Если часть полей не найдена, запись не падает, а сохраняется частично.
- При повторном запуске `scrape` используется checkpoint/state, поэтому уже собранные данные не теряются.
- Парсить весь `cars/all/` нецелесообразно: лучше ограничивать выборку по страницам или сегментам.

## Как интерпретировать результаты модели

Смотреть:

- `reports/stage2_model_metrics.csv`
- `reports/stage2_summary.json`
- `reports/stage2_report.md`

Главные метрики:

- `MAE` — средняя абсолютная ошибка в рублях
- `RMSE` — ошибка с усиленным штрафом за крупные промахи
- `R²` — доля объяснённой вариации цены
- `MAPE` — средняя относительная ошибка в процентах

Пример:

- `MAE = 368000` означает, что модель в среднем ошибается примерно на `368 тыс. руб.`
- `MAPE = 7.2%` означает средний относительный промах около `7.2%`

## Где лежат результаты

- сырые данные: `data/raw/`
- очищенные данные: `data/processed/`
- статистические таблицы: `reports/`
- графики: `figures/`
- лучшая модель: `models/best_price_model.joblib`
- логи запусков: `logs/`

## Тесты

Запуск:

```bash
pytest -q
```

Что покрыто:

- parsing helpers
- resume-сценарий парсера
- preprocessing

## Типичные проблемы

### `python: command not found`

Используй явный бинарник:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

### `playwright: command not found`

После активации окружения установи зависимости и браузер:

```bash
python -m pip install -r requirements.txt
playwright install chromium
```

### `Input dataset is empty`

Проверь порядок запуска:

```bash
python main.py scrape --url "https://auto.ru/cars/all/" --pages 20
python main.py preprocess --input data/raw/autoru_raw.parquet --output-dir data/processed
python main.py analyze1 --input data/processed/cleaned_dataset.parquet
```

### Повторный `scrape` ничего не скачивает

Это нормально, если:

- ты уже проходил те же страницы;
- состояние сохранено в `data/raw/autoru_state.json`;
- данные есть в `data/raw/autoru_checkpoint.jsonl`.

В таком случае проект повторно использует уже собранные записи.

## Что лучше улучшать дальше

- собирать более однородные сегменты рынка вместо `cars/all/`
- расширить схему признаков
- добавить отдельный модуль инференса для прогноза по одной машине
- экспортировать сегментные метрики модели в отдельный CSV/Markdown
