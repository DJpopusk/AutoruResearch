"""CLI-обёртка для парсера.

Зачем: Playwright sync API не работает внутри Jupyter (конфликт с asyncio-циклом
ядра — окно не открывается, скрипт молча падает). Поэтому интерактивный режим
с видимым окном надо запускать из обычного терминала.

Использование:

    # из ноутбука: сгенерируется JSON-конфиг и команда вида
    python scrape_cli.py path/to/scrape_config.json

Файл конфига — JSON со всеми полями ScrapeConfig (Path-ы как строки).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from functions.scraper import ScrapeConfig, run_scrape


def _coerce_paths(payload: dict) -> dict:
    for key in ("output_csv", "output_parquet", "state_file", "checkpoint_jsonl", "aggregate_parquet"):
        if key in payload and payload[key] is not None and not isinstance(payload[key], Path):
            payload[key] = Path(payload[key]).expanduser().resolve()
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Запустить парсер Auto.ru из терминала.")
    parser.add_argument("config", type=Path, help="JSON-файл с полями ScrapeConfig.")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.config.exists():
        print(f"Конфиг не найден: {args.config}", file=sys.stderr)
        return 1

    payload = json.loads(args.config.read_text(encoding="utf-8"))
    payload = _coerce_paths(payload)

    config = ScrapeConfig(**payload)
    print(f"[{datetime.now():%H:%M:%S}] Старт парсера. Конфиг: {args.config}")
    print(f"  catalog_url={config.catalog_url}")
    print(f"  pages={config.pages}  interactive_confirm={config.interactive_confirm}")
    print(f"  output → {config.output_parquet}")

    df = run_scrape(config)
    print(f"[{datetime.now():%H:%M:%S}] Готово. Спарсено строк: {len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
