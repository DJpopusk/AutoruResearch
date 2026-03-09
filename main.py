"""CLI entrypoint for Auto.ru market research pipeline."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from src.analysis.analysis_stage1 import Stage1Config, run_stage1_analysis
from src.analysis.analysis_stage2_regression import Stage2Config, run_stage2_analysis
from src.parser.scraper import ScrapeConfig, run_scrape
from src.preprocessing.cleaner import PreprocessConfig, preprocess_dataset
from src.utils.constants import FIGURES_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR
from src.utils.logging_utils import setup_logging

LOGGER = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auto.ru market research pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scrape_parser = subparsers.add_parser("scrape", help="Scrape Auto.ru catalog and listings")
    scrape_parser.add_argument("--url", required=True, help="Auto.ru catalog/search URL")
    scrape_parser.add_argument("--pages", type=int, default=20, help="Number of catalog pages to scan")
    scrape_parser.add_argument(
        "--start-page",
        type=int,
        default=None,
        help="Start scanning from this catalog page instead of resume state",
    )
    scrape_parser.add_argument(
        "--output-csv",
        type=Path,
        default=RAW_DATA_DIR / "autoru_raw.csv",
        help="Output CSV path",
    )
    scrape_parser.add_argument(
        "--output-parquet",
        type=Path,
        default=RAW_DATA_DIR / "autoru_raw.parquet",
        help="Output Parquet path",
    )
    scrape_parser.add_argument(
        "--state-file",
        type=Path,
        default=RAW_DATA_DIR / "autoru_state.json",
        help="State file for resume",
    )
    scrape_parser.add_argument(
        "--checkpoint-jsonl",
        type=Path,
        default=RAW_DATA_DIR / "autoru_checkpoint.jsonl",
        help="JSONL checkpoint file",
    )
    scrape_parser.add_argument("--playwright", action="store_true", help="Use Playwright backend")
    scrape_parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    scrape_parser.add_argument("--min-delay", type=float, default=1.2, help="Minimum delay between requests")
    scrape_parser.add_argument("--max-delay", type=float, default=3.0, help="Maximum delay between requests")
    scrape_parser.add_argument(
        "--challenge-cooldown",
        type=int,
        default=90,
        help="Cooldown in seconds before retrying after suspected anti-bot/challenge page",
    )
    scrape_parser.add_argument(
        "--max-consecutive-challenge-pages",
        type=int,
        default=2,
        help="Stop scrape after this many consecutive suspected challenge pages",
    )
    scrape_parser.add_argument(
        "--challenge-retries-per-page",
        type=int,
        default=1,
        help="Number of retries for a page that looks like a challenge/stub response",
    )
    scrape_parser.add_argument(
        "--condition-filter",
        choices=["all", "used", "new"],
        default="all",
        help="Filter listings by condition/type",
    )
    scrape_parser.add_argument(
        "--seller-type-filter",
        choices=["all", "private", "dealer"],
        default="all",
        help="Filter listings by seller type",
    )
    scrape_parser.add_argument(
        "--min-mileage",
        type=int,
        default=None,
        help="Keep only listings with mileage greater than or equal to this value",
    )

    preprocess_parser = subparsers.add_parser("preprocess", help="Clean and normalize dataset")
    preprocess_parser.add_argument("--input", type=Path, required=True, help="Raw CSV/Parquet input path")
    preprocess_parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory for cleaned artifacts",
    )
    preprocess_parser.add_argument(
        "--exclude-commercial-like",
        action="store_true",
        help="Exclude listings that look like commercial/dealer marketing pages by heuristic",
    )

    analyze1_parser = subparsers.add_parser("analyze1", help="Run stage-1 statistical analysis")
    analyze1_parser.add_argument("--input", type=Path, required=True, help="Cleaned dataset path")
    analyze1_parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    analyze1_parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)

    analyze2_parser = subparsers.add_parser("analyze2", help="Run stage-2 regression analysis")
    analyze2_parser.add_argument("--input", type=Path, required=True, help="Cleaned dataset path")
    analyze2_parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    analyze2_parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    analyze2_parser.add_argument("--models-dir", type=Path, default=Path("models"))

    full_parser = subparsers.add_parser("full-pipeline", help="Run all stages in order")
    full_parser.add_argument("--url", required=True, help="Auto.ru catalog/search URL")
    full_parser.add_argument("--pages", type=int, default=20)
    full_parser.add_argument("--start-page", type=int, default=None)
    full_parser.add_argument("--playwright", action="store_true")
    full_parser.add_argument("--timeout", type=int, default=30)
    full_parser.add_argument("--min-delay", type=float, default=1.2)
    full_parser.add_argument("--max-delay", type=float, default=3.0)
    full_parser.add_argument("--challenge-cooldown", type=int, default=90)
    full_parser.add_argument("--max-consecutive-challenge-pages", type=int, default=2)
    full_parser.add_argument("--challenge-retries-per-page", type=int, default=1)
    full_parser.add_argument("--condition-filter", choices=["all", "used", "new"], default="all")
    full_parser.add_argument("--seller-type-filter", choices=["all", "private", "dealer"], default="all")
    full_parser.add_argument("--min-mileage", type=int, default=None)

    return parser


def _run_scrape(args: argparse.Namespace) -> Path:
    config = ScrapeConfig(
        catalog_url=args.url,
        pages=args.pages,
        start_page=args.start_page,
        output_csv=args.output_csv,
        output_parquet=args.output_parquet,
        state_file=args.state_file,
        checkpoint_jsonl=args.checkpoint_jsonl,
        use_playwright=args.playwright,
        timeout_seconds=args.timeout,
        min_delay_seconds=args.min_delay,
        max_delay_seconds=args.max_delay,
        condition_filter=args.condition_filter,
        seller_type_filter=args.seller_type_filter,
        min_mileage=args.min_mileage,
        challenge_cooldown_seconds=args.challenge_cooldown,
        max_consecutive_challenge_pages=args.max_consecutive_challenge_pages,
        challenge_retries_per_page=args.challenge_retries_per_page,
    )
    run_scrape(config)
    return args.output_parquet


def _run_preprocess(args: argparse.Namespace) -> Path:
    config = PreprocessConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        exclude_commercial_like=args.exclude_commercial_like,
    )
    preprocess_dataset(config)
    return args.output_dir / "cleaned_dataset.parquet"


def _run_analyze1(args: argparse.Namespace) -> None:
    config = Stage1Config(
        input_path=args.input,
        reports_dir=args.reports_dir,
        figures_dir=args.figures_dir,
    )
    run_stage1_analysis(config)


def _run_analyze2(args: argparse.Namespace) -> None:
    config = Stage2Config(
        input_path=args.input,
        reports_dir=args.reports_dir,
        figures_dir=args.figures_dir,
        models_dir=args.models_dir,
    )
    run_stage2_analysis(config)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    log_file = Path("logs") / f"autoru_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(log_file=log_file)
    LOGGER.info("Command started: %s", args.command)

    try:
        if args.command == "scrape":
            _run_scrape(args)

        elif args.command == "preprocess":
            _run_preprocess(args)

        elif args.command == "analyze1":
            _run_analyze1(args)

        elif args.command == "analyze2":
            _run_analyze2(args)

        elif args.command == "full-pipeline":
            scrape_args = argparse.Namespace(
                url=args.url,
                pages=args.pages,
                start_page=args.start_page,
                output_csv=RAW_DATA_DIR / "autoru_raw.csv",
                output_parquet=RAW_DATA_DIR / "autoru_raw.parquet",
                state_file=RAW_DATA_DIR / "autoru_state.json",
                checkpoint_jsonl=RAW_DATA_DIR / "autoru_checkpoint.jsonl",
                playwright=args.playwright,
                timeout=args.timeout,
                min_delay=args.min_delay,
                max_delay=args.max_delay,
                challenge_cooldown=args.challenge_cooldown,
                max_consecutive_challenge_pages=args.max_consecutive_challenge_pages,
                challenge_retries_per_page=args.challenge_retries_per_page,
                condition_filter=args.condition_filter,
                seller_type_filter=args.seller_type_filter,
                min_mileage=args.min_mileage,
            )
            raw_parquet = _run_scrape(scrape_args)

            preprocess_args = argparse.Namespace(
                input=raw_parquet,
                output_dir=PROCESSED_DATA_DIR,
                exclude_commercial_like=False,
            )
            cleaned_parquet = _run_preprocess(preprocess_args)

            analyze1_args = argparse.Namespace(
                input=cleaned_parquet,
                reports_dir=REPORTS_DIR,
                figures_dir=FIGURES_DIR,
            )
            _run_analyze1(analyze1_args)

            analyze2_args = argparse.Namespace(
                input=cleaned_parquet,
                reports_dir=REPORTS_DIR,
                figures_dir=FIGURES_DIR,
                models_dir=Path("models"),
            )
            _run_analyze2(analyze2_args)

        LOGGER.info("Command finished successfully: %s", args.command)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Command failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
