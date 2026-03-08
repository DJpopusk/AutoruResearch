"""Stage 2 wrapper for regression modeling."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.modeling.regression import RegressionConfig, run_regression_experiment
from src.utils.constants import FIGURES_DIR, REPORTS_DIR
from src.utils.io_utils import load_tabular

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Stage2Config:
    """Configuration for stage 2 CLI entrypoint."""

    input_path: Path
    reports_dir: Path = REPORTS_DIR
    figures_dir: Path = FIGURES_DIR
    models_dir: Path = Path("models")


def run_stage2_analysis(config: Stage2Config) -> None:
    """Load cleaned data, train models, and save stage-2 artifacts."""
    LOGGER.info("Loading cleaned dataset from %s", config.input_path)
    df = load_tabular(config.input_path)
    if df.empty:
        raise ValueError("Input dataset is empty")

    reg_config = RegressionConfig(
        reports_dir=config.reports_dir,
        figures_dir=config.figures_dir,
        models_dir=config.models_dir,
    )
    run_regression_experiment(df, reg_config)
