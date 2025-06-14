from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel
from yaml import safe_load

import titanic_model

# Project Directories
PACKAGE_ROOT = Path(titanic_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
LOG_DIR = PACKAGE_ROOT / "logs"


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    training_data_file: str
    test_data_file: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    target: str
    features: List[str]
    variables_to_drop: Sequence[str]
    categorical_vars: Sequence[str]
    numerical_vars: Sequence[str]
    categorical_vars_with_na: List[str]
    numerical_vars_with_na: List[str]
    var_for_letter_extraction: List[str]
    test_size: float
    random_state: int


class TuneConfig(BaseModel):
    """
    All configuration relevant to model
    tuning and optimization.
    """

    strategy: str
    random_forest_params: Dict[str, List[Any]]


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config_params: ModelConfig
    tune_config: TuneConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise FileNotFoundError(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> Dict[str, Any]:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path) as conf_file:
            parsed_config = safe_load(conf_file.read())
            if not isinstance(parsed_config, dict):
                raise ValueError("Expected YAML config to be a dictionary")
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: Optional[Dict[str, Any]] = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config),
        model_config_params=ModelConfig(**parsed_config),
        tune_config=TuneConfig(**parsed_config),
    )

    return _config


config = create_and_validate_config()
