"""Typed, validated configuration loaded from YAML at startup."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DataConfig(BaseModel):
    """Settings for obtaining and describing the dataset (configs/data.yaml)."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["synthetic", "csv"]
    csv_path: Path | None = None
    seed: int
    n_rows: int = Field(gt=0)
    schema_version: str
    target: str
    id_column: str

    @model_validator(mode="after")
    def _csv_needs_path(self) -> "DataConfig":
        if self.source == "csv" and self.csv_path is None:
            raise ValueError("data.source is 'csv' but csv_path is not set")
        return self


class BaselineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["dummy_prior"]


class CandidateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["logistic_regression"]
    params: dict[str, Any] = {}


class GatesConfig(BaseModel):
    """Minimum acceptance thresholds; the candidate is ineligible if any is missed."""

    model_config = ConfigDict(extra="forbid")
    min_roc_auc: float = Field(ge=0, le=1)
    min_pr_auc: float = Field(ge=0, le=1)
    min_recall: float = Field(ge=0, le=1)
    min_roc_auc_lift_over_baseline: float


class ModelConfig(BaseModel):
    """Training and evaluation settings (configs/model.yaml)."""

    model_config = ConfigDict(extra="forbid")

    seed: int
    test_size: float = Field(gt=0, lt=1)
    decision_threshold: float = Field(gt=0, lt=1)
    baseline: BaselineConfig
    candidate: CandidateConfig
    gates: GatesConfig


def load_model_config(path: Path) -> ModelConfig:
    """Load and validate the model config; raises pydantic.ValidationError on bad input."""
    return ModelConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_data_config(path: Path) -> DataConfig:
    """Load and validate the data config; raises pydantic.ValidationError on bad input."""
    return DataConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
