"""Typed, validated configuration loaded from YAML at startup."""

from pathlib import Path
from typing import Literal

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


def load_data_config(path: Path) -> DataConfig:
    """Load and validate the data config; raises pydantic.ValidationError on bad input."""
    return DataConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
