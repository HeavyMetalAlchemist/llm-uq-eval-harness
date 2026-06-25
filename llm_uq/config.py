from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    name: str = "microsoft/Phi-3-mini-4k-instruct"
    quantize_4bit: bool = True
    device_map: str = "auto"
    max_new_tokens: int = 512
    chunk_size: int = 128


class DatasetConfig(BaseModel):
    source: Literal["huggingface", "csv"] = "huggingface"
    name: str  # HF dataset name or path to CSV
    split: str = "test"
    input_col: str = "question"
    output_col: str = "answer"
    seed: int = 42
    n_samples: int = 200
    scoring: Literal["numeric", "exact_match", "contains", "f1", "cosine"] = "numeric"
    # for cosine scoring: minimum similarity to count as correct
    cosine_threshold: float = 0.7
    # for HF datasets that need a config name (e.g. gsm8k "main", triviaqa "rc")
    hf_config: Optional[str] = None


class PromptConfig(BaseModel):
    system: Optional[str] = None
    system_file: Optional[Path] = None

    @model_validator(mode="after")
    def resolve_system(self) -> "PromptConfig":
        if self.system_file is not None:
            self.system = self.system_file.read_text().strip()
        return self

    def get_system(self) -> Optional[str]:
        return self.system


class MetricsConfig(BaseModel):
    names: list[str] = Field(
        default=["confidence", "entropy", "std_logp", "auroc", "brier", "ece", "risk_coverage"]
    )


class OutputConfig(BaseModel):
    format: Literal["html", "markdown"] = "html"
    path: Path = Path("report.html")
    save_csv: bool = True


class RunConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset: DatasetConfig
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        raw = yaml.safe_load(Path(path).read_text())
        # allow metrics as a plain list at top level
        if "metrics" in raw and isinstance(raw["metrics"], list):
            raw["metrics"] = {"names": raw["metrics"]}
        return cls.model_validate(raw)
