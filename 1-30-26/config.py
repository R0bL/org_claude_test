"""
Entity Resolution Pipeline Configuration (Minimal)
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class DatabaseConfig:
    """AWS Athena connection configuration."""


@dataclass
class TableConfig:
    """Table names"""
    DIM_ORG_TABLE: str = "allsci_prod_gold.dim_organization"
    GRID_TABLE: str = "allsci_prod_legacy_data_silver.grid"
    MISMATCHED_TABLE: str = "allsci_prod_gold.potential_mismatched_organizations"


@dataclass
class SamplingConfig:
    """Sample sizes"""
    TRAINING_DIM_ORG_SAMPLE: int = 110_000
    TRAINING_GRID_SAMPLE: int = 110_000
    VALIDATION_HOLDOUT_RATIO: float = 0.2
    U_PROBABILITY_MAX_PAIRS: float = 1e7


@dataclass
class MatchingConfig:
    """Splink matching parameters"""
    THRESHOLD_PREDICTION: float = 0.50
    CLUSTER_THRESHOLD: float = 0.85


@dataclass
class FilterConfig:
    """Record filtering"""
    MIN_NAME_LENGTH: int = 5
    MAX_NAME_LENGTH: int = 200
    GENERIC_PATTERNS: List[str] = field(default_factory=lambda: [
        'research site', 'study site', 'investigative site', 'clinical site',
        'site name', 'not available', 'n/a', 'unknown', 'tbd', 'to be determined',
    ])
    PERSON_RECORD_TYPE: str = "SPONSOR_INVESTIGATOR"


@dataclass
class BlockingConfig:
    """Blocking rule definitions"""
    NAME_PREFIX_SHORT: int = 5
    NAME_PREFIX_LONG: int = 10


@dataclass
class PathConfig:
    """File paths"""
    BASE_DIR: str = "/Users/robertlalani/Desktop/entity_resolution_12_18_25/1-30-26"

    @property
    def DATA_DIR(self) -> str:
        return os.path.join(self.BASE_DIR, "data")

    @property
    def MODELS_DIR(self) -> str:
        return os.path.join(self.BASE_DIR, "models")

    @property
    def MODEL_FILE(self) -> str:
        return os.path.join(self.MODELS_DIR, "model_v1.json")


@dataclass
class LLMJudgeConfig:
    """LLM judge configuration"""
    ENABLE_LLM_VALIDATION: bool = True
    PRIMARY_MODEL: str = "claude-sonnet-4-20250514"


@dataclass
class Config:
    """Master configuration"""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    tables: TableConfig = field(default_factory=TableConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    filtering: FilterConfig = field(default_factory=FilterConfig)
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    llm_judge: LLMJudgeConfig = field(default_factory=LLMJudgeConfig)

    def __post_init__(self):
        os.makedirs(self.paths.DATA_DIR, exist_ok=True)
        os.makedirs(self.paths.MODELS_DIR, exist_ok=True)


config = Config()

