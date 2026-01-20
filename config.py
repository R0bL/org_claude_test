"""
Entity Resolution Pipeline Configuration
=========================================

Central configuration for all pipeline parameters.
Edit this file to adjust thresholds, paths, and settings.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class DatabaseConfig:
    """
    AWS Athena connection configuration.

    IMPORTANT: Do not hardcode credentials in this repo. Use environment variables
    (or AWS SSO/instance profiles/role-based auth) instead.
    """

    AWS_ACCESS_KEY_ID: str = os.getenv('AWS_ACCESS_KEY_ID', '')
    AWS_SECRET_ACCESS_KEY: str = os.getenv('AWS_SECRET_ACCESS_KEY', '')
    AWS_REGION: str = os.getenv('AWS_REGION', 'us-east-1')
    S3_STAGING_DIR: str = os.getenv('S3_STAGING_DIR', 's3://allsci-athena-results/')




@dataclass
class TableConfig:
    """Table names and schemas"""
    DIM_ORG_TABLE: str = "allsci_prod_gold.dim_organization"
    GRID_TABLE: str = "allsci_prod_legacy_data_silver.grid"
    MISMATCHED_TABLE: str = "allsci_prod_gold.potential_mismatched_organizations"


@dataclass
class SamplingConfig:
    """Sample sizes for different stages"""
    # Exploration phase
    EXPLORATION_SAMPLE_PER_SOURCE: int = 500
    
    # Training phase
    TRAINING_DIM_ORG_SAMPLE: int = 110_000
    TRAINING_GRID_SAMPLE: int = 110_000
    VALIDATION_HOLDOUT_RATIO: float = 0.2
    
    # Inference phase
    INFERENCE_CHUNK_SIZE: int = 100_000
    
    # Random sampling for u-probability estimation
    U_PROBABILITY_MAX_PAIRS = 1e7


@dataclass
class MatchingConfig:
    """Splink matching parameters"""
    # Match probability thresholds
    THRESHOLD_HIGH_CONFIDENCE: float = 0.95
    THRESHOLD_MEDIUM_CONFIDENCE: float = 0.85
    THRESHOLD_LOW_CONFIDENCE: float = 0.70
    THRESHOLD_PREDICTION: float = 0.50  # Minimum to consider a match
    
    # Jaro-Winkler thresholds for name matching
    JW_THRESHOLD_HIGH: float = 0.95
    JW_THRESHOLD_MEDIUM: float = 0.88
    JARO_THRESHOLD_LOW: float = 0.80
    
    # Geographic distance thresholds (km)
    GEO_DISTANCE_CLOSE: int = 1
    GEO_DISTANCE_MEDIUM: int = 10
    GEO_DISTANCE_FAR: int = 50
    
    # Clustering threshold
    CLUSTER_THRESHOLD: float = 0.85


@dataclass 
class BlockingConfig:
    """Blocking rule definitions"""
    # Name prefix lengths for blocking keys
    NAME_PREFIX_SHORT: int = 5
    NAME_PREFIX_LONG: int = 10
    
    # Blocking rules (SQL format for Splink)
    BLOCKING_RULES: List[str] = field(default_factory=lambda: [
        "l.name_prefix_5 = r.name_prefix_5",
        "l.name_prefix_10 = r.name_prefix_10", 
        "l.country_code = r.country_code AND l.country_code IS NOT NULL",
    ])
    
    # Training blocking rules (tighter for efficiency)
    TRAINING_BLOCKING_RULES: List[str] = field(default_factory=lambda: [
        "l.name_normalized = r.name_normalized",
    ])


@dataclass
class FilterConfig:
    """Record filtering configuration"""
    # Minimum name length to keep
    MIN_NAME_LENGTH: int = 5
    
    # Maximum name length (very long names are suspicious)
    MAX_NAME_LENGTH: int = 200
    
    # Generic name patterns to filter out
    GENERIC_PATTERNS: List[str] = field(default_factory=lambda: [
        'research site',
        'study site', 
        'investigative site',
        'clinical site',
        'site name',
        'not available',
        'n/a',
        'unknown',
        'tbd',
        'to be determined',
    ])
    
    # Person record indicator (from sponsors source)
    PERSON_RECORD_TYPE: str = "SPONSOR_INVESTIGATOR"


@dataclass
class PathConfig:
    """File paths for outputs and checkpoints"""
    BASE_DIR: str = "/Users/robertlalani/Desktop/entity_resolution_12_18_25/org_claude_test"
    
    @property
    def DATA_DIR(self) -> str:
        return os.path.join(self.BASE_DIR, "data")
    
    @property
    def MODELS_DIR(self) -> str:
        return os.path.join(self.BASE_DIR, "models")
    
    # Checkpoint files
    @property
    def EXPLORATION_SUMMARY(self) -> str:
        return os.path.join(self.DATA_DIR, "exploration_summary.json")
    
    @property
    def TRAINING_DATA(self) -> str:
        return os.path.join(self.DATA_DIR, "training_data.parquet")
    
    @property
    def GROUND_TRUTH(self) -> str:
        return os.path.join(self.DATA_DIR, "ground_truth.parquet")
    
    @property
    def MODEL_FILE(self) -> str:
        return os.path.join(self.MODELS_DIR, "model_v1.json")
    
    @property
    def TRAINING_REPORT(self) -> str:
        return os.path.join(self.DATA_DIR, "training_report.html")
    
    @property
    def PREDICTIONS(self) -> str:
        return os.path.join(self.DATA_DIR, "predictions.parquet")
    
    @property
    def CLUSTERS(self) -> str:
        return os.path.join(self.DATA_DIR, "clusters.parquet")
    
    @property
    def NO_MATCH(self) -> str:
        return os.path.join(self.DATA_DIR, "no_match.parquet")
    
    @property
    def FINAL_MATCHES(self) -> str:
        return os.path.join(self.DATA_DIR, "final_matches.csv")


@dataclass
class ActiveLearningConfig:
    """Active learning configuration for LLM validation"""
    # Enable active learning (if False, validate all predictions)
    ENABLE_ACTIVE_LEARNING: bool = True

    # Budget for LLM validation calls
    VALIDATION_BUDGET: int = 500  # Number of predictions to validate

    # Minimum samples per source table
    MIN_SAMPLES_PER_SOURCE: int = 10

    # Source-specific priority weights (higher = more important to validate)
    # None = auto-calculate based on inverse frequency
    SOURCE_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        # Rare/difficult sources get higher weight
        'chinese_clinical_trials_silver.trials': 0.9,
        'chinese_clinical_trials_silver.trial_contacts': 0.9,
        'legacy_alpha_silver._organizations_consolidated': 0.8,
        # Common sources get lower weight
        'open_fda_silver.ndc_drugs': 0.4,
        'uspto_silver.patents': 0.4,
        'nih_clinical_trials_gov_silver.cl_trial_locations': 0.5,
    })


@dataclass
class LLMJudgeConfig:
    """LLM judge configuration"""
    # Enable LLM validation
    ENABLE_LLM_VALIDATION: bool = True

    # Model selection
    PRIMARY_MODEL: str = "claude-sonnet-4-20250514"  # Most capable
    FAST_MODEL: str = "claude-3-5-haiku-20241022"    # Cheaper/faster

    # Bias mitigation
    ENABLE_POSITION_BIAS_MITIGATION: bool = False  # Doubles API calls
    ENABLE_ENSEMBLE: bool = False                   # Uses multiple models

    # Parallel processing
    MAX_WORKERS: int = 20  # Number of concurrent API calls

    # Confidence calibration
    ENABLE_CALIBRATION: bool = True
    MIN_SAMPLES_FOR_CALIBRATION: int = 10


@dataclass
class MultiAgentConfig:
    """Multi-agent validation configuration"""
    # Enable multi-agent routing (vs always using full LLM)
    ENABLE_MULTI_AGENT: bool = True

    # Thresholds for routing decisions
    # Use fast/deterministic agents for obvious cases
    OBVIOUS_MATCH_THRESHOLD: float = 0.98  # Very high Splink score
    OBVIOUS_REJECT_THRESHOLD: float = 0.60  # Very low Splink score

    # Use ensemble for critical boundary cases
    ENSEMBLE_LOWER_BOUND: float = 0.80
    ENSEMBLE_UPPER_BOUND: float = 0.90


@dataclass
class TrainingConfig:
    """Training configuration including hard negatives"""
    # Hard negative mining
    USE_HARD_NEGATIVES: bool = True
    HARD_NEGATIVE_RATIO: float = 0.5  # 50% hard, 50% random

    # Hard negative strategies
    HARD_NEG_STRATEGIES: List[str] = field(default_factory=lambda: [
        'same_country',
        'similar_names',
        'same_city',
        'token_overlap'
    ])

    # Total negative samples
    TOTAL_NEGATIVE_SAMPLES: int = 100_000


@dataclass
class FeedbackConfig:
    """Human feedback collection configuration"""
    # Enable feedback collection
    ENABLE_FEEDBACK: bool = True

    # Feedback file paths
    FEEDBACK_FILE: str = "data/feedback.jsonl"
    FEEDBACK_CONFIG_FILE: str = "data/feedback_config.json"

    # Review candidate selection
    REVIEW_PRIORITY: str = 'disagreements'  # or 'uncertainty', 'diversity'
    MAX_REVIEW_CANDIDATES: int = 100

    # Disagreement thresholds
    HIGH_SPLINK_THRESHOLD: float = 0.95  # High Splink but LLM rejects
    LOW_SPLINK_THRESHOLD: float = 0.60   # Low Splink but LLM confirms


@dataclass
class Config:
    """Master configuration combining all sub-configs"""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    tables: TableConfig = field(default_factory=TableConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    filtering: FilterConfig = field(default_factory=FilterConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    active_learning: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    llm_judge: LLMJudgeConfig = field(default_factory=LLMJudgeConfig)
    multi_agent: MultiAgentConfig = field(default_factory=MultiAgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)

    def __post_init__(self):
        """Ensure directories exist"""
        os.makedirs(self.paths.DATA_DIR, exist_ok=True)
        os.makedirs(self.paths.MODELS_DIR, exist_ok=True)


# Default config instance
config = Config()


if __name__ == "__main__":
    print("Entity Resolution Pipeline Configuration")
    print("=" * 50)
    print(f"\nDatabase: {config.database.AWS_REGION}")
    print(f"Tables: {config.tables.DIM_ORG_TABLE}")
    print(f"Training sample: {config.sampling.TRAINING_DIM_ORG_SAMPLE:,}")
    print(f"Match threshold: {config.matching.THRESHOLD_PREDICTION}")
    print(f"Output dir: {config.paths.DATA_DIR}")

