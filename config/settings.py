from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import List, Dict, Any
import os

class MLConfig(BaseSettings):
    # Data settings
    DATA_PATH: str = "data/lung_cancer.csv"
    TARGET_COLUMN: str = "LUNG_CANCER"
    
    DEVICE: str = "cpu"
    N_JOBS: int = 4
    RANDOM_STATE: int = 42
    
    # Preprocessing settings
    ENCODING_COLUMNS: List[int] = [0, -1]
    IMPUTATION_STRATEGY: str = "mean"
    TEST_SIZE: float = 0.2
    
    # Feature selection settings
    FEATURE_SELECTION_METHOD: str = "chi2"
    FEATURE_PERCENTILE: int = 80
    
    # Model settings
    MODELS: List[Dict[str, Any]] = [
        {
            "name": "SVC",
            "params": {"kernel": "rbf", "gamma": 0.5, "C": 1.0}
        },
        {
            "name": "RandomForest", 
            "params": {"n_estimators": 15}
        },
        {
            "name": "LogisticRegression",
            "params": {"max_iter": 200}
        }
    ]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_PATH: str = "pipeline.log"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    @field_validator("DEVICE", mode="before")
    @classmethod
    def validate_device(cls, v):
        import platform
        if platform.system() == "Windows":
            return "cpu"
        v = str(v).strip().lower()
        return "cpu" if v in ["cpu", "gpu"] else "cpu"

def get_config() -> MLConfig:
    return MLConfig()
