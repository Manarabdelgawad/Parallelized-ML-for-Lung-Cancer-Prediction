from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, ConfigDict
from typing import List, Dict, Any, Literal
import os
import platform

class MLConfig(BaseSettings):
    # Data settings
    DATA_PATH: str = "data/lung_cancer.csv"
    TARGET_COLUMN: str = "LUNG_CANCER"
    
    # Execution settings
    DEVICE: str = "cpu"
    N_JOBS: int = 4
    RANDOM_STATE: int = 42
    
    # Preprocessing settings
    ENCODING_COLUMNS: List[int] = [0, -1]
    IMPUTATION_STRATEGY: Literal["mean", "median", "most_frequent"] = "mean"
    TEST_SIZE: float = 0.2
    
    # Feature selection settings
    FEATURE_SELECTION_METHOD: Literal["chi2", "mutual_info", "f_classif", "variance"] = "chi2"
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
    
    # Performance settings
    ENABLE_BENCHMARK: bool = False  # Set to False by default for stability
    USE_SHARED_MEMORY: bool = False  # Disable for now - complex to implement
    LOG_PROCESS_INFO: bool = True
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_PATH: str = "pipeline.log"
    
    # Pydantic v2 configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="ML_",  # Add prefix for environment variables
        validate_default=True
    )

    @field_validator("N_JOBS", mode="before")
    @classmethod
    def validate_n_jobs(cls, v) -> int:
        """Optimize n_jobs based on platform"""
        try:
            cpu_count = os.cpu_count() or 1
            n_jobs = int(v)
            
            if platform.system() == "Windows":
                return max(1, min(n_jobs, min(4, cpu_count)))
            else:
                return max(1, min(n_jobs, cpu_count))
        except (ValueError, TypeError):
            return 1  # Default fallback

    @field_validator("DEVICE", mode="before")
    @classmethod
    def validate_device(cls, v) -> str:
        """Validate and optimize device setting"""
        if isinstance(v, str):
            v = v.strip().lower()
        
        # Force CPU on Windows for stability
        if platform.system() == "Windows":
            return "cpu"
        
        return v if v in ["cpu", "gpu"] else "cpu"

    @field_validator("TEST_SIZE", mode="before")
    @classmethod
    def validate_test_size(cls, v) -> float:
        """Ensure test size is between 0.1 and 0.5"""
        try:
            size = float(v)
            return max(0.1, min(0.5, size))
        except (ValueError, TypeError):
            return 0.2  # Default fallback

    @field_validator("FEATURE_PERCENTILE", mode="before")
    @classmethod
    def validate_percentile(cls, v) -> int:
        """Ensure percentile is between 10 and 100"""
        try:
            percentile = int(v)
            return max(10, min(100, percentile))
        except (ValueError, TypeError):
            return 80  # Default fallback

    @field_validator("DATA_PATH", mode="before")
    @classmethod
    def validate_data_path(cls, v) -> str:
        """Ensure data path uses correct separators"""
        if isinstance(v, str):
            # Convert to cross-platform path
            return v.replace("\\", "/")
        return "data/lung_cancer.csv"

    @field_validator("MODELS", mode="before")
    @classmethod
    def validate_models(cls, v):
        """Ensure models configuration is valid"""
        if not v or not isinstance(v, list):
            # Return default models if invalid
            return [
                {"name": "SVC", "params": {"kernel": "rbf", "gamma": 0.5, "C": 1.0}},
                {"name": "RandomForest", "params": {"n_estimators": 15}},
                {"name": "LogisticRegression", "params": {"max_iter": 200}}
            ]
        return v


def get_config() -> MLConfig:
    """Get configuration instance with error handling"""
    try:
        return MLConfig()
    except Exception as e:
        print(f"⚠️  Configuration loading failed: {e}")
        print("🔄 Using default configuration...")
        # Create instance with defaults
        return MLConfig(
            DATA_PATH="data/lung_cancer.csv",
            TARGET_COLUMN="LUNG_CANCER",
            DEVICE="cpu",
            N_JOBS=1,  # Conservative default
            ENABLE_BENCHMARK=False  # Disable benchmark for stability
        )