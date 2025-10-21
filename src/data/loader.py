import pandas as pd
import logging
from config.settings import get_config
from src.utils.decorators import timeit

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self):
        self.config = get_config()
        self.device = self.config.DEVICE

    @timeit
    def load_data(self):
        logger.info(f"Loading data from {self.config.DATA_PATH}")
        try:
           
            df = pd.read_csv(self.config.DATA_PATH, na_values=['', ' ', 'NA', 'N/A', 'null', 'NULL'])
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Log data info
            logger.info(f"Data loaded successfully: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            logger.info(f"Data types:\n{df.dtypes}")
            logger.info(f"Missing values:\n{df.isnull().sum()}")
            
            # Basic data validation
            if self.config.TARGET_COLUMN not in df.columns:
                raise ValueError(f"Target column '{self.config.TARGET_COLUMN}' not found in data")
            
            return df
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise