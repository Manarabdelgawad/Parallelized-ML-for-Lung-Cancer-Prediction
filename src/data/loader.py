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
            df = pd.read_csv(self.config.DATA_PATH)
            logger.info(f"Data loaded successfully: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise