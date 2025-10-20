import numpy as np
import pandas as pd
import logging
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder

from config.settings import get_config
from src.utils.decorators import timeit

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.config = get_config()
        self.device = self.config.DEVICE

    def _preprocess_column(self, col_idx, X):
        X_col = X[:, col_idx].copy()
        if col_idx in self.config.ENCODING_COLUMNS:
            le = LabelEncoder()
            X_col = le.fit_transform(X_col)
        X_col = pd.Series(X_col).fillna(pd.Series(X_col).mean()).values
        return X_col

    @timeit
    def preprocess_data(self, X, y=None):
        logger.info("Starting data preprocessing")
        
        X_processed = np.array(X)

        # Parallel preprocessing
        processed_columns = Parallel(n_jobs=self.config.N_JOBS)(
            delayed(self._preprocess_column)(i, X_processed) 
            for i in range(X_processed.shape[1])
        )

        # Stack columns and transpose
        X_processed = np.array(processed_columns).T
            
        logger.info(f"Data preprocessing completed: {X_processed.shape}")
        return X_processed

    def encode_target(self, y):
        le = LabelEncoder()
        return le.fit_transform(y)