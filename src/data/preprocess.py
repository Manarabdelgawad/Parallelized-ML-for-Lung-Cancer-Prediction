import numpy as np
import pandas as pd
import logging
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from config.settings import get_config
from src.utils.decorators import timeit
import warnings

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.config = get_config()
        self.device = self.config.DEVICE

    def _safe_isnan(self, arr):
        """Safely check for NaN values in any array type"""
        try:
            if hasattr(arr, 'dtype') and np.issubdtype(arr.dtype, np.number):
                return np.isnan(arr)
            else:
                if hasattr(arr, 'isna'):  
                    return arr.isna()
                else:
                    arr_obj = np.asarray(arr, dtype=object)
                    isnan_mask = np.zeros(len(arr_obj), dtype=bool)
                    for i, val in enumerate(arr_obj):
                        isnan_mask[i] = pd.isna(val) if hasattr(pd, 'isna') else (val is None or (isinstance(val, float) and np.isnan(val)))
                    return isnan_mask
        except Exception as e:
            logger.warning(f"Error checking NaN values: {e}")
            return np.zeros(len(arr), dtype=bool)

    def _convert_to_numeric(self, arr):
        """Safely convert array to numeric, handling errors"""
        try:
            if hasattr(arr, 'astype'):
                return arr.astype(float)
            else:
                series = pd.Series(arr)
                return pd.to_numeric(series, errors='coerce').values
        except Exception as e:
            logger.warning(f"Error converting to numeric: {e}")
            return np.zeros(len(arr))

    def _preprocess_column(self, col_idx, X):

        try:
            X_col = X[:, col_idx].copy()
            
            col_dtype = str(type(X_col[0])) if len(X_col) > 0 else 'unknown'
            logger.debug(f"Processing column {col_idx}, dtype: {col_dtype}, sample: {X_col[0] if len(X_col) > 0 else 'empty'}")
            
            if col_idx in self.config.ENCODING_COLUMNS:
                le = LabelEncoder()
                # Convert to string and handle NaN values
                X_col_str = np.array([str(x) if not pd.isna(x) else 'missing' for x in X_col])
                X_col = le.fit_transform(X_col_str)
                logger.debug(f"Encoded column {col_idx} to numeric")
            else:

                if not hasattr(X_col, 'dtype') or not np.issubdtype(X_col.dtype, np.number):
                    X_col = self._convert_to_numeric(X_col)
                    logger.debug(f"Converted column {col_idx} to numeric")
            
            # Handle missing values 
            missing_mask = self._safe_isnan(X_col)
            if np.any(missing_mask):
                logger.debug(f"Found {np.sum(missing_mask)} missing values in column {col_idx}")
                
                # Use SimpleImputer for imputation
                imputer = SimpleImputer(strategy=self.config.IMPUTATION_STRATEGY)
                
                # Reshape for sklearn 
                X_col_2d = X_col.reshape(-1, 1)
                X_col_imputed = imputer.fit_transform(X_col_2d).flatten()
                
                # Ensure we have the right data type
                X_col = X_col_imputed.astype(float)
                logger.debug(f"Imputed missing values in column {col_idx}")
            
            return X_col
            
        except Exception as e:
            logger.error(f"Error preprocessing column {col_idx}: {e}")
            return np.zeros(len(X[:, col_idx]))

    @timeit
    def preprocess_data(self, X, y=None):

        logger.info("Starting data preprocessing")
        
        try:
            if hasattr(X, 'values'):  
                X_array = X.values
            elif hasattr(X, 'to_numpy'):  
                X_array = X.to_numpy()
            else:
                X_array = np.array(X)
            
            logger.info(f"Input data shape: {X_array.shape}, dtype: {X_array.dtype}")
            
            if X_array.dtype == object:
                logger.info("Converting object array to string for safe processing")
                X_array = np.array([[str(x) if not pd.isna(x) else 'missing' for x in row] for row in X_array])
            
            # Parallel preprocessing 
            processed_columns = []
            for i in range(X_array.shape[1]):
                try:
                    processed_col = self._preprocess_column(i, X_array)
                    processed_columns.append(processed_col)
                except Exception as e:
                    logger.error(f"Failed to process column {i}: {e}")
                    processed_columns.append(X_array[:, i].copy())
            
            # Stack columns and transpose
            X_processed = np.column_stack(processed_columns)
            
            X_processed = pd.DataFrame(X_processed).apply(pd.to_numeric, errors='coerce').fillna(0).values
            
            logger.info(f"Data preprocessing completed: {X_processed.shape}")
            logger.debug(f"Processed data - Mean: {np.mean(X_processed):.3f}, "
                        f"Std: {np.std(X_processed):.3f}, "
                        f"NaN count: {np.isnan(X_processed).sum()}")
            
            return X_processed
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            return X

    def encode_target(self, y):

        try:
            le = LabelEncoder()
            y_clean = np.array([str(val) if not pd.isna(val) else 'missing' for val in y])
            y_encoded = le.fit_transform(y_clean)
            
            logger.info(f"Target encoded: {len(np.unique(y_encoded))} classes")
            return y_encoded
            
        except Exception as e:
            logger.error(f"Target encoding failed: {e}")
            try:
                return np.array([1 if str(val).lower() in ['yes', '1', 'true'] else 0 for val in y])
            except:
                return np.zeros(len(y))