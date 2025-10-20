import numpy as np
import logging
from joblib import Parallel, delayed
from sklearn.feature_selection import chi2
from config.settings import get_config
from src.utils.decorators import timeit

logger = logging.getLogger(__name__)

class FeatureSelector:
    def __init__(self):
        self.config = get_config()
        self.device = self.config.DEVICE
        
        if self.device == 'gpu':
            try:
                import cupy as cp
                self.cp = cp
            except ImportError:
                logger.warning("CuPy not available, falling back to CPU")
                self.device = 'cpu'

    def _compute_feature_score(self, feature, y):
        return chi2(feature.reshape(-1, 1), y)[0][0]

    @timeit
    def select_features(self, X, y):
        logger.info("Starting feature selection")
        
        if self.device == 'gpu':
            X_arr = self.cp.array(X)
            y_arr = self.cp.array(y)
        else:
            X_arr = np.array(X)
            y_arr = np.array(y)

        # Compute feature scores in parallel
        scores = Parallel(n_jobs=self.config.N_JOBS)(
            delayed(self._compute_feature_score)(
                X_arr[:, i].get() if self.device == 'gpu' else X_arr[:, i],
                y_arr.get() if self.device == 'gpu' else y_arr
            ) for i in range(X_arr.shape[1])
        )

        scores = np.array(scores)
        selected_indices = np.argsort(scores)[-int(len(scores) * self.config.FEATURE_PERCENTILE / 100):]
        X_selected = X_arr[:, selected_indices]
        
        if self.device == 'gpu':
            X_selected = self.cp.asnumpy(X_selected)
            
        logger.info(f"Feature selection completed: {X_selected.shape}")
        return X_selected