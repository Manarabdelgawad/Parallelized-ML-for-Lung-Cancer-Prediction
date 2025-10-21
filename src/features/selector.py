import numpy as np
import logging
from joblib import Parallel, delayed
from sklearn.feature_selection import SelectPercentile, chi2, mutual_info_classif, f_classif
from config.settings import get_config
from src.utils.decorators import timeit, log_process_info

logger = logging.getLogger(__name__)

class FeatureSelector:
    def __init__(self):
        self.config = get_config()
        self.device = self.config.DEVICE
        
        # Map method names to sklearn functions
        self.selection_methods = {
            'chi2': chi2,
            'mutual_info': mutual_info_classif,
            'f_classif': f_classif,
            'variance': None  #
        }

    def _compute_feature_score(self, feature, y, method):
        """Compute feature score using specified method"""
        if method == 'chi2':
            return chi2(feature.reshape(-1, 1), y)[0][0]
        elif method == 'mutual_info':
            return mutual_info_classif(feature.reshape(-1, 1), y)[0]
        elif method == 'f_classif':
            return f_classif(feature.reshape(-1, 1), y)[0][0]
        else:
            return np.var(feature)  

    @timeit
    @log_process_info
    def select_features(self, X, y):
        logger.info(f"Starting feature selection using {self.config.FEATURE_SELECTION_METHOD}")
        
        method = self.config.FEATURE_SELECTION_METHOD
        
        if method == 'variance':
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=np.percentile(
                np.var(X, axis=0), 
                100 - self.config.FEATURE_PERCENTILE
            ))
            X_selected = selector.fit_transform(X)
        else:
            # Use percentile-based selection
            selector = SelectPercentile(
                score_func=self.selection_methods[method],
                percentile=self.config.FEATURE_PERCENTILE
            )
            X_selected = selector.fit_transform(X, y)
            
            # Log feature scores
            if hasattr(selector, 'scores_'):
                logger.debug(f"Feature scores: {selector.scores_}")
        
        logger.info(f"Feature selection completed: {X.shape} -> {X_selected.shape}")
        return X_selected

    def select_features_parallel(self, X, y):
        """Alternative parallel implementation"""
        logger.info("Starting parallel feature selection")
        
        scores = Parallel(n_jobs=self.config.N_JOBS)(
            delayed(self._compute_feature_score)(
                X[:, i], y, self.config.FEATURE_SELECTION_METHOD
            ) for i in range(X.shape[1])
        )
        
        scores = np.array(scores)
        k = int(X.shape[1] * self.config.FEATURE_PERCENTILE / 100)
        selected_indices = np.argsort(scores)[-k:]
        X_selected = X[:, selected_indices]
        
        logger.info(f"Parallel feature selection completed: {X_selected.shape}")
        return X_selected