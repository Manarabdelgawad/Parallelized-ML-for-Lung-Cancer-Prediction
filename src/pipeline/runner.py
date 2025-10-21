import logging
import numpy as np
from sklearn.model_selection import train_test_split
from config.settings import get_config
from src.data.loader import DataLoader
from src.data.preprocess import DataPreprocessor
from src.features.selector import FeatureSelector
from src.models.trainer import ModelTrainer
from src.utils.decorators import timeit, log_process_info
from src.utils.performance import PerformanceComparator

logger = logging.getLogger(__name__)

class PipelineRunner:
    def __init__(self):
        self.config = get_config()
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.feature_selector = FeatureSelector()
        self.model_trainer = ModelTrainer()
        self.performance_comparator = PerformanceComparator()
        
        # Initialize shared memory tracking
        self.shared_memory_objects = []

    @timeit
    @log_process_info
    def run(self):
        print("pipeline starting ")
        
        try:
            # Step 1: Load data 
            df = self._load_data()
            
            # Step 2: Prepare features and target
            X, y = self._prepare_features_target(df)
            
            # Step 3: Preprocess data with shared memory option
            X_processed = self._preprocess_features(X, y)
            
            # Step 4: Feature selection with multiple methods
            X_selected = self._select_features(X_processed, y)
            
            # Step 5: Split data
            X_train, X_test, y_train, y_test = self._split_data(X_selected, y)
            
            # Step 6: Train models with performance comparison
            results = self._train_models_with_comparison(X_train, X_test, y_train, y_test)
            
            # Step 7: Cleanup shared memory
            self._cleanup_shared_memory()
            
            logger.info("Pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self._cleanup_shared_memory()
            raise

    @log_process_info
    def _load_data(self):
        logger.info("Loading data")
        df = self.data_loader.load_data()
        
        logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return df

    @log_process_info
    def _prepare_features_target(self, df):
        logger.info("Preparing features ")
        
        # Extract features
        X = df.drop(self.config.TARGET_COLUMN, axis=1)
        
        if hasattr(X, 'to_pandas') and self.config.DEVICE == 'gpu':
            X_values = X.to_pandas().values
            logger.debug("Converted GPU DataFrame to pandas for processing")
        else:
            X_values = X.values
        
        # Extract and encode target
        y = df[self.config.TARGET_COLUMN].values
        y_encoded = self.preprocessor.encode_target(y)
        
        logger.info(f"Features shape: {X_values.shape}")
        logger.info(f"Target encoded: {len(np.unique(y_encoded))} classes")
        
        return X_values, y_encoded

    @log_process_info
    def _preprocess_features(self, X, y):
        logger.info("Preprocessing features")
        
        # Log preprocessing configuration
        logger.info(f"Preprocessing config: encoding_columns={self.config.ENCODING_COLUMNS}, "
                   f"imputation={self.config.IMPUTATION_STRATEGY}")
        
        if self.config.USE_SHARED_MEMORY and X.size > 100:
            logger.debug("Using shared memory for large dataset")
            
            X_processed = self.preprocessor.preprocess_data(X)
        else:
            X_processed = self.preprocessor.preprocess_data(X)
        
        # Log preprocessing results
        logger.info(f"Preprocessing completed: {X.shape} → {X_processed.shape}")
        
        return X_processed

    @log_process_info
    def _select_features(self, X, y):
        logger.info("Selecting features")
        
        # Log feature selection configuration
        logger.info(f"Feature selection: method={self.config.FEATURE_SELECTION_METHOD}, "
                   f"percentile={self.config.FEATURE_PERCENTILE}%")
        
        # Perform feature selection
        X_selected = self.feature_selector.select_features(X, y)
        
        # Log feature selection results
        feature_reduction = ((X.shape[1] - X_selected.shape[1]) / X.shape[1]) * 100
        logger.info(f"Feature selection completed: {X.shape[1]} → {X_selected.shape[1]} "
                   f"features ({feature_reduction:.1f}% reduction)")
        
        return X_selected

    @log_process_info
    def _split_data(self, X, y):
        logger.info("Splitting data")
        
        split_params = {
            'test_size': self.config.TEST_SIZE,
            'random_state': self.config.RANDOM_STATE,
            'stratify': y  # Important for imbalanced data
        }
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, **split_params)
        
        logger.info(f"Data split: Train={X_train.shape}, Test={X_test.shape}")
        
        
        return X_train, X_test, y_train, y_test

    @log_process_info
    def _train_models_with_comparison(self, X_train, X_test, y_train, y_test):
        """Enhanced model training with performance comparison"""
        logger.info("Training models")
        
        # Log training configuration
        model_names = [model['name'] for model in self.config.MODELS]
        logger.info(f"Training {len(model_names)} models: {', '.join(model_names)}")
        logger.info(f"Parallel jobs: {self.config.N_JOBS}")
        
        results = self.model_trainer.train_models(X_train, X_test, y_train, y_test)
        
        # Log training results summary
        self._log_training_summary(results)
        
        return results

    def _log_training_summary(self, results):
        logger.info("Training Results Summary:")
        
        for name, acc, prec, rec, f1, model in results:
            logger.info(f"  {name:20} | Acc: {acc:.3f} | Prec: {prec:.3f} | "
                       f"Rec: {rec:.3f} | F1: {f1:.3f}")
        
        # Calculate averages
        avg_accuracy = np.mean([r[1] for r in results])
        best_model = max(results, key=lambda x: x[1])
        worst_model = min(results, key=lambda x: x[1])
        
        logger.info(f"Overall - Avg Accuracy: {avg_accuracy:.3f}, "
                   f"Best: {best_model[0]} ({best_model[1]:.3f})"
                  )

    def _cleanup_shared_memory(self):
        """Cleanup shared memory resources"""
        if hasattr(self, 'shared_memory_objects') and self.shared_memory_objects:
            logger.debug("Cleaning up shared memory resources")
            for shm in self.shared_memory_objects:
                try:
                    shm.close()
                    shm.unlink()
                except Exception as e:
                    logger.warning(f"Failed to cleanup shared memory: {e}")
            self.shared_memory_objects.clear()

    