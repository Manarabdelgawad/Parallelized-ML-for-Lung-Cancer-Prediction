import logging
from sklearn.model_selection import train_test_split
from config.settings import get_config
from src.data.loader import DataLoader
from src.data.preprocess import DataPreprocessor  # Fixed import

from src.features.selector import FeatureSelector
from src.models.trainer import ModelTrainer
from src.utils.decorators import timeit

logger = logging.getLogger(__name__)

class PipelineRunner:
    def __init__(self):
        self.config = get_config()
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.feature_selector = FeatureSelector()
        self.model_trainer = ModelTrainer()

    @timeit
    def run(self):
        logger.info("Starting ML pipeline")
        
        # Load data
        df = self.data_loader.load_data()
        
        # Prepare features and target
        X = df.drop(self.config.TARGET_COLUMN, axis=1)
        if self.config.DEVICE == 'gpu':
            X = X.to_pandas().values
        else:
            X = X.values
            
        y = df[self.config.TARGET_COLUMN].values
        
        # Preprocess target
        y = self.preprocessor.encode_target(y)
        
        # Preprocess features
        X_processed = self.preprocessor.preprocess_data(X)
        
        # Feature selection
        X_selected = self.feature_selector.select_features(X_processed, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, 
            test_size=self.config.TEST_SIZE, 
            random_state=self.config.RANDOM_STATE
        )
        
        # Train models
        results = self.model_trainer.train_models(X_train, X_test, y_train, y_test)
        
        logger.info("Pipeline completed successfully")
        return results