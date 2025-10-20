import logging
import os
from multiprocessing import Pool
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config.settings import get_config
from src.utils.decorators import timeit

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.config = get_config()
        self.device = self.config.DEVICE
        
        # CPU-only models
        self.models = {
            'SVC': SVC,
            'RandomForest': RandomForestClassifier,
            'LogisticRegression': LogisticRegression
        }
        self.metrics = {
            'accuracy': accuracy_score,
            'precision': precision_score,
            'recall': recall_score,
            'f1': f1_score
        }

    def _train_single_model(self, model_config, X_train, X_test, y_train, y_test):
        model_name = model_config['name']
        logger.info(f"Training {model_name} (Process ID: {os.getpid()})")
        
        try:
            model_class = self.models[model_name]
            model = model_class(**model_config['params'])
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            
            acc = self.metrics['accuracy'](y_test, pred)
            prec = self.metrics['precision'](y_test, pred, pos_label=1)
            rec = self.metrics['recall'](y_test, pred, pos_label=1)
            f1 = self.metrics['f1'](y_test, pred, pos_label=1)
            
            logger.info(f"{model_name} - Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1-Score: {f1:.2f}")
            return (model_name, acc, prec, rec, f1, model)
            
        except Exception as e:
            logger.error(f"Training failed for {model_name}: {e}")
            return (model_name, 0, 0, 0, 0, None)

    @timeit
    def train_models(self, X_train, X_test, y_train, y_test):
        logger.info("Starting model training")
        
        with Pool(processes=self.config.N_JOBS) as pool:
            results = [
                pool.apply_async(
                    self._train_single_model, 
                    args=(model, X_train, X_test, y_train, y_test)
                ) for model in self.config.MODELS
            ]
            return [r.get() for r in results]