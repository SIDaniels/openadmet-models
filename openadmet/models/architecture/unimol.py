
from typing import Any, ClassVar, Optional, Union
import tempfile
import os

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import field_validator
from unimol_tools import MolTrain, MolPredict

from openadmet.models.architecture.model_base import PickleableModelBase
from openadmet.models.architecture.model_base import models as model_registry


@model_registry.register("UniMolModel")
class UniMolModel(PickleableModelBase):
    """
    UniMol model wrapper for ADMET prediction
    
    Simple wrapper that directly uses UniMol's native API for training and prediction.
    """
    
    type: ClassVar[str] = "UniMolModel"
    
    # UniMol parameters (matching official API)
    task: str = "classification"
    data_type: str = "molecule" 
    epochs: int = 10
    batch_size: int = 16
    metrics: str = "auc"  # auc for classification, mse/mae for regression
    model_name: str = "unimolv1"  # unimolv1, unimolv2
    model_size: str = "84m"  # 84m, 164m, 310m, 570m, 1.1B (for unimolv2)
    remove_hs: bool = False
    
    # Internal state
    _exp_path: Optional[str] = None
    
    @field_validator("task")
    @classmethod
    def validate_task(cls, value):
        allowed = ["classification", "regression"]
        if value not in allowed:
            raise ValueError(f"Task must be one of {allowed}")
        return value
    
    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value):
        allowed = ["unimolv1", "unimolv2"]
        if value not in allowed:
            raise ValueError(f"Model name must be one of {allowed}")
        return value
    
    @field_validator("model_size")
    @classmethod
    def validate_model_size(cls, value):
        allowed = ["84m", "164m", "310m", "570m", "1.1B"]
        if value not in allowed:
            raise ValueError(f"Model size must be one of {allowed}")
        return value
    
    @classmethod
    def from_params(cls, class_params: dict = None, mod_params: dict = None):
        """Create model instance from parameters"""
        if class_params is None:
            class_params = {}
        if mod_params is None:
            mod_params = {}
            
        all_params = {**class_params, **mod_params}
        instance = cls(**all_params)
        instance.build()
        return instance
    
    def build(self, **kwargs):
        """Build the UniMol model (create trainer instance)"""
        logger.info(f"Building UniMol {self.model_name} for {self.task}")
        
        # Create UniMol trainer with current parameters
        self._estimator = MolTrain(
            task=self.task,
            data_type=self.data_type,
            epochs=self.epochs,
            batch_size=self.batch_size,
            metrics=self.metrics,
            model_name=self.model_name,
            model_size=self.model_size,
            remove_hs=self.remove_hs,
            split='random',  # Use random split instead of cross-validation for small datasets
            kfold=3,         # Reduce k-fold to 3 for smaller datasets
        )
        
        logger.info("UniMol model built successfully")
    
    def train(self, train_data: Union[str, pd.DataFrame], **kwargs):
        """
        Train the UniMol model using the official API
        
        Args:
            train_data: Training data (CSV file path or DataFrame with SMILES)
        """
        if self._estimator is None:
            self.build()
        
        logger.info("Starting UniMol training...")
        
        # Handle DataFrame input by saving to temporary CSV
        if isinstance(train_data, pd.DataFrame):
            temp_file = os.path.join(tempfile.gettempdir(), "unimol_train_data.csv")
            train_data.to_csv(temp_file, index=False)
            train_data = temp_file
        
        # Train using UniMol's native method
        self._estimator.fit(data=train_data)
        # Get the save path from the estimator
        self._exp_path = getattr(self._estimator, 'exp_path', getattr(self._estimator, 'save_path', './exp'))
        
        logger.info(f"UniMol training completed. Model saved to: {self._exp_path}")
    
    def predict(self, data: Union[str, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using trained UniMol model
        
        Args:
            data: Input data (CSV file path or DataFrame with SMILES)
            
        Returns:
            Predictions as numpy array
        """
        if self._exp_path is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        logger.info("Making UniMol predictions...")
        
        # Handle DataFrame input
        if isinstance(data, pd.DataFrame):
            temp_file = os.path.join(tempfile.gettempdir(), "unimol_predict_data.csv")
            data.to_csv(temp_file, index=False)
            data = temp_file
        
        # Create predictor and predict
        predictor = MolPredict(load_model=self._exp_path)
        results = predictor.predict(data=data)
        
        # Convert to numpy array
        if isinstance(results, pd.DataFrame):
            # Find prediction column
            pred_cols = [col for col in results.columns if 'pred' in col.lower()]
            if pred_cols:
                predictions = results[pred_cols[0]].values
            else:
                predictions = results.iloc[:, -1].values
        else:
            predictions = np.array(results)
        
        # Ensure correct shape
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        
        logger.info(f"UniMol predictions completed. Shape: {predictions.shape}")
        return predictions
    
    def save(self, path: str):
        """Save model state"""
        if self._exp_path is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        model_state = {
            'exp_path': self._exp_path,
            'task': self.task,
            'model_name': self.model_name,
            'model_size': self.model_size,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'metrics': self.metrics,
            'remove_hs': self.remove_hs,
        }
        
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(model_state, f)
        
        logger.info(f"UniMol model state saved to {path}")
    
    def load(self, path: str):
        """Load model state"""
        import pickle
        with open(path, 'rb') as f:
            model_state = pickle.load(f)
        
        self._exp_path = model_state['exp_path']
        
        # Update parameters
        for key, value in model_state.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        logger.info(f"UniMol model state loaded from {path}")
    
    def make_new(self) -> "UniMolModel":
        """Create new instance with same parameters"""
        return self.__class__(**self.dict(exclude={"_estimator", "_exp_path"}))
    
    def get_model_summary(self):
        """Get model summary"""
        return {
            "model_type": f"UniMol {self.model_name}",
            "model_size": self.model_size,
            "task": self.task,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "metrics": self.metrics,
            "trained": self._exp_path is not None,
        }
