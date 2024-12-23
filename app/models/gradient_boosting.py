from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import numpy as np
import pandas as pd
from config.settings import MODEL_SAVE_PATH

class PoultryGBRegressor:
    def __init__(self, params=None):
        """Initialize the Gradient Boosting model with optional parameters."""
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'subsample': 1.0,
            'random_state': 42
        }
        self.params = params if params is not None else default_params
        self.model = GradientBoostingRegressor(**self.params)
        self._is_trained = False
        
    @property
    def is_trained(self):
        """Check if the model is trained."""
        return self._is_trained
    
    def train(self, X_train, y_train):
        """Train the model."""
        if X_train is None or y_train is None:
            raise ValueError("Training data cannot be None")
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Training data cannot be empty")
            
        try:
            print("Training Gradient Boosting model...")
            print(f"Training data shapes: X={X_train.shape}, y={y_train.shape}")
            self.model.fit(X_train, y_train)
            self._is_trained = True
            print("Model trained successfully")
            
            # Store feature importance
            self.feature_importances_ = self.model.feature_importances_
            
            return self
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
        
    def predict(self, X):
        """Make predictions using the trained model."""
        if not self._is_trained:
            raise ValueError("Model needs to be trained before making predictions")
        
        if X is None or len(X) == 0:
            raise ValueError("Input data cannot be empty")
            
        try:
            return self.model.predict(X)
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model performance."""
        if not self._is_trained:
            raise ValueError("Model needs to be trained before evaluation")
            
        try:
            # Make predictions
            y_pred = self.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate additional metrics
            mae = np.mean(np.abs(y_test - y_pred))
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'mape': mape
            }
            
            return metrics, y_pred
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            raise
    
    def get_feature_importance(self, feature_names):
        """Get feature importance based on the trained model."""
        if not self._is_trained:
            raise ValueError("Model needs to be trained before getting feature importance")
            
        try:
            # Create feature importance dictionary
            importance_dict = dict(zip(feature_names, self.feature_importances_))
            
            # Sort by importance
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            print(f"Error getting feature importance: {str(e)}")
            raise
    
    def get_model_params(self):
        """Get the current model parameters."""
        return self.params
    
    def set_model_params(self, params):
        """Update model parameters."""
        self.params = params
        self.model.set_params(**params)
        
    def save(self, filepath):
        """Save the model to a file."""
        if not self._is_trained:
            raise ValueError("Model needs to be trained before saving")
            
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            # Save the entire object
            joblib.dump(self, filepath)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            raise
            
    @classmethod
    def load(cls, filepath):
        """Load a model from a file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        try:
            # Load the model
            model = joblib.load(filepath)
            if not isinstance(model, cls):
                raise ValueError("Loaded file is not a valid model")
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def get_hyperparameter_grid(self):
        """Get hyperparameter grid for tuning."""
        return {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [2, 3, 4],
            'min_samples_split': [2, 4],
            'min_samples_leaf': [1, 2],
            'subsample': [0.8, 0.9, 1.0]
        }