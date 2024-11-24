from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import numpy as np
from config.settings import POLYNOMIAL_DEGREE, MODEL_SAVE_PATH

class PoultryWeightPredictor:
    def __init__(self):
        """Initialize the model pipeline."""
        self.model = Pipeline([
            ('poly', PolynomialFeatures(degree=POLYNOMIAL_DEGREE)),
            ('regressor', LinearRegression())
        ])
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
            print("Training model with data shapes:", X_train.shape, y_train.shape)
            self.model.fit(X_train, y_train)
            self._is_trained = True
            print("Model trained successfully")
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
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            }
            
            return metrics, y_pred
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            raise
    
    def get_feature_importance(self, feature_names):
        """Get feature importance based on coefficient magnitudes."""
        if not self._is_trained:
            raise ValueError("Model needs to be trained before getting feature importance")
            
        try:
            # Get polynomial feature names
            poly = self.model.named_steps['poly']
            feature_names_poly = poly.get_feature_names_out(feature_names)
            
            # Get coefficients
            coefficients = self.model.named_steps['regressor'].coef_
            
            # Calculate absolute importance
            importance = np.abs(coefficients)
            
            # Create feature importance dictionary
            importance_dict = dict(zip(feature_names_poly, importance))
            
            # Sort by importance
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            print(f"Error getting feature importance: {str(e)}")
            raise
            
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