from .polynomial_regression import PoultryWeightPredictor
from .gradient_boosting import PoultryGBRegressor

class ModelFactory:
    @staticmethod
    def get_model(model_type: str, params=None):
        """
        Factory method to create model instances.
        
        Args:
            model_type (str): Type of model ('polynomial' or 'gradient_boosting')
            params (dict): Optional model parameters
            
        Returns:
            Model instance
        """
        if model_type.lower() == 'polynomial':
            return PoultryWeightPredictor()
        elif model_type.lower() == 'gradient_boosting':
            return PoultryGBRegressor(params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_available_models():
        """Get list of available model types with descriptions."""
        return {
            'polynomial': {
                'name': 'Polynomial Regression',
                'description': 'A polynomial regression model that captures non-linear relationships between features.',
                'strengths': [
                    'Good for capturing non-linear patterns',
                    'Simple and interpretable',
                    'Works well with small datasets'
                ],
                'limitations': [
                    'May overfit with high polynomial degrees',
                    'Sensitive to outliers',
                    'Less effective for complex patterns'
                ]
            },
            'gradient_boosting': {
                'name': 'Gradient Boosting',
                'description': 'An ensemble learning model that builds strong predictors by combining weak learners.',
                'strengths': [
                    'Handles non-linear relationships well',
                    'Robust to outliers',
                    'Provides feature importance',
                    'Generally high accuracy'
                ],
                'limitations': [
                    'More computationally intensive',
                    'Requires more hyperparameter tuning',
                    'Can overfit if not properly configured'
                ]
            }
        }
    
    @staticmethod
    def get_model_params(model_type: str):
        """Get default parameters for a model type."""
        if model_type.lower() == 'polynomial':
            return {
                'degree': 2
            }
        elif model_type.lower() == 'gradient_boosting':
            return {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'subsample': 1.0,
                'random_state': 42
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_param_descriptions(model_type: str):
        """Get descriptions for model parameters."""
        if model_type.lower() == 'polynomial':
            return {
                'degree': 'The degree of the polynomial features (higher values capture more complex patterns)'
            }
        elif model_type.lower() == 'gradient_boosting':
            return {
                'n_estimators': 'Number of boosting stages to perform',
                'learning_rate': 'Step size shrinkage to prevent overfitting',
                'max_depth': 'Maximum depth of individual regression estimators',
                'min_samples_split': 'Minimum samples required to split an internal node',
                'min_samples_leaf': 'Minimum samples required to be at a leaf node',
                'subsample': 'Fraction of samples to be used for fitting individual base learners',
                'random_state': 'Random number seed for reproducibility'
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")