# Poultry Weight Predictor Documentation

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Data Requirements](#data-requirements)
4. [Features and Functionality](#features-and-functionality)
5. [Technical Architecture](#technical-architecture)
6. [Machine Learning Models](#machine-learning-models)
7. [User Guide](#user-guide)
8. [Troubleshooting](#troubleshooting)
9. [Development Guide](#development-guide)
10. [API Reference](#api-reference)

## Overview

The Poultry Weight Predictor is a machine learning application built with Streamlit that helps poultry farmers and researchers predict poultry weight based on environmental and feeding data. The application provides comprehensive data analysis, model training, and prediction capabilities.

### Key Features
- Data upload and validation
- Interactive data analysis and visualization
- Multiple machine learning model support
- Model training and evaluation
- Real-time predictions
- Model comparison and performance analysis
- Export capabilities for reports and predictions

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/bomino/PoultryPredict.git
cd PoultryPredict
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Required Dependencies
```text
streamlit==1.32.0
pandas==2.2.0
numpy==1.26.4
scikit-learn==1.4.0
plotly==5.18.0
pytest==8.0.0
python-dotenv==1.0.0
joblib==1.3.2
xlsxwriter==3.1.9
```

## Data Requirements

### Input Data Format
The application expects CSV files with the following columns:

| Column Name    | Description                | Unit  | Type    |
|---------------|---------------------------|-------|---------|
| Int Temp      | Internal Temperature      | °C    | float   |
| Int Humidity  | Internal Humidity         | %     | float   |
| Air Temp      | Air Temperature           | °C    | float   |
| Wind Speed    | Wind Speed               | m/s   | float   |
| Feed Intake   | Feed Intake              | g     | float   |
| Weight        | Poultry Weight (target)  | g     | float   |

### Data Quality Requirements
- No missing values
- Numerical values only
- Consistent units
- Reasonable value ranges for each parameter

## Features and Functionality

### 1. Data Upload (Page 1)
- CSV file upload capability
- Automatic data validation
- Data preview and summary statistics
- Data quality checks
- Sample template download

### 2. Data Analysis (Page 2)
- Time series analysis
- Feature relationship exploration
- Correlation analysis
- Outlier detection
- Interactive visualizations

### 3. Model Training (Page 3)
- Multiple model types support:
  - Polynomial Regression
  - Gradient Boosting
- Hyperparameter tuning
- Cross-validation
- Model performance metrics
- Feature importance analysis
- Model saving capabilities

### 4. Predictions (Page 4)
- Single prediction through manual input
- Batch predictions via CSV upload
- Prediction history tracking
- Confidence intervals
- Export capabilities

### 5. Model Comparison (Page 5)
- Side-by-side model comparison
- Performance metrics comparison
- Prediction accuracy comparison
- Feature importance comparison
- Export comparison reports

## Technical Architecture

### Project Structure
```
poultry_weight_predictor/
├── app/
│   ├── main.py                 # Main application entry
│   ├── config/
│   │   └── settings.py         # Application settings
│   ├── models/
│   │   ├── model_factory.py    # Model factory class
│   │   ├── polynomial_regression.py
│   │   └── gradient_boosting.py
│   ├── pages/
│   │   ├── 1_Data_Upload.py
│   │   ├── 2_Data_Analysis.py
│   │   ├── 3_Model_Training.py
│   │   ├── 4_Predictions.py
│   │   └── 5_Model_Comparison.py
│   └── utils/
│       ├── data_processor.py   # Data processing utilities
│       ├── visualizations.py   # Visualization functions
│       └── model_comparison.py # Model comparison utilities
├── models/                    # Saved models directory
├── tests/                    # Unit tests
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

### Key Components

#### DataProcessor Class
Handles all data processing operations:
- Data validation
- Preprocessing
- Feature engineering
- Data splitting
- Scaling

#### ModelFactory Class
Manages model creation and configuration:
- Model instantiation
- Parameter management
- Model registration
- Default configurations

#### Visualization Class
Handles all data visualization:
- Interactive plots
- Performance metrics visualization
- Feature importance plots
- Prediction comparison plots

#### ModelComparison Class
Manages model comparison functionality:
- Metrics comparison
- Prediction comparison
- Feature importance comparison
- Report generation

## Machine Learning Models

### 1. Polynomial Regression
- **Description**: Non-linear regression using polynomial features
- **Parameters**:
  - degree: Polynomial degree (default: 2)
- **Strengths**:
  - Good for capturing non-linear patterns
  - Simple and interpretable
  - Works well with small datasets
- **Limitations**:
  - May overfit with high polynomial degrees
  - Sensitive to outliers

### 2. Gradient Boosting
- **Description**: Ensemble learning using gradient boosting
- **Parameters**:
  - n_estimators: Number of boosting stages
  - learning_rate: Learning rate
  - max_depth: Maximum tree depth
  - min_samples_split: Minimum samples for split
  - min_samples_leaf: Minimum samples in leaf
  - subsample: Subsample ratio
- **Strengths**:
  - Handles non-linear relationships well
  - Robust to outliers
  - High accuracy
- **Limitations**:
  - More computationally intensive
  - Requires more tuning

## User Guide

### Getting Started
1. **Data Preparation**
   - Prepare CSV file with required columns
   - Ensure data quality requirements are met
   - Download and use template if needed

2. **Data Upload**
   - Navigate to "Data Upload" page
   - Upload CSV file
   - Review data summary and quality checks
   - Address any validation errors

3. **Data Analysis**
   - Explore data distributions
   - Check feature relationships
   - Identify outliers
   - Review correlation analysis

4. **Model Training**
   - Select model type
   - Configure parameters
   - Set training/test split
   - Train model
   - Review performance metrics

5. **Making Predictions**
   - Choose prediction method (single/batch)
   - Input or upload data
   - Get predictions
   - Export results

6. **Model Comparison**
   - Train multiple models
   - Compare performance
   - Select best model
   - Export comparison report

### Best Practices
1. **Data Quality**
   - Clean data before upload
   - Remove obvious outliers
   - Ensure consistent units
   - Validate data ranges

2. **Model Selection**
   - Start with simple models
   - Compare multiple models
   - Consider computational resources
   - Balance accuracy vs. complexity

3. **Parameter Tuning**
   - Start with default parameters
   - Use cross-validation
   - Monitor for overfitting
   - Document optimal parameters

## Troubleshooting

### Common Issues and Solutions

1. **Data Upload Issues**
- **Problem**: File format errors
  - **Solution**: Ensure CSV format and column names match requirements
- **Problem**: Missing values
  - **Solution**: Clean data before upload, fill or remove missing values

2. **Model Training Issues**
- **Problem**: Convergence warnings
  - **Solution**: Adjust learning rate or iterations
- **Problem**: Overfitting
  - **Solution**: Reduce model complexity, increase training data

3. **Prediction Issues**
- **Problem**: Unreasonable predictions
  - **Solution**: Check input data ranges, validate model
- **Problem**: Slow batch predictions
  - **Solution**: Reduce batch size, optimize parameters

### Error Messages
- Detailed explanation of common error messages
- Step-by-step resolution guides
- Prevention tips

## Development Guide

### Adding New Models
1. Create model class in models directory
2. Implement required methods:
   - train()
   - predict()
   - evaluate()
   - get_feature_importance()
3. Add to ModelFactory
4. Update model documentation

### Adding Features
1. Follow existing code structure
2. Maintain consistent style
3. Add appropriate error handling
4. Update documentation
5. Add unit tests

### Testing
- Run unit tests: `pytest tests/`
- Test data validation
- Test model performance
- Test UI functionality

## API Reference

### DataProcessor Class
```python
class DataProcessor:
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame
    def prepare_features(self, df: pd.DataFrame, test_size: float) -> tuple
    def validate_columns(self, df: pd.DataFrame) -> tuple[bool, list]
    def scale_features(self, X: pd.DataFrame) -> np.ndarray
```

### ModelFactory Class
```python
class ModelFactory:
    def get_model(self, model_type: str, params: dict = None)
    def get_available_models(self) -> dict
    def get_model_params(self, model_type: str) -> dict
    def get_param_descriptions(self, model_type: str) -> dict
```

### ModelComparison Class
```python
class ModelComparison:
    def add_model_results(self, model_name: str, metrics: dict, predictions: np.ndarray)
    def get_metrics_comparison(self) -> pd.DataFrame
    def get_prediction_comparison(self) -> pd.DataFrame
    def plot_metrics_comparison(self, metric: str = 'r2')
    def export_comparison_report(self) -> dict
```

For detailed API documentation, refer to the docstrings in each module.