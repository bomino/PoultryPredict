# Poultry Weight Predictor 🐔

A sophisticated machine learning application built with Streamlit for predicting poultry weight based on environmental and feeding data. This tool helps poultry farmers and researchers make data-driven decisions using multiple machine learning models and comprehensive analysis tools.

## Features

### 1. Data Management
- **Data Upload**:
  - CSV file upload support
  - Automated data validation and preprocessing
  - Dynamic data type handling
  - Missing value detection and handling

- **Data Preprocessing**:
  - Automatic data cleaning
  - Feature scaling
  - Outlier detection
  - Data validation checks

### 2. Model Training
- **Multiple Models Support**:
  - Polynomial Regression
  - Gradient Boosting Regressor
  - Extensible architecture for adding more models

- **Model Configuration**:
  - Dynamic parameter settings
  - Model-specific parameter validation
  - Interactive parameter tuning
  - Performance metrics tracking

- **Training Features**:
  - Configurable train/test split
  - Cross-validation support
  - Feature importance analysis
  - Model performance visualization

### 3. Model Comparison
- **Comparison Metrics**:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R-squared (R²) Score
  - Feature importance comparison
  - Prediction accuracy analysis

- **Visualization Tools**:
  - Interactive performance plots
  - Feature importance charts
  - Prediction comparison graphs
  - Error analysis visualizations

### 4. Prediction Capabilities
- **Flexible Input Methods**:
  - Single prediction through UI
  - Batch predictions via CSV
  - Real-time prediction updates

- **Output Features**:
  - Detailed prediction analysis
  - Confidence metrics
  - Error estimates
  - Exportable results

## Installation

### Prerequisites
- Python 3.9+
- pip package manager
- Virtual environment (recommended)

### Setup

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

4. Run the application:
```bash
streamlit run app/main.py
```

## Project Structure
```
poultry_weight_predictor/
│
├── app/
│   ├── main.py                  # Application entry point
│   │
│   ├── pages/
│   │   ├── 1_Data_Upload.py       # Data upload and validation
│   │   ├── 2_Data_Analysis.py     # Data analysis and visualization
│   │   ├── 3_Model_Training.py    # Model training and evaluation
│   │   ├── 4_Predictions.py       # Prediction interface
│   │   └── 5_Model_Comparison.py  # Model comparison tools
│   │
│   ├── models/
│   │   ├── model_factory.py       # Model creation and management
│   │   ├── polynomial_regression.py# Polynomial regression model
│   │   └── gradient_boosting.py   # Gradient boosting model
│   │
│   ├── utils/
│   │   ├── data_processor.py      # Data processing utilities
│   │   ├── visualizations.py      # Visualization functions
│   │   └── model_comparison.py    # Model comparison utilities
│   │
│   └── config/
│       └── settings.py            # Application settings
│
├── models/                      # Saved models directory
├── data/                       # Sample data directory
├── tests/                      # Unit tests
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Usage Guide

### 1. Data Preparation
Required CSV format:
```csv
Int Temp,Int Humidity,Air Temp,Wind Speed,Feed Intake,Weight
29.87,59,29.6,4.3,11.00,42.39
32.50,47,30.5,4.3,12.47,45.67
```

### 2. Model Training
1. Upload data in the Data Upload page
2. Navigate to Model Training
3. Select model type
4. Configure parameters
5. Train model
6. View results and save model

### 3. Making Predictions
- **Single Prediction**:
  1. Enter values manually
  2. Get instant prediction

- **Batch Prediction**:
  1. Upload CSV file
  2. Get predictions for all rows
  3. Download results

### 4. Model Comparison
1. Train multiple models
2. Visit Model Comparison page
3. Compare performance metrics
4. Analyze feature importance
5. Export comparison report

## Configuration

### Model Parameters
Each model type has configurable parameters:

1. Polynomial Regression:
   - Degree
   - Feature selection

2. Gradient Boosting:
   - Number of estimators
   - Learning rate
   - Maximum depth
   - Minimum samples split

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Streamlit
- Uses scikit-learn for machine learning
- Plotly for visualizations
- Pandas for data handling

## Support

For support:
- Open an issue
- Contact [maintainer email]
- Check documentation

---

Made with ❤️ for poultry farmers and researchers