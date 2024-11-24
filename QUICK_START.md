# Quick Start Guide 🚀

## Setting Up Poultry Weight Predictor

### 1. First Time Setup
```bash
# Clone repository
git clone https://github.com/bomino/PoultryPredict.git

# Navigate to project
cd PoultryPredict

# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install requirements
pip install -r requirements.txt

# Start application
streamlit run app/main.py
```

### 2. Prepare Your Data
Your CSV file should include:
```csv
Int Temp,Int Humidity,Air Temp,Wind Speed,Feed Intake,Weight
29.87,59,29.6,4.3,11.00,42.39
```

Required columns:
- `Int Temp`: Internal Temperature (°C)
- `Int Humidity`: Internal Humidity (%)
- `Air Temp`: Air Temperature (°C)
- `Wind Speed`: Wind Speed (m/s)
- `Feed Intake`: Feed Intake (g)
- `Weight`: Weight (g) - for training only

### 3. Using the Application

#### a. Data Upload
1. Click "Browse files"
2. Select your CSV file
3. Review data preview
4. Check validation results

#### b. Model Training
1. Select model type:
   - Polynomial Regression
   - Gradient Boosting
2. Configure parameters
3. Set test split size
4. Click "Train Model"
5. Review results

#### c. Making Predictions
1. Choose input method:
   - Manual Input
   - Batch Prediction
2. Enter/upload data
3. Get predictions
4. Download results

#### d. Model Comparison
1. Train multiple models
2. Visit comparison page
3. Review metrics
4. Export report

### 4. Troubleshooting

Common issues and solutions:

1. **Data Upload Errors**
   - Check column names
   - Ensure numeric values
   - Remove special characters

2. **Training Errors**
   - Verify data quality
   - Check parameter values
   - Ensure sufficient data

3. **Prediction Errors**
   - Validate input ranges
   - Check data format
   - Verify model training

### 5. Best Practices

1. **Data Quality**
   - Clean your data
   - Remove outliers
   - Use consistent units

2. **Model Training**
   - Start with default parameters
   - Try multiple models
   - Compare performance

3. **Making Predictions**
   - Stay within training ranges
   - Validate unusual results
   - Keep prediction logs

### 6. Getting Help

If you need assistance:
1. Check error messages
2. Review documentation
3. Open an issue

---

For detailed information, see [README.md](README.md)