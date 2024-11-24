import streamlit as st

def app():
    st.title("ℹ️ About")
    
    st.markdown("""
    # Poultry Weight Predictor

    This application helps poultry farmers and researchers predict poultry weight based on environmental 
    and feeding data using machine learning techniques.

    ## Features

    1. **Data Upload and Analysis**
       - Upload CSV files with poultry data
       - Automatic data validation and preprocessing
       - Interactive data visualization
       - Basic statistical analysis

    2. **Advanced Analytics**
       - Time series analysis of weight progression
       - Feature relationship analysis
       - Outlier detection
       - Correlation analysis

    3. **Machine Learning Model**
       - Polynomial regression model
       - Feature importance analysis
       - Model performance metrics
       - Model saving and loading capabilities

    4. **Predictions**
       - Single prediction through manual input
       - Batch predictions through CSV upload
       - Prediction history tracking
       - Downloadable prediction results

    ## How to Use

    1. Start by uploading your data in the **Data Upload** page
    2. Analyze your data in the **Data Analysis** page
    3. Train a model in the **Model Training** page
    4. Make predictions in the **Predictions** page

    ## Data Requirements

    Your input data should contain the following features:
    - Internal Temperature (°C)
    - Internal Humidity (%)
    - Air Temperature (°C)
    - Wind Speed (m/s)
    - Feed Intake (g)
    - Weight (g) - for training data only

    ## Technical Details

    - Built with Streamlit
    - Uses Scikit-learn for machine learning
    - Plotly for interactive visualizations
    - Pandas for data manipulation

    ## Support

    For support or feature requests, please contact:
    - Email: mlawali@qidaya.com
    - GitHub: [Project Repository](https://github.com/bomino/poultry)
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit")

if __name__ == "__main__":
    app()