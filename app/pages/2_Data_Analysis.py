import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px 
from utils.data_processor import DataProcessor
from utils.visualizations import Visualizer
from config.settings import REQUIRED_COLUMNS

def app():
    st.title("📊 Data Analysis")
    
    # Check if data exists in session state
    if 'data' not in st.session_state:
        st.error("Please upload data in the Data Upload page first!")
        st.stop()
        
    # Initialize objects
    data_processor = DataProcessor()
    visualizer = Visualizer()
    df = st.session_state['data']
    
    # Sidebar for analysis options
    st.sidebar.subheader("Analysis Options")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Time Series Analysis", "Feature Relationships", "Outlier Detection"]
    )
    
    if analysis_type == "Time Series Analysis":
        st.subheader("Weight Progression Over Time")
        weight_plot = visualizer.plot_weight_over_time(df)
        st.plotly_chart(weight_plot, use_container_width=True)
        
        # Show growth rate
        st.subheader("Growth Rate Analysis")
        df['Growth Rate'] = df['Weight'].pct_change() * 100
        growth_plot = visualizer.plot_feature_distribution(df, 'Growth Rate')
        st.plotly_chart(growth_plot, use_container_width=True)
        
    elif analysis_type == "Feature Relationships":
        st.subheader("Feature Relationships")
        
        # Select features to compare
        col1, col2 = st.columns(2)
        with col1:
            feature_1 = st.selectbox("Select first feature", REQUIRED_COLUMNS)
        with col2:
            feature_2 = st.selectbox("Select second feature", 
                                   [col for col in REQUIRED_COLUMNS if col != feature_1])
        
        # Create scatter plot
        fig = px.scatter(df, x=feature_1, y=feature_2, 
                        
                        title=f"{feature_1} vs {feature_2}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show correlation coefficient
        correlation = df[feature_1].corr(df[feature_2])
        st.metric("Correlation Coefficient", f"{correlation:.3f}")
        
    else:  # Outlier Detection
        st.subheader("Outlier Detection")
        
        # Select feature for outlier detection
        feature = st.selectbox("Select feature to detect outliers", REQUIRED_COLUMNS)
        
        # Detect outliers
        outliers = data_processor.detect_outliers(df, feature)
        
        # Show statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of Outliers", outliers.sum())
        with col2:
            st.metric("Percentage of Outliers", f"{(outliers.sum()/len(df))*100:.2f}%")
        
        # Plot with outliers highlighted
        fig = px.scatter(
            df,
            x=df.index,
            y=feature,
            color=outliers,
            title=f"Outliers in {feature}",
            color_discrete_map={True: "red", False: "blue"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Download analyzed data
    st.sidebar.markdown("---")
    if st.sidebar.button("Download Analyzed Data"):
        csv = df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name="analyzed_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    app()