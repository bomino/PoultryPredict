import streamlit as st
import pandas as pd
import os
import joblib
import traceback
import hashlib
from datetime import datetime
from utils.data_processor import DataProcessor, FEATURE_COLUMNS
from utils.visualizations import Visualizer
from models.model_factory import ModelFactory
from config.settings import MODEL_SAVE_PATH

VERSION = "1.0.0"

def generate_data_hash(df):
    """Generate a hash of the training data for versioning."""
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()

def validate_model_params(params, model_type, model_factory):
    """Validate model parameters against expected ranges."""
    default_params = model_factory.get_model_params(model_type)
    for param, value in params.items():
        if param not in default_params:
            return False, f"Unexpected parameter: {param}"
        if isinstance(value, (int, float)) and value <= 0:
            return False, f"Parameter {param} must be positive"
    return True, ""

def app():
    st.title("🎯 Model Training")
    
    # Check if data exists in session state
    if 'data' not in st.session_state:
        st.error("Please upload data in the Data Upload page first!")
        st.stop()
    
    # Initialize objects
    data_processor = DataProcessor()
    visualizer = Visualizer()
    model_factory = ModelFactory()
    
    # Get and process data
    df = st.session_state['data']
    
    try:
        df_processed = data_processor.preprocess_data(df)
        st.success(f"Data preprocessed successfully: {df_processed.shape[0]} rows")
        
        # Generate data hash for versioning
        data_hash = generate_data_hash(df_processed)
        
        # Save data_processor in session state
        st.session_state['data_processor'] = data_processor
        
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        st.stop()
    
    # Sidebar - Model Selection
    st.sidebar.subheader("Model Selection")
    available_models = model_factory.get_available_models()
    
    selected_model_type = st.sidebar.selectbox(
        "Select Model Type",
        list(available_models.keys()),
        format_func=lambda x: available_models[x]['name']
    )
    
    # Show model information
    with st.sidebar.expander("Model Information"):
        st.write(available_models[selected_model_type]['description'])
        st.write("**Strengths:**")
        for strength in available_models[selected_model_type]['strengths']:
            st.write(f"- {strength}")
        st.write("**Limitations:**")
        for limitation in available_models[selected_model_type]['limitations']:
            st.write(f"- {limitation}")
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    default_params = model_factory.get_model_params(selected_model_type)
    param_descriptions = model_factory.get_param_descriptions(selected_model_type)

    # Parameter input with validation
    model_params = {}
    for param, default_value in default_params.items():
        if param == 'random_state':
            model_params[param] = default_value  # Keep default random state
        elif isinstance(default_value, int):
            model_params[param] = st.sidebar.number_input(
                param,
                min_value=1,
                value=default_value,
                help=param_descriptions.get(param, f"Parameter: {param}")
            )
        elif isinstance(default_value, float):
            # Different ranges for different float parameters
            if param == 'learning_rate':
                max_val = 1.0
                step = 0.01
            elif param == 'subsample':
                max_val = 1.0
                step = 0.1
            else:
                max_val = 10.0
                step = 0.1
                
            model_params[param] = st.sidebar.slider(
                param,
                min_value=0.0,
                max_value=max_val,
                value=default_value,
                step=step,
                format="%.3f",
                help=param_descriptions.get(param, f"Parameter: {param}")
            )
        
    # Validate parameters
    is_valid_params, param_error = validate_model_params(model_params, selected_model_type, model_factory)
    if not is_valid_params:
        st.error(f"Invalid parameters: {param_error}")
        st.stop()
    
    # Training settings
    st.sidebar.subheader("Training Settings")
    min_test_size = max(0.1, 1 / len(df_processed))
    test_size = st.sidebar.slider(
        "Test Set Size", 
        min_value=min_test_size,
        max_value=0.4,
        value=0.2,
        step=0.05,
        help="Proportion of dataset to include in the test split"
    )
    
    # Data information
    with st.sidebar.expander("Data Information"):
        total_samples = len(df_processed)
        train_samples = int(total_samples * (1 - test_size))
        test_samples = total_samples - train_samples
        
        st.write("Data Split:")
        st.write(f"- Total samples: {total_samples}")
        st.write(f"- Training samples: {train_samples}")
        st.write(f"- Testing samples: {test_samples}")
    
    # Main content
    st.subheader("Data Split and Model Training")
    st.write("Features being used:", FEATURE_COLUMNS)
    
    # Prepare features
    try:
        X_train, X_test, y_train, y_test = data_processor.prepare_features(
            df_processed, 
            test_size=test_size
        )
        st.success("Features prepared successfully")
        st.write(f"Training set shape: {X_train.shape}")
        st.write(f"Test set shape: {X_test.shape}")
        
    except Exception as e:
        st.error(f"Error preparing features: {str(e)}")
        st.stop()
    
    # Save test data
    st.session_state['test_data'] = {
        'X_test': X_test,
        'y_test': y_test
    }
    
    # Training progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize session states
    if 'training_results' not in st.session_state:
        st.session_state['training_results'] = None
    if 'trained_models' not in st.session_state:
        st.session_state['trained_models'] = {}
    
    # Create model instance
    model = model_factory.get_model(selected_model_type, model_params)
    
    # Train model button
    if st.button("Train Model"):
        try:
            status_text.text("Training model...")
            progress_bar.progress(25)
            
            # Train and evaluate
            model.train(X_train, y_train)
            progress_bar.progress(50)
            
            metrics, y_pred = model.evaluate(X_test, y_test)
            progress_bar.progress(75)
            
            importance_dict = model.get_feature_importance(FEATURE_COLUMNS)
            progress_bar.progress(100)
            
            # Create metadata
            metadata = {
                'version': VERSION,
                'training_date': datetime.now().isoformat(),
                'data_hash': data_hash,
                'model_type': selected_model_type,
                'model_params': model_params,
                'feature_columns': FEATURE_COLUMNS,
                'test_size': test_size,
                'training_samples': train_samples,
                'test_samples': test_samples
            }
            
            # Store results
            st.session_state['model'] = model
            st.session_state['model_metadata'] = metadata
            st.session_state['training_results'] = {
                'metrics': metrics,
                'predictions': y_pred,
                'feature_importance': importance_dict,
                'test_size': test_size
            }
            
            # Store for comparison
            model_name = f"{available_models[selected_model_type]['name']}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            st.session_state['trained_models'][model_name] = {
                'model': model,
                'metadata': metadata,
                'metrics': metrics,
                'predictions': y_pred,
                'actual': y_test,
                'feature_importance': importance_dict,
                'parameters': model_params
            }
            
            status_text.text("Training completed!")
            
        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            st.code(traceback.format_exc())
            progress_bar.empty()
            status_text.empty()
    
    # Display results if available
    if st.session_state.get('training_results'):
        results = st.session_state['training_results']
        metrics = results['metrics']
        y_pred = results['predictions']
        importance_dict = results['feature_importance']
        
        # Create tabs for different sections
        tabs = st.tabs(["Model Performance", "Predictions", "Feature Importance", "Model Details"])
        
        # Model Performance tab
        with tabs[0]:
            st.subheader("Model Performance Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Squared Error", f"{metrics['mse']:.2f}")
            with col2:
                st.metric("Root MSE", f"{metrics['rmse']:.2f}")
            with col3:
                st.metric("R² Score", f"{metrics['r2']:.4f}")
        
        # Predictions tab
        with tabs[1]:
            st.subheader("Actual vs Predicted Values")
            prediction_plot = visualizer.plot_actual_vs_predicted(
                st.session_state['test_data']['y_test'],
                y_pred
            )
            st.plotly_chart(prediction_plot, use_container_width=True)
            
            if st.checkbox("Show detailed predictions"):
                n_examples = min(10, len(y_pred))
                examples = pd.DataFrame({
                    'Actual Weight': st.session_state['test_data']['y_test'][:n_examples],
                    'Predicted Weight': y_pred[:n_examples],
                    'Absolute Error': abs(
                        st.session_state['test_data']['y_test'][:n_examples] - 
                        y_pred[:n_examples]
                    ),
                    'Relative Error (%)': abs(
                        st.session_state['test_data']['y_test'][:n_examples] - 
                        y_pred[:n_examples]
                    ) / st.session_state['test_data']['y_test'][:n_examples] * 100
                })
                st.dataframe(examples)
        
        # Feature Importance tab
        with tabs[2]:
            st.subheader("Feature Importance Analysis")
            importance_plot = visualizer.plot_feature_importance(
                list(importance_dict.keys()),
                list(importance_dict.values())
            )
            st.plotly_chart(importance_plot, use_container_width=True)
            
            with st.expander("Feature Importance Details"):
                importance_df = pd.DataFrame({
                    'Feature': importance_dict.keys(),
                    'Importance': importance_dict.values()
                })
                st.dataframe(importance_df)
        
        # Model Details tab
        with tabs[3]:
            if 'model_metadata' in st.session_state:
                st.subheader("Model Metadata")
                metadata = st.session_state['model_metadata']
                for key, value in metadata.items():
                    st.write(f"**{key}:** {value}")
        
        # Model saving section
        st.subheader("Save Model")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            model_name = st.text_input(
                "Model Name", 
                value=f"{selected_model_type}_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
                help="Enter a name for the model or use the default timestamp-based name"
            )
        
        with col2:
            if st.button("Save Model"):
                try:
                    # Validation checks
                    if 'model' not in st.session_state:
                        st.error("No model found! Please train a model first.")
                        st.stop()
                    
                    trained_model = st.session_state['model']
                    if not hasattr(trained_model, 'is_trained') or not trained_model.is_trained:
                        st.error("Model needs to be trained before saving")
                        st.stop()
                    
                    if not model_name:
                        st.error("Please enter a model name")
                        st.stop()
                    
                    # Prepare save path
                    if not model_name.endswith('.joblib'):
                        model_name += '.joblib'
                    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
                    full_path = os.path.join(MODEL_SAVE_PATH, model_name)
                    
                    # Create save dictionary with metadata
                    save_dict = {
                        'model': trained_model,
                        'data_processor': st.session_state.get('data_processor'),
                        'metadata': st.session_state.get('model_metadata'),
                        'training_results': st.session_state.get('training_results'),
                        'feature_columns': FEATURE_COLUMNS
                    }
                    
                    # Save model
                    joblib.dump(save_dict, full_path)
                    
                    # Success messages
                    st.success("Model saved successfully!")
                    st.info(f"Save location: {full_path}")
                    
                    # Show saved contents
                    with st.expander("Saved Model Contents"):
                        for key in save_dict.keys():
                            st.write(f"- {key}")
                    
                except Exception as e:
                    st.error(f"Error saving model: {str(e)}")
                    st.code(traceback.format_exc())
        
        # Model comparison suggestion
        if len(st.session_state.get('trained_models', {})) > 1:
            st.info("💡 You have trained multiple models. Visit the Model Comparison page to compare their performance!")

if __name__ == "__main__":
    app()