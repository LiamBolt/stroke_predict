import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
import time
from datetime import datetime
import os
import sys
import plotly.express as px

# Add the project root to Python path
sys.path.insert(0, os.path.abspath("."))

# Import utilities
from utils.data_processor import DataProcessor
from utils.model_handler import ModelHandler
from utils.visualizer import Visualizer
from utils.report_generator import ReportGenerator

def app():
    """Medical expert page for the Stroke Risk Assessment Platform."""
    
    # Initialize session state for storing results
    if "dataset" not in st.session_state:
        st.session_state.dataset = None
    if "dataset_predictions" not in st.session_state:
        st.session_state.dataset_predictions = None
    if "evaluation_metrics" not in st.session_state:
        st.session_state.evaluation_metrics = None
    if "dataset_charts" not in st.session_state:
        st.session_state.dataset_charts = None
    
    # Initialize utilities
    data_processor = DataProcessor()
    model_handler = ModelHandler()
    visualizer = Visualizer()
    report_generator = ReportGenerator()
    
    # Page title and introduction
    st.title("Stroke Risk Assessment for Medical Experts")
    
    st.markdown("""
    This professional interface allows medical experts to analyze stroke risk across patient populations.
    Upload a dataset of patient information to receive detailed risk assessments and statistical analysis.
    
    <div style="background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
        <strong>📊 Data Requirements:</strong> Your dataset must include the following columns: 
        'age', 'gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 
        'avg_glucose_level', 'bmi', 'smoking_status'. Optional: 'stroke' (actual outcomes for evaluation).
    </div>
    """, unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader("Upload Patient Dataset (CSV format)", type=["csv"])
    
    # Sample dataset option
    st.markdown("**Don't have a dataset?** Use our sample dataset:")
    if st.button("Load Sample Dataset"):
        # Create a sample dataset
        sample_data = {
            'age': np.random.normal(65, 15, 100).clip(18, 100),
            'gender': np.random.choice(['Male', 'Female'], 100),
            'hypertension': np.random.choice([0, 1], 100, p=[0.7, 0.3]),
            'heart_disease': np.random.choice([0, 1], 100, p=[0.8, 0.2]),
            'ever_married': np.random.choice(['Yes', 'No'], 100),
            'work_type': np.random.choice(['Private', 'Self-employed', 'Govt_job'], 100),
            'Residence_type': np.random.choice(['Urban', 'Rural'], 100),
            'avg_glucose_level': np.random.normal(110, 40, 100).clip(50, 300),
            'bmi': np.random.normal(28, 7, 100).clip(15, 50),
            'smoking_status': np.random.choice(['never smoked', 'formerly smoked', 'smokes', 'Unknown'], 100),
            'stroke': np.random.choice([0, 1], 100, p=[0.95, 0.05])
        }
        df = pd.DataFrame(sample_data)
        st.session_state.dataset = df
        st.success("Sample dataset loaded successfully!")
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            # Read dataset
            df = pd.read_csv(uploaded_file)
            
            # Display dataset preview
            st.subheader("Dataset Preview")
            st.dataframe(df.head())
            
            # Store in session state
            st.session_state.dataset = df
            
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
    
    # Analyze dataset button
    if st.session_state.dataset is not None:
        if st.button("Analyze Dataset"):
            with st.spinner("Analyzing patient data..."):
                try:
                    # Get dataset
                    df = st.session_state.dataset.copy()
                    
                    # Check required columns
                    required_cols = ['age', 'gender', 'hypertension', 'heart_disease', 
                                    'ever_married', 'work_type', 'Residence_type', 
                                    'avg_glucose_level', 'bmi', 'smoking_status']
                    
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    else:
                        # Preprocess data
                        processed_data, df_with_nans_handled = data_processor.preprocess_dataset(df)
                        
                        # Make predictions
                        predictions, probabilities = model_handler.predict(processed_data)
                        
                        # Add predictions to dataframe
                        df_with_predictions = df_with_nans_handled.copy()
                        df_with_predictions['predicted_stroke'] = predictions
                        df_with_predictions['stroke_probability'] = probabilities
                        
                        # Calculate risk categories
                        risk_categories = []
                        for prob in probabilities:
                            category, _, _ = model_handler.get_risk_category(prob)
                            risk_categories.append(category)
                        
                        df_with_predictions['risk_category'] = risk_categories
                        
                        # Store results
                        st.session_state.dataset_predictions = df_with_predictions
                        
                        # If dataset contains actual stroke labels, evaluate model
                        if 'stroke' in df.columns:
                            evaluation_metrics = model_handler.evaluate_dataset(
                                df['stroke'].values, probabilities
                            )
                            st.session_state.evaluation_metrics = evaluation_metrics
                        
                        # Create charts
                        feature_importances = model_handler._get_feature_importances()
                        feature_importance_img = visualizer.create_feature_importance_plot(
                            feature_importances, "Key Risk Factors"
                        )
                        
                        # Create dataset distribution visualizations
                        dist_plots = visualizer.create_dataset_distribution_plots(df_with_predictions)
                        
                        # Create confusion matrix if actual labels exist
                        confusion_matrix_img = None
                        roc_curve_img = None
                        calibration_plot_img = None
                        
                        if 'stroke' in df.columns and 'confusion_matrix' in evaluation_metrics:
                            confusion_matrix_img = visualizer.create_confusion_matrix_plot(
                                evaluation_metrics['confusion_matrix']
                            )
                            roc_curve_img = visualizer.create_roc_curve_plot(
                                evaluation_metrics['fpr'],
                                evaluation_metrics['tpr'],
                                evaluation_metrics['roc_auc']
                            )
                            calibration_plot_img = visualizer.create_calibration_plot(
                                evaluation_metrics['prob_true'],
                                evaluation_metrics['prob_pred']
                            )
                        
                        # Store charts
                        st.session_state.dataset_charts = {
                            'feature_importance': feature_importance_img,
                            'distribution_plots': dist_plots,
                            'confusion_matrix': confusion_matrix_img,
                            'roc_curve': roc_curve_img,
                            'calibration_plot': calibration_plot_img
                        }
                        
                except Exception as e:
                    st.error(f"Error analyzing dataset: {str(e)}")
    
    # Display results if available
    if st.session_state.dataset_predictions is not None:
        st.markdown("---")
        st.subheader("Analysis Results")
        
        # Get results dataframe
        df_results = st.session_state.dataset_predictions
        
        # Display summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Patients", len(df_results))
        
        with col2:
            high_risk_count = len(df_results[df_results['stroke_probability'] >= 0.5])
            st.metric("High Risk Patients", high_risk_count)
        
        with col3:
            if 'stroke' in df_results.columns:
                accuracy = (df_results['predicted_stroke'] == df_results['stroke']).mean()
                st.metric("Model Accuracy", f"{accuracy:.2%}")
            else:
                avg_prob = df_results['stroke_probability'].mean()
                st.metric("Average Risk Score", f"{avg_prob:.2%}")
        
        # Display results table with filters
        st.subheader("Patient Risk Assessment")
        
        # Add filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_filter = st.multiselect(
                "Filter by Risk Category",
                options=sorted(df_results['risk_category'].unique()),
                default=[]
            )
        
        with col2:
            min_age, max_age = int(df_results['age'].min()), int(df_results['age'].max())
            age_range = st.slider(
                "Age Range",
                min_value=min_age,
                max_value=max_age,
                value=(min_age, max_age)
            )
        
        with col3:
            gender_filter = st.multiselect(
                "Filter by Gender",
                options=sorted(df_results['gender'].unique()),
                default=[]
            )
        
        # Apply filters
        filtered_df = df_results.copy()
        
        if risk_filter:
            filtered_df = filtered_df[filtered_df['risk_category'].isin(risk_filter)]
        
        filtered_df = filtered_df[(filtered_df['age'] >= age_range[0]) & (filtered_df['age'] <= age_range[1])]
        
        if gender_filter:
            filtered_df = filtered_df[filtered_df['gender'].isin(gender_filter)]
        
        # Display filtered results
        st.dataframe(filtered_df.style.background_gradient(
            subset=['stroke_probability'], 
            cmap='RdYlGn_r',
            vmin=0,
            vmax=1
        ))
        
        # Download CSV button
        csv = filtered_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="stroke_risk_predictions.csv">Download Results CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # Display visualizations
        st.markdown("---")
        st.subheader("Data Visualizations")
        
        charts = st.session_state.dataset_charts
        
        # Feature importance
        st.markdown("### Key Risk Factors")
        st.image(charts['feature_importance'])
        
        # Distribution plots
        st.markdown("### Data Distributions")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Age Distribution", 
            "Glucose Level Distribution",
            "BMI Distribution",
            "Gender Distribution"
        ])
        
        with tab1:
            st.plotly_chart(charts['distribution_plots']['age_dist'], use_container_width=True)
        
        with tab2:
            st.plotly_chart(charts['distribution_plots']['glucose_dist'], use_container_width=True)
        
        with tab3:
            st.plotly_chart(charts['distribution_plots']['bmi_dist'], use_container_width=True)
        
        with tab4:
            st.plotly_chart(charts['distribution_plots']['gender_dist'], use_container_width=True)
        
        # Model evaluation if available
        if st.session_state.evaluation_metrics is not None:
            st.markdown("---")
            st.subheader("Model Performance")
            
            # Display confusion matrix
            if charts['confusion_matrix'] is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Confusion Matrix")
                    st.markdown("""
                    The confusion matrix shows how well our model classifies stroke vs. non-stroke cases:
                    - **True Negatives (top-left)**: Correctly identified non-stroke cases
                    - **False Positives (top-right)**: Incorrectly flagged as stroke
                    - **False Negatives (bottom-left)**: Missed stroke cases
                    - **True Positives (bottom-right)**: Correctly identified stroke cases
                    """)
                    st.image(charts['confusion_matrix'])
                
                with col2:
                    st.markdown("### ROC Curve")
                    st.markdown("""
                    The ROC curve shows the trade-off between:
                    - **True Positive Rate (Sensitivity)**: Percentage of actual stroke cases correctly identified
                    - **False Positive Rate**: Percentage of non-stroke cases incorrectly flagged
                    
                    The Area Under the Curve (AUC) measures overall performance - higher is better.
                    """)
                    st.image(charts['roc_curve'])
            
            # Display calibration plot
            if charts['calibration_plot'] is not None:
                st.markdown("### Calibration Plot")
                st.markdown("""
                The calibration plot shows how well the model's predicted probabilities align with actual outcomes.
                Points on the diagonal line indicate perfect calibration.
                """)
                st.image(charts['calibration_plot'])
            
            # Display classification metrics
            st.markdown("### Classification Metrics")
            
            metrics = st.session_state.evaluation_metrics['classification_report']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            
            with col2:
                st.metric("Precision", f"{metrics['1']['precision']:.3f}")
            
            with col3:
                st.metric("Recall", f"{metrics['1']['recall']:.3f}")
            
            with col4:
                st.metric("F1 Score", f"{metrics['1']['f1-score']:.3f}")
            
            st.markdown("""
            **Interpreting these metrics:**
            
            - **Accuracy**: Overall correctness (both stroke and non-stroke cases)
            - **Precision**: Out of all predicted stroke cases, how many were actually strokes
            - **Recall (Sensitivity)**: Out of all actual stroke cases, how many were correctly identified
            - **F1 Score**: Harmonic mean of precision and recall, balancing both concerns
            """)
        
        # Generate report section
        st.markdown("---")
        st.subheader("Generate Comprehensive Report")
        
        if st.button("Generate PDF Report"):
            with st.spinner("Generating comprehensive analysis report..."):
                try:
                    # Get feature descriptions
                    feature_descriptions = data_processor.get_feature_descriptions()
                    
                    # Generate PDF report
                    pdf_data = report_generator.generate_dataset_report(
                        st.session_state.dataset_predictions,
                        st.session_state.evaluation_metrics if st.session_state.evaluation_metrics else {},
                        st.session_state.dataset_charts,
                        feature_descriptions
                    )
                    
                    # Create download button
                    b64_pdf = base64.b64encode(pdf_data).decode()
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    pdf_filename = f"stroke_risk_analysis_{current_date}.pdf"
                    
                    # Display success message
                    st.success("Report generated successfully!")
                    
                    # Create download link
                    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{pdf_filename}">Download PDF Report</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
    
    # Add professional disclaimer at the bottom
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px;">
        <strong>Professional Use Disclaimer:</strong> This tool is designed to supplement, not replace, clinical judgment. 
        Results should be interpreted in the context of each individual patient's clinical presentation and history. 
        The model provides risk stratification based on available data but has inherent limitations.
    </div>
    """, unsafe_allow_html=True)