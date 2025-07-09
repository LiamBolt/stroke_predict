import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
import time
from datetime import datetime
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.abspath("."))

# Import utilities
from utils.data_processor import DataProcessor
from utils.model_handler import ModelHandler
from utils.visualizer import Visualizer
from utils.report_generator import ReportGenerator

def app():
    """Individual assessment page for the Stroke Risk Assessment Platform."""
    
    # Initialize session state for storing results
    if "individual_result" not in st.session_state:
        st.session_state.individual_result = None
    if "individual_input" not in st.session_state:
        st.session_state.individual_input = None
    if "individual_charts" not in st.session_state:
        st.session_state.individual_charts = None
    if "risk_info" not in st.session_state:
        st.session_state.risk_info = None
    if "report_generated" not in st.session_state:
        st.session_state.report_generated = False
    
    # Initialize utilities
    data_processor = DataProcessor()
    model_handler = ModelHandler()
    visualizer = Visualizer()
    report_generator = ReportGenerator()
    
    # Page title and introduction
    st.title("Stroke Risk Assessment for Individuals")
    
    st.markdown("""
    This tool helps you assess your personal risk of stroke based on your health information.
    Please fill in the form below with your current health data for an accurate assessment.
    
    <div style="background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
        <strong>🔒 Privacy Note:</strong> All data entered is processed locally in your browser and is not stored on any server.
    </div>
    """, unsafe_allow_html=True)
    
    # Create form for user input
    with st.form("individual_assessment_form"):
        st.subheader("Your Health Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age (years)", min_value=18, max_value=120, value=50)
            gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
            hypertension = st.radio("Do you have hypertension?", options=["No", "Yes"])
            heart_disease = st.radio("Do you have heart disease?", options=["No", "Yes"])
            ever_married = st.radio("Have you ever been married?", options=["No", "Yes"])
        
        with col2:
            avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", 
                                              min_value=50.0, max_value=300.0, value=100.0)
            bmi = st.number_input("BMI (Body Mass Index)", 
                                min_value=10.0, max_value=50.0, value=25.0)
            work_type = st.selectbox("Work Type", 
                                   options=["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
            residence_type = st.selectbox("Residence Type", options=["Urban", "Rural"])
            smoking_status = st.selectbox("Smoking Status", 
                                        options=["never smoked", "formerly smoked", "smokes", "Unknown"])
        
        # Submit button
        submit_button = st.form_submit_button("Analyze My Risk")
    
    # Process form submission
    if submit_button:
        with st.spinner("Analyzing your risk factors..."):
            # Prepare input data
            input_data = {
                'age': age,
                'gender': gender,
                'hypertension': 1 if hypertension == "Yes" else 0,
                'heart_disease': 1 if heart_disease == "Yes" else 0,
                'ever_married': "Yes" if ever_married == "Yes" else "No",
                'work_type': work_type,
                'Residence_type': residence_type,
                'avg_glucose_level': avg_glucose_level,
                'bmi': bmi,
                'smoking_status': smoking_status
            }
            
            # Validate input
            is_valid, error_message = data_processor.validate_individual_input(input_data)
            
            if not is_valid:
                st.error(f"Invalid input: {error_message}")
            else:
                try:
                    # Preprocess data
                    processed_data = data_processor.preprocess_individual(input_data)
                    
                    # Make prediction
                    _, probability = model_handler.predict(processed_data)
                    risk_probability = float(probability[0])
                    
                    # Determine risk category
                    risk_category, risk_color, risk_description = model_handler.get_risk_category(risk_probability)
                    
                    # Store results in session state
                    st.session_state.individual_result = risk_probability
                    st.session_state.individual_input = input_data
                    st.session_state.risk_info = (risk_category, risk_color, risk_description)
                    
                    # Create visualizations
                    risk_gauge = visualizer.create_individual_risk_gauge(risk_probability)
                    feature_comparison = visualizer.create_feature_comparison_plot(
                        input_data, data_processor.feature_ranges
                    )
                    
                    # Get feature importances
                    feature_importances = model_handler._get_feature_importances()
                    feature_importance_img = visualizer.create_feature_importance_plot(
                        feature_importances, "Key Factors in Stroke Risk"
                    )
                    
                    # Store charts in session state
                    st.session_state.individual_charts = {
                        "risk_gauge": risk_gauge,
                        "feature_comparison": feature_comparison,
                        "feature_importance": feature_importance_img
                    }
                    
                    # Reset report generation flag
                    st.session_state.report_generated = False
                    
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
    
    # Display results if available
    if st.session_state.individual_result is not None:
        st.markdown("---")
        st.subheader("Your Stroke Risk Assessment Results")
        
        # Get risk information
        risk_probability = st.session_state.individual_result
        risk_category, risk_color, risk_description = st.session_state.risk_info
        
        # Display risk gauge
        st.plotly_chart(st.session_state.individual_charts["risk_gauge"], use_container_width=True)
        
        # Display risk details
        st.markdown(f"""
        <div style="background-color: {risk_color}; padding: 20px; border-radius: 10px; color: white;">
            <h3>Risk Category: {risk_category}</h3>
            <p style="font-size: 18px;">{risk_description}</p>
            <p style="font-size: 16px;">Probability: {risk_probability:.2%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display feature comparison
        st.subheader("Your Risk Factors Compared to Normal Ranges")
        st.plotly_chart(st.session_state.individual_charts["feature_comparison"], use_container_width=True)
        
        # Display important risk factors
        st.subheader("Key Factors in Stroke Risk")
        st.image(st.session_state.individual_charts["feature_importance"])
        
        # Display personalized recommendations
        st.subheader("Personalized Recommendations")
        
        recommendations = []
        input_data = st.session_state.individual_input
        
        # Age recommendation
        if input_data["age"] > 65:
            recommendations.append("Age is a significant risk factor. Regular check-ups are especially important.")
        
        # Hypertension recommendation
        if input_data["hypertension"] == 1:
            recommendations.append("Manage your hypertension through medication, diet, and regular monitoring.")
        
        # Heart disease recommendation
        if input_data["heart_disease"] == 1:
            recommendations.append("Follow your cardiologist's advice and treatment plan for your heart condition.")
        
        # BMI recommendation
        if input_data["bmi"] > 25:
            recommendations.append("Consider a weight management program to achieve a healthier BMI.")
        
        # Glucose recommendation
        if input_data["avg_glucose_level"] > 140:
            recommendations.append("Your glucose levels are elevated. Consider diabetes screening and blood sugar management.")
        
        # Smoking recommendation
        if input_data["smoking_status"] == "smokes":
            recommendations.append("Quitting smoking is one of the most effective ways to reduce stroke risk.")
        
        # Add general recommendations
        recommendations.append("Maintain regular physical activity (at least 150 minutes of moderate activity per week).")
        recommendations.append("Adopt a heart-healthy diet rich in fruits, vegetables, and whole grains.")
        recommendations.append("Limit alcohol consumption.")
        recommendations.append("Know the warning signs of stroke (F.A.S.T.: Face drooping, Arm weakness, Speech difficulty, Time to call emergency services).")
        
        # Display recommendations
        for i, rec in enumerate(recommendations):
            st.markdown(f"**{i+1}.** {rec}")
        
        # Generate report section
        st.markdown("---")
        st.subheader("Download Your Assessment Report")
        
        if st.button("Generate PDF Report"):
            with st.spinner("Generating your personal report..."):
                try:
                    # Get feature descriptions
                    feature_descriptions = data_processor.get_feature_descriptions()
                    
                    # Generate PDF report
                    pdf_data = report_generator.generate_individual_report(
                        st.session_state.individual_input,
                        risk_probability,
                        st.session_state.risk_info,
                        feature_descriptions,
                        st.session_state.individual_charts
                    )
                    
                    # Create download button
                    b64_pdf = base64.b64encode(pdf_data).decode()
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    pdf_filename = f"stroke_risk_assessment_{current_date}.pdf"
                    
                    # Mark report as generated
                    st.session_state.report_generated = True
                    
                    # Display success message
                    st.success("Report generated successfully!")
                    
                    # Create download link
                    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{pdf_filename}">Download your PDF report</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
    
    # Add disclaimer at the bottom
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px;">
        <strong>Medical Disclaimer:</strong> This tool provides an assessment of stroke risk factors but is not a diagnostic device. 
        The results should be discussed with a qualified healthcare provider. Always seek the advice of your physician or other 
        qualified health provider with any questions you may have regarding a medical condition.
    </div>
    """, unsafe_allow_html=True)