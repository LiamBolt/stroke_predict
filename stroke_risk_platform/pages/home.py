import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image

def app():
    """Home page for the Stroke Risk Assessment Platform."""
    
    # Page title
    st.title("Welcome to StrokeRisk Insight")
    
    # Introduction
    st.markdown("""
    ## Advanced Stroke Risk Assessment Platform
    
    StrokeRisk Insight uses cutting-edge machine learning technology to provide accurate 
    stroke risk assessments. Our platform is designed for both individuals seeking to understand 
    their personal risk factors and medical experts who need to evaluate risk across patient populations.
    """)
    
    # Display logo or relevant image if available
    try:
        image = Image.open("assets/images/stroke_infographic.png")
        st.image(image, use_column_width=True)
    except:
        # Display information cards if image is not available
        st.markdown("""
        <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
            <div style="background-color: #3498db; color: white; padding: 20px; border-radius: 10px; margin: 10px; width: 45%;">
                <h3>Fast Facts About Stroke</h3>
                <ul>
                    <li>Stroke is the 2nd leading cause of death globally</li>
                    <li>Every 40 seconds, someone in the US has a stroke</li>
                    <li>Up to 80% of strokes are preventable</li>
                    <li>Time is critical - early treatment saves lives</li>
                </ul>
            </div>
            <div style="background-color: #2ecc71; color: white; padding: 20px; border-radius: 10px; margin: 10px; width: 45%;">
                <h3>Know the Signs (F.A.S.T.)</h3>
                <ul>
                    <li><b>F</b>ace Drooping</li>
                    <li><b>A</b>rm Weakness</li>
                    <li><b>S</b>peech Difficulty</li>
                    <li><b>T</b>ime to Call Emergency Services</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Platform features section
    st.markdown("## Platform Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### For Individuals
        
        - **Personalized Risk Assessment:** Evaluate your stroke risk based on personal health factors
        - **Interactive Visualizations:** Understand how different factors contribute to your risk
        - **Downloadable Reports:** Get a detailed PDF report of your assessment
        - **AI Chatbot Support:** Ask questions about your results and stroke prevention
        """)
        
        st.button("Go to Individual Assessment", on_click=lambda: st.session_state.update({"navigation": "Individual Assessment"}))
        
    with col2:
        st.markdown("""
        ### For Medical Experts
        
        - **Batch Analysis:** Upload patient datasets for efficient analysis
        - **Comprehensive Metrics:** Access detailed model performance statistics
        - **Advanced Visualizations:** Explore data distributions and model insights
        - **Exportable Reports:** Generate clinical-grade PDF reports for patient records
        """)
        
        st.button("Go to Medical Expert Interface", on_click=lambda: st.session_state.update({"navigation": "Medical Expert"}))
    
    # About the model
    st.markdown("## About Our Prediction Model")
    
    st.markdown("""
    Our stroke prediction model is built using XGBoost, a powerful machine learning algorithm known for its accuracy and interpretability.
    The model has been trained on comprehensive health data and validated using rigorous evaluation metrics.
    
    Key features the model considers include:
    - Age
    - Average glucose level
    - BMI (Body Mass Index)
    - Hypertension status
    - Heart disease history
    - Smoking status
    - Work type and lifestyle factors
    """)
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px;">
        <strong>Medical Disclaimer:</strong> This tool provides an assessment of stroke risk factors but is not a diagnostic device. 
        The results should be discussed with a qualified healthcare provider. Always seek the advice of your physician or other 
        qualified health provider with any questions you may have regarding a medical condition.
    </div>
    """, unsafe_allow_html=True)