# StrokeRisk Insight - Advanced Stroke Risk Assessment Platform

![StrokeRisk Insight](https://img.shields.io/badge/StrokeRisk-Insight-3498db)
![Python](https://img.shields.io/badge/Python-3.7+-2ecc71)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

StrokeRisk Insight is an advanced machine learning-powered platform that provides accurate stroke risk assessments for both individuals and medical experts. This application uses an XGBoost model trained on comprehensive health data to predict stroke risk based on key health indicators.

![Application Screenshot](assets/images/app_screenshot.png)

## Features

### For Individuals
- **Personalized Risk Assessment:** Evaluate your stroke risk based on personal health factors
- **Interactive Visualizations:** Understand how different factors contribute to your risk
- **Downloadable Reports:** Get a detailed PDF report of your assessment
- **AI Chatbot Support:** Ask questions about your results and stroke prevention

### For Medical Experts
- **Batch Analysis:** Upload patient datasets for efficient analysis
- **Comprehensive Metrics:** Access detailed model performance statistics
- **Advanced Visualizations:** Explore data distributions and model insights
- **Exportable Reports:** Generate clinical-grade PDF reports for patient records

## Model Details

The stroke prediction model used in this application is built with XGBoost, a powerful gradient boosting framework. Key risk factors assessed include:

- Age
- Average glucose level
- BMI (Body Mass Index)
- Hypertension status
- Heart disease history
- Smoking status
- Work type and lifestyle factors

## Installation and Setup

### Prerequisites
- Python 3.7+
- wkhtmltopdf (for PDF generation)

### Installation Steps

1. Clone this repository: