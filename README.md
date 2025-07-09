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

- git clone https://github.com/yourusername/strokerisk-insight.git 
- cd strokerisk-insight

2. Install required Python packages:


3. Install wkhtmltopdf (required for PDF generation):
- Windows: Download from https://wkhtmltopdf.org/downloads.html
- Mac: `brew install wkhtmltopdf`
- Linux: `sudo apt-get install wkhtmltopdf`

4. Set up your Gemini API key:
- Create a `.env` file in the root directory with:
  ```
  GEMINI_API_KEY=your_api_key_here
  ```
- Or set it directly in the environment:
  ```
  export GEMINI_API_KEY="your_api_key_here"
  ```

### Running the Application


Access the application in your browser at `http://localhost:8501`.

## Project Structure

- to be updated.


## Usage

### For Individuals
1. Select "Individual" user type from the sidebar
2. Navigate to "Individual Assessment" 
3. Enter your health information
4. Receive a personalized risk assessment and recommendations
5. Download your detailed risk report as a PDF

### For Medical Experts
1. Select "Medical Expert" user type from the sidebar
2. Navigate to "Medical Expert" interface
3. Upload a patient dataset (CSV format)
4. View analysis results and performance metrics
5. Generate a comprehensive report for clinical use

## Chatbot Support
- The integrated AI chatbot can answer questions about:
  - Stroke risk factors and prevention
  - Interpretation of assessment results
  - General information about stroke warning signs
  - Medical terminology and statistics

## Security and Privacy
This application processes health data locally in your browser. No data is stored on external servers, ensuring the privacy and security of sensitive medical information.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer
This tool provides an assessment of stroke risk factors but is not a diagnostic device. The results should be discussed with a qualified healthcare provider. Always seek the advice of your physician or other qualified health provider with any questions regarding medical conditions.

## Acknowledgements
- Streamlit for the interactive web framework
- XGBoost for the machine learning model implementation
- Google's Gemini API for powering the chatbot functionality