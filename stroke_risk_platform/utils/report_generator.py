import pdfkit
import pandas as pd
import jinja2
import os
import base64
from datetime import datetime
import plotly.io as pio
from io import BytesIO
import matplotlib.pyplot as plt
from tempfile import NamedTemporaryFile

class ReportGenerator:
    """
    Class for generating PDF reports for stroke risk assessment results.
    """
    
    def __init__(self):
        """Initialize the ReportGenerator with template settings."""
        # Setup Jinja2 environment
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader('assets/templates'),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Config options for pdfkit (wkhtmltopdf)
        self.pdf_options = {
            'page-size': 'A4',
            'margin-top': '15mm',
            'margin-right': '15mm',
            'margin-bottom': '15mm',
            'margin-left': '15mm',
            'encoding': 'UTF-8',
            'no-outline': None,
            'enable-local-file-access': None
        }
    
    def generate_individual_report(self, input_data, prediction_result, risk_info, 
                                   feature_descriptions, charts):
        """
        Generate a PDF report for an individual's stroke risk assessment.
        
        Args:
            input_data (dict): Dictionary with individual's input data
            prediction_result (float): Predicted probability of stroke
            risk_info (tuple): (risk_category, color, description)
            feature_descriptions (dict): Dictionary with feature descriptions
            charts (dict): Dictionary with encoded chart images
            
        Returns:
            bytes: PDF report as bytes
        """
        try:
            # Load template
            template = self.env.get_template('individual_report_template.html')
            
            # Prepare data for template
            risk_category, risk_color, risk_description = risk_info
            
            # Format timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Render template
            html_content = template.render(
                patient_data=input_data,
                prediction=prediction_result,
                risk_category=risk_category,
                risk_description=risk_description,
                risk_color=risk_color,
                feature_descriptions=feature_descriptions,
                charts=charts,
                timestamp=timestamp,
                report_type="Individual Assessment"
            )
            
            # Create a temporary file for the HTML content
            with NamedTemporaryFile(suffix='.html', delete=False) as f:
                html_path = f.name
                f.write(html_content.encode('utf-8'))
            
            # Generate PDF from HTML
            pdf_data = pdfkit.from_file(html_path, False, options=self.pdf_options)
            
            # Clean up temporary file
            os.unlink(html_path)
            
            return pdf_data
        
        except Exception as e:
            # Handle any errors in report generation
            print(f"Error generating individual report: {str(e)}")
            raise
    
    def generate_dataset_report(self, df, evaluation_metrics, charts, feature_descriptions):
        """
        Generate a PDF report for a dataset assessment.
        
        Args:
            df (DataFrame): Original dataset with predictions
            evaluation_metrics (dict): Dictionary with evaluation metrics
            charts (dict): Dictionary with encoded chart images
            feature_descriptions (dict): Dictionary with feature descriptions
            
        Returns:
            bytes: PDF report as bytes
        """
        try:
            # Load template
            template = self.env.get_template('medical_report_template.html')
            
            # Format timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Extract key metrics
            classification_report = evaluation_metrics['classification_report']
            roc_auc = evaluation_metrics['roc_auc']
            
            # Get dataset statistics
            dataset_size = len(df)
            if 'stroke' in df.columns:
                positive_cases = df['stroke'].sum()
                negative_cases = dataset_size - positive_cases
                class_distribution = {
                    'Stroke': int(positive_cases),
                    'No Stroke': int(negative_cases)
                }
            else:
                class_distribution = {"Note": "No actual stroke labels in dataset"}
            
            # Get descriptive statistics for numeric features
            numeric_features = ['age', 'avg_glucose_level', 'bmi']
            descriptive_stats = df[numeric_features].describe().round(2).to_dict()
            
            # Render template
            html_content = template.render(
                dataset_size=dataset_size,
                class_distribution=class_distribution,
                classification_metrics=classification_report,
                roc_auc=roc_auc,
                descriptive_stats=descriptive_stats,
                feature_descriptions=feature_descriptions,
                charts=charts,
                timestamp=timestamp,
                report_type="Medical Expert Assessment"
            )
            
            # Create a temporary file for the HTML content
            with NamedTemporaryFile(suffix='.html', delete=False) as f:
                html_path = f.name
                f.write(html_content.encode('utf-8'))
            
            # Generate PDF from HTML
            pdf_data = pdfkit.from_file(html_path, False, options=self.pdf_options)
            
            # Clean up temporary file
            os.unlink(html_path)
            
            return pdf_data
        
        except Exception as e:
            # Handle any errors in report generation
            print(f"Error generating dataset report: {str(e)}")
            raise