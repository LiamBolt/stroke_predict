import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io
import base64
from matplotlib.figure import Figure
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Visualizer:
    """
    Class for creating visualizations for the stroke prediction app.
    """
    
    def __init__(self):
        """Initialize the Visualizer with styling settings."""
        # Set default style for matplotlib
        sns.set_style("whitegrid")
        self.colors = {
            'primary': '#3498db',
            'secondary': '#2ecc71',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'light': '#ecf0f1',
            'dark': '#2c3e50'
        }
    
    def create_feature_importance_plot(self, importances, title="Feature Importance"):
        """
        Create a horizontal bar chart of feature importances.
        
        Args:
            importances (dict): Dictionary of feature names and importance values
            title (str): Title for the plot
            
        Returns:
            str: Base64 encoded image
        """
        # Sort features by importance
        features = list(importances.keys())
        values = list(importances.values())
        
        # Create DataFrame for sorting
        df = pd.DataFrame({'Feature': features, 'Importance': values})
        df = df.sort_values('Importance', ascending=False)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot horizontal bars
        sns.barplot(x='Importance', y='Feature', data=df, palette='viridis', ax=ax)
        
        # Customize plot
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        
        # Encode the bytes buffer to base64
        data = base64.b64encode(buf.getbuffer()).decode("utf-8")
        return f"data:image/png;base64,{data}"
    
    def create_confusion_matrix_plot(self, cm, title="Confusion Matrix"):
        """
        Create a heatmap visualization of the confusion matrix.
        
        Args:
            cm (array): Confusion matrix array
            title (str): Title for the plot
            
        Returns:
            str: Base64 encoded image
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=['No Stroke', 'Stroke'],
                    yticklabels=['No Stroke', 'Stroke'], ax=ax)
        
        # Customize plot
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        
        # Encode the bytes buffer to base64
        data = base64.b64encode(buf.getbuffer()).decode("utf-8")
        return f"data:image/png;base64,{data}"
    
    def create_roc_curve_plot(self, fpr, tpr, roc_auc, title="ROC Curve"):
        """
        Create a ROC curve plot.
        
        Args:
            fpr (array): False positive rates
            tpr (array): True positive rates
            roc_auc (float): Area under the ROC curve
            title (str): Title for the plot
            
        Returns:
            str: Base64 encoded image
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color=self.colors['primary'], lw=2, 
                label=f'ROC curve (area = {roc_auc:.3f})')
        
        # Plot diagonal line (random guessing)
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        
        # Customize plot
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.legend(loc="lower right")
        
        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        
        # Encode the bytes buffer to base64
        data = base64.b64encode(buf.getbuffer()).decode("utf-8")
        return f"data:image/png;base64,{data}"
    
    def create_calibration_plot(self, prob_true, prob_pred, title="Calibration Plot"):
        """
        Create a calibration curve plot.
        
        Args:
            prob_true (array): True probabilities
            prob_pred (array): Predicted probabilities
            title (str): Title for the plot
            
        Returns:
            str: Base64 encoded image
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot calibration curve
        ax.plot(prob_pred, prob_true, 's-', color=self.colors['primary'], label='Model')
        
        # Plot diagonal line (perfect calibration)
        ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly calibrated')
        
        # Customize plot
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True)
        
        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        
        # Encode the bytes buffer to base64
        data = base64.b64encode(buf.getbuffer()).decode("utf-8")
        return f"data:image/png;base64,{data}"
    
    def create_individual_risk_gauge(self, probability):
        """
        Create an interactive gauge chart for individual risk visualization using Plotly.
        
        Args:
            probability (float): Probability of stroke
            
        Returns:
            fig: A plotly figure object
        """
        # Determine risk category and color
        if probability < 0.2:
            risk_level = "Low Risk"
            color = "#2ECC71"  # Green
        elif probability < 0.4:
            risk_level = "Moderate Risk"
            color = "#F39C12"  # Orange
        elif probability < 0.6:
            risk_level = "Elevated Risk"
            color = "#E67E22"  # Dark Orange
        elif probability < 0.8:
            risk_level = "High Risk"
            color = "#E74C3C"  # Red
        else:
            risk_level = "Very High Risk"
            color = "#C0392B"  # Dark Red
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Stroke Risk: {risk_level}", 'font': {'size': 24}},
            delta={'reference': 20, 'increasing': {'color': "red"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 20], 'color': '#2ECC71'},
                    {'range': [20, 40], 'color': '#F39C12'},
                    {'range': [40, 60], 'color': '#E67E22'},
                    {'range': [60, 80], 'color': '#E74C3C'},
                    {'range': [80, 100], 'color': '#C0392B'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': probability * 100
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="white",
            font={'color': "darkblue", 'family': "Arial"}
        )
        
        return fig
    
    def create_feature_comparison_plot(self, individual_data, feature_ranges):
        """
        Create a radar chart comparing individual's values with typical ranges.
        
        Args:
            individual_data (dict): Dictionary containing individual's values
            feature_ranges (dict): Dictionary with min/max values for features
            
        Returns:
            fig: A plotly figure object
        """
        # Select numeric features for the radar chart
        numeric_features = ['age', 'avg_glucose_level', 'bmi']
        
        # Normalize values between 0 and 1 for radar chart
        max_values = [feature_ranges[feat][1] for feat in numeric_features]
        min_values = [feature_ranges[feat][0] for feat in numeric_features]
        
        normalized_values = []
        for i, feat in enumerate(numeric_features):
            value = individual_data[feat]
            normalized = (value - min_values[i]) / (max_values[i] - min_values[i])
            normalized_values.append(normalized)
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_values,
            theta=numeric_features,
            fill='toself',
            name='Your Values',
            line_color='rgba(52, 152, 219, 0.8)',
            fillcolor='rgba(52, 152, 219, 0.3)'
        ))
        
        # Add reference values (mid-point)
        fig.add_trace(go.Scatterpolar(
            r=[0.5, 0.5, 0.5],
            theta=numeric_features,
            fill='toself',
            name='Reference',
            line_color='rgba(46, 204, 113, 0.8)',
            fillcolor='rgba(46, 204, 113, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Your Values Compared to Normal Ranges",
            height=400
        )
        
        return fig
    
    def create_dataset_distribution_plots(self, df):
        """
        Create distribution plots for a dataset.
        
        Args:
            df (DataFrame): DataFrame with patient data
            
        Returns:
            dict: Dictionary of plotly figures
        """
        figures = {}
        
        # Age distribution
        fig_age = px.histogram(
            df, x="age", 
            color="stroke" if "stroke" in df.columns else None,
            marginal="box", 
            title="Age Distribution",
            color_discrete_map={0: "#3498db", 1: "#e74c3c"},
            labels={"stroke": "Stroke", "age": "Age (years)"}
        )
        figures["age_dist"] = fig_age
        
        # Glucose level distribution
        fig_glucose = px.histogram(
            df, x="avg_glucose_level", 
            color="stroke" if "stroke" in df.columns else None,
            marginal="box", 
            title="Average Glucose Level Distribution",
            color_discrete_map={0: "#3498db", 1: "#e74c3c"},
            labels={"stroke": "Stroke", "avg_glucose_level": "Glucose Level (mg/dL)"}
        )
        figures["glucose_dist"] = fig_glucose
        
        # BMI distribution
        fig_bmi = px.histogram(
            df, x="bmi", 
            color="stroke" if "stroke" in df.columns else None,
            marginal="box", 
            title="BMI Distribution",
            color_discrete_map={0: "#3498db", 1: "#e74c3c"},
            labels={"stroke": "Stroke", "bmi": "BMI"}
        )
        figures["bmi_dist"] = fig_bmi
        
        # Gender counts
        fig_gender = px.bar(
            df.groupby("gender").size().reset_index(name="count"), 
            x="gender", y="count", 
            title="Gender Distribution",
            color="gender", 
            labels={"gender": "Gender", "count": "Count"}
        )
        figures["gender_dist"] = fig_gender
        
        return figures