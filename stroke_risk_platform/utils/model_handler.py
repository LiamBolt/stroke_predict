import joblib
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.calibration import calibration_curve
import pickle

class ModelHandler:
    """
    Class for handling the stroke prediction model, including loading,
    prediction, and evaluation functionalities.
    """
    
    def __init__(self, model_path="models/xgboost_stroke_model.joblib"):
        """
        Initialize the model handler.
        
        Args:
            model_path (str): Path to the model file
        """
        self.model_path = model_path
        self.model = self._load_model()
        
    def _load_model(self):
        """
        Load the model from disk.
        
        Returns:
            object: The loaded model
        """
        try:
            if os.path.exists(self.model_path):
                return joblib.load(self.model_path)
            else:
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")
    
    def predict(self, processed_data):
        """
        Make predictions using the loaded model.
        
        Args:
            processed_data: Preprocessed data ready for the model
            
        Returns:
            tuple: (Predicted class, Prediction probability)
        """
        try:
            # Get raw probabilities
            probabilities = self.model.predict_proba(processed_data)
            
            # Extract probability of positive class (stroke)
            positive_probs = probabilities[:, 1]
            
            # Get class predictions
            predictions = self.model.predict(processed_data)
            
            return predictions, positive_probs
        except Exception as e:
            raise ValueError(f"Error making predictions: {str(e)}")
    
    def get_risk_category(self, probability):
        """
        Convert probability to a risk category.
        
        Args:
            probability (float): Probability of stroke
            
        Returns:
            tuple: (risk_category, color, description)
        """
        if probability < 0.2:
            return "Low Risk", "#2ECC71", "Your risk factors suggest a low probability of stroke."
        elif probability < 0.4:
            return "Moderate Risk", "#F39C12", "You have some risk factors that merit attention."
        elif probability < 0.6:
            return "Elevated Risk", "#E67E22", "Your risk factors indicate an elevated stroke probability."
        elif probability < 0.8:
            return "High Risk", "#E74C3C", "Multiple significant risk factors detected, indicating high risk."
        else:
            return "Very High Risk", "#C0392B", "Urgent attention to risk factors is strongly recommended."
    
    def evaluate_dataset(self, y_true, y_pred_proba):
        """
        Evaluate model performance on a dataset.
        
        Args:
            y_true (array): True labels
            y_pred_proba (array): Predicted probabilities
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Convert probabilities to binary predictions using 0.5 threshold
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
        
        # Calculate classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        return {
            'confusion_matrix': cm,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'prob_true': prob_true,
            'prob_pred': prob_pred,
            'classification_report': report
        }
    
    def get_model_info(self):
        """
        Get information about the model.
        
        Returns:
            dict: Dictionary containing model information
        """
        return {
            'model_type': 'XGBoost Classifier',
            'features': [
                'age', 'avg_glucose_level', 'bmi', 'gender', 
                'hypertension', 'heart_disease', 'ever_married', 
                'work_type', 'Residence_type', 'smoking_status'
            ],
            'feature_importances': self._get_feature_importances(),
            'model_description': (
                "This XGBoost model was trained on a comprehensive stroke dataset, "
                "optimized to identify individuals at risk of stroke based on "
                "demographic and health indicators."
            )
        }
    
    def _get_feature_importances(self):
        """
        Extract feature importances from the model.
        
        Returns:
            dict: Dictionary with feature names and their importance scores
        """
        try:
            # For XGBoost models
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                # Return top features (adjust as needed)
                feature_names = [
                    'age', 'avg_glucose_level', 'bmi', 'gender', 
                    'hypertension', 'heart_disease', 'ever_married', 
                    'work_type', 'Residence_type', 'smoking_status'
                ]
                # Create sorted dictionary of feature importances
                importance_dict = {name: float(score) for name, score in zip(feature_names, importances)}
                sorted_importances = {k: v for k, v in sorted(importance_dict.items(), 
                                                          key=lambda item: item[1], 
                                                          reverse=True)}
                return sorted_importances
            else:
                return {"Note": "Feature importance not available for this model"}
        except:
            return {"Note": "Could not extract feature importances"}