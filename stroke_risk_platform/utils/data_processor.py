import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

class DataProcessor:
    """
    Class for processing stroke prediction data, handling both individual inputs
    and datasets for medical experts.
    """
    
    def __init__(self):
        """Initialize the DataProcessor with feature definitions."""
        # Define feature categories
        self.numeric_features = ['age', 'avg_glucose_level', 'bmi']
        self.categorical_features = ['gender', 'hypertension', 'heart_disease', 
                                    'ever_married', 'work_type', 'Residence_type', 
                                    'smoking_status']
        
        # Setup preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        # Feature ranges for validation
        self.feature_ranges = {
            'age': (0, 120),
            'avg_glucose_level': (50, 300),
            'bmi': (10, 50),
            'gender': ['Male', 'Female', 'Other'],
            'hypertension': [0, 1],
            'heart_disease': [0, 1],
            'ever_married': ['Yes', 'No'],
            'work_type': ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'],
            'Residence_type': ['Urban', 'Rural'],
            'smoking_status': ['formerly smoked', 'never smoked', 'smokes', 'Unknown']
        }
    
    def validate_individual_input(self, input_data):
        """
        Validate individual input data against acceptable ranges.
        
        Args:
            input_data (dict): Dictionary containing individual patient data
            
        Returns:
            tuple: (is_valid, error_message)
        """
        for feature, value in input_data.items():
            if feature in self.feature_ranges:
                # Check numeric ranges
                if feature in self.numeric_features:
                    min_val, max_val = self.feature_ranges[feature]
                    if value < min_val or value > max_val:
                        return False, f"{feature} should be between {min_val} and {max_val}"
                
                # Check categorical values
                elif feature in self.categorical_features:
                    if value not in self.feature_ranges[feature]:
                        valid_values = ', '.join(str(x) for x in self.feature_ranges[feature])
                        return False, f"{feature} should be one of: {valid_values}"
        
        return True, ""
    
    def preprocess_individual(self, input_data):
        """
        Process a single individual's data for prediction.
        
        Args:
            input_data (dict): Dictionary containing individual patient data
            
        Returns:
            DataFrame: Preprocessed data ready for the model
        """
        # Convert dictionary to DataFrame
        df = pd.DataFrame([input_data])
        
        # Handle potential missing values
        df['bmi'].fillna(df['bmi'].median() if not df['bmi'].empty else 25, inplace=True)
        
        # Apply preprocessing
        try:
            # Fit and transform in one step for a single instance
            processed_data = self.preprocessor.fit_transform(df)
            return processed_data
        except Exception as e:
            raise ValueError(f"Error preprocessing individual data: {str(e)}")
    
    def preprocess_dataset(self, df):
        """
        Process a dataset uploaded by a medical expert.
        
        Args:
            df (DataFrame): DataFrame with patient data
            
        Returns:
            tuple: (preprocessed_data, original_df_with_predictions)
        """
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Check for required columns
        required_columns = self.numeric_features + self.categorical_features
        missing_cols = [col for col in required_columns if col not in df_copy.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        
        # Handle missing values
        if 'bmi' in df_copy.columns:
            df_copy['bmi'].fillna(df_copy['bmi'].median(), inplace=True)
        
        # Apply preprocessing
        try:
            processed_data = self.preprocessor.fit_transform(df_copy)
            return processed_data, df_copy
        except Exception as e:
            raise ValueError(f"Error preprocessing dataset: {str(e)}")
    
    def get_feature_descriptions(self):
        """
        Returns descriptions of features for the report and UI.
        
        Returns:
            dict: Feature descriptions and medical context
        """
        return {
            'age': "Patient's age in years. Age is a significant risk factor for stroke.",
            'avg_glucose_level': "Average glucose level in blood (mg/dL). Elevated levels may indicate diabetes, a major stroke risk factor.",
            'bmi': "Body Mass Index, a measure of body fat based on height and weight. Higher BMI is associated with increased stroke risk.",
            'gender': "Patient's gender. Some stroke risk factors vary by gender.",
            'hypertension': "Whether the patient has hypertension (1) or not (0). High blood pressure significantly increases stroke risk.",
            'heart_disease': "Whether the patient has heart disease (1) or not (0). Heart conditions can increase stroke risk.",
            'ever_married': "Marital status can be a proxy for various lifestyle and social factors.",
            'work_type': "Type of employment, which can correlate with stress levels, activity, and other health factors.",
            'Residence_type': "Whether the patient lives in an urban or rural area, which may affect healthcare access.",
            'smoking_status': "Smoking habit of the patient. Smoking significantly increases stroke risk."
        }