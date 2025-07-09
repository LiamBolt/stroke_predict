import os

# Application configuration
APP_NAME = "StrokeRisk Insight"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "An advanced stroke risk assessment platform for individuals and medical experts"

# Model configuration
MODEL_PATH = "models/xgboost_stroke_model.joblib"

# Default risk threshold
RISK_THRESHOLD = 0.5

# Gemini API configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")  # Replace with your key if not in env vars

# Report configuration
REPORT_TEMPLATES_DIR = "assets/templates"

# User categories
USER_TYPES = ["Individual", "Medical Expert"]

# Navigation
PAGES = {
    "Home": "home",
    "Individual Assessment": "individual",
    "Medical Expert": "medical_expert",
    "Chatbot Support": "chatbot_interface"
}

# Create directories if they don't exist
for directory in ["models", "assets/templates", "assets/images", "assets/css"]:
    os.makedirs(directory, exist_ok=True)