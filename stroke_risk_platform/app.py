import streamlit as st
import pandas as pd
import os
import sys
from PIL import Image
import importlib
import config

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath('.'))

# Create necessary directories
for directory in ["models", "assets/templates", "assets/images", "assets/css"]:
    os.makedirs(directory, exist_ok=True)

def load_css():
    """Load custom CSS styles."""
    with open('assets/css/style.css', 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

class MultiApp:
    """Framework for combining multiple Streamlit applications."""
    
    def __init__(self):
        """Initialize the multi-app framework."""
        self.apps = {}
    
    def add_app(self, title, func):
        """
        Add a new application to the multi-app framework.
        
        Args:
            title (str): Application title
            func (callable): Function to render the app
        """
        self.apps[title] = func
    
    def run(self):
        """Run the selected application."""
        # Add custom CSS
        try:
            load_css()
        except:
            st.write("Custom CSS not loaded")
        
        # Set up the sidebar
        st.sidebar.title(config.APP_NAME)
        st.sidebar.markdown("---")
        
        # User type selection
        user_type = st.sidebar.radio(
            "Select User Type",
            config.USER_TYPES,
            index=0
        )
        
        st.sidebar.markdown("---")
        
        # Define available pages based on user type
        if user_type == "Individual":
            available_pages = ["Home", "Individual Assessment", "Chatbot Support"]
        else:  # Medical Expert
            available_pages = ["Home", "Medical Expert", "Chatbot Support"]
        
        # App selection
        app_selection = st.sidebar.radio(
            "Navigation",
            available_pages
        )
        
        # Run the selected app
        self.apps[app_selection]()
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.info(
            f"{config.APP_NAME} v{config.APP_VERSION}\n\n"
            "© 2025 StrokeRisk Insight. All rights reserved."
        )

# Import individual pages
def load_page(page_name):
    """Dynamically import a page module."""
    try:
        return importlib.import_module(f"pages.{page_name}")
    except ImportError as e:
        st.error(f"Could not load page {page_name}: {e}")
        return None

# Initialize the multi-app
app = MultiApp()

# Load pages
home_page = load_page("home")
individual_page = load_page("individual")
medical_expert_page = load_page("medical_expert")
chatbot_page = load_page("chatbot_interface")

# Add pages to the app
if home_page:
    app.add_app("Home", home_page.app)
if individual_page:
    app.add_app("Individual Assessment", individual_page.app)
if medical_expert_page:
    app.add_app("Medical Expert", medical_expert_page.app)
if chatbot_page:
    app.add_app("Chatbot Support", chatbot_page.app)

# Run the app
if __name__ == "__main__":
    app.run()