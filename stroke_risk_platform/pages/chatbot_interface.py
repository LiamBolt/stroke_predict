import streamlit as st
import pandas as pd
import os
import sys
from PIL import Image
import time

# Add the project root to Python path
sys.path.insert(0, os.path.abspath("."))

# Import utilities
from utils.chatbot import StrokeChatbot
import config

def app():
    """Chatbot interface page for the Stroke Risk Assessment Platform."""
    
    # Page title and introduction
    st.title("StrokeAssist AI Chatbot")
    
    st.markdown("""
    Have questions about stroke risk, prevention, or your assessment results? Chat with our AI assistant!
    StrokeAssist is designed to provide information about stroke risk factors, explain your assessment results,
    and answer general questions about stroke prevention and management.
    
    <div style="background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
        <strong>💬 How can StrokeAssist help?</strong>
        <ul>
            <li>Explain risk factors in your assessment</li>
            <li>Provide information about stroke prevention</li>
            <li>Answer questions about stroke warning signs</li>
            <li>Clarify stroke statistics and research findings</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chatbot" not in st.session_state:
        try:
            st.session_state.chatbot = StrokeChatbot(api_key=config.GEMINI_API_KEY)
        except Exception as e:
            st.error(f"Error initializing chatbot: {str(e)}")
            st.error("Please ensure a valid Gemini API key is provided in config.py or as an environment variable.")
            return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Get user input
    if prompt := st.chat_input("Ask StrokeAssist a question about stroke risk or your assessment..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get risk information if available
        risk_info = None
        if "individual_result" in st.session_state and st.session_state.individual_result is not None:
            risk_probability = st.session_state.individual_result
            risk_category, _, _ = st.session_state.risk_info
            input_data = st.session_state.individual_input
            
            risk_info = {
                "category": risk_category,
                "probability": risk_probability,
                "age": input_data["age"],
                "bmi": input_data["bmi"],
                "avg_glucose_level": input_data["avg_glucose_level"],
                "hypertension": input_data["hypertension"],
                "heart_disease": input_data["heart_disease"],
                "smoking_status": input_data["smoking_status"]
            }
        
        # Display assistant response with a typing effect
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Show a loading indicator
            with st.spinner("StrokeAssist is thinking..."):
                try:
                    # Get response from chatbot
                    response = st.session_state.chatbot.get_response(prompt, risk_info)
                    
                    # Simulate typing effect
                    for chunk in response.split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        # Add a blinking cursor
                        message_placeholder.markdown(full_response + "▌")
                    
                    # Display final response
                    message_placeholder.markdown(full_response)
                    
                    # Add response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    error_message = f"I'm sorry, I encountered an error: {str(e)}. Please try again later."
                    message_placeholder.markdown(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    # Add reset button to clear chat history
    if st.session_state.messages and st.button("Reset Conversation"):
        st.session_state.messages = []
        st.session_state.chatbot.reset_conversation()
        st.experimental_rerun()
    
    # Add disclaimer
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px;">
        <strong>AI Assistant Disclaimer:</strong> StrokeAssist provides general information and is not a substitute 
        for professional medical advice. The assistant may occasionally provide inaccurate information. Always consult 
        with a qualified healthcare provider for medical concerns.
    </div>
    """, unsafe_allow_html=True)