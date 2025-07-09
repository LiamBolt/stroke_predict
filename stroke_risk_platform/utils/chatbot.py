import requests
import json
import os
from datetime import datetime
import textwrap

class StrokeChatbot:
    """
    Class for handling the Gemini-powered chatbot functionality 
    for the stroke risk assessment platform.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the chatbot with API key and context.
        
        Args:
            api_key (str, optional): Gemini API key. If None, read from environment.
        """
        # Use provided API key or get from environment
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or provide in constructor.")
            
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        
        # Setup context for the chatbot
        self.system_context = self._get_system_context()
        
        # Initialize conversation history
        self.conversation_history = [
            {"role": "system", "content": self.system_context}
        ]
    
    def _get_system_context(self):
        """
        Create the system context for the chatbot.
        
        Returns:
            str: System context describing the chatbot's role and knowledge
        """
        return textwrap.dedent("""
            You are StrokeAssist, a specialized healthcare AI assistant focused on stroke risk assessment.
            
            Key Facts About You:
            - You're integrated with a stroke risk prediction model based on XGBoost
            - You can explain risk factors and provide general stroke-related information
            - Your primary purpose is to help users understand their stroke risk assessment results
            - You're knowledgeable about stroke prevention, warning signs, and risk factors
            
            Guidelines:
            1. Provide accurate medical information related to stroke risk, prevention, and treatment
            2. Explain model predictions and risk factors in simple, understandable terms
            3. Avoid giving specific medical advice that should come from healthcare professionals
            4. Clearly state the limitations of your knowledge when appropriate
            5. Maintain a professional, empathetic tone
            6. If asked about topics completely unrelated to stroke or health, politely redirect to stroke-related topics
            7. Never claim to be a substitute for professional medical care
            
            Key stroke risk factors include:
            - Age (risk increases with age)
            - High blood pressure (hypertension)
            - Smoking
            - Diabetes
            - High cholesterol
            - Physical inactivity
            - Obesity
            - Family history of stroke
            - Previous stroke or TIA
            - Heart disease
            - Gender (risk varies)
            
            Warning signs of stroke (F.A.S.T.):
            - Face drooping
            - Arm weakness
            - Speech difficulties
            - Time to call emergency services
            
            Current date: {current_date}
        """).format(current_date=datetime.now().strftime("%Y-%m-%d"))
    
    def get_response(self, user_message, risk_info=None):
        """
        Get a response from the Gemini API.
        
        Args:
            user_message (str): The user's message
            risk_info (dict, optional): The user's risk assessment information
            
        Returns:
            str: The chatbot's response
        """
        # Add risk information context if available
        context_message = ""
        if risk_info:
            context_message = f"""
            User's stroke risk assessment information:
            - Risk category: {risk_info['category']}
            - Risk probability: {risk_info['probability']:.2f}
            - Age: {risk_info['age']}
            - BMI: {risk_info['bmi']:.1f}
            - Average Glucose: {risk_info['avg_glucose_level']:.1f}
            - Hypertension: {"Yes" if risk_info['hypertension'] else "No"}
            - Heart Disease: {"Yes" if risk_info['heart_disease'] else "No"}
            - Smoking Status: {risk_info['smoking_status']}
            """
            
            # Add this context message to conversation history
            self.conversation_history.append({
                "role": "system",
                "content": context_message
            })
        
        # Add user message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Prepare request payload
        payload = {
            "contents": [
                {
                    "role": message["role"],
                    "parts": [{"text": message["content"]}]
                }
                for message in self.conversation_history
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1024
            }
        }
        
        # Make request to Gemini API
        try:
            response = requests.post(
                f"{self.api_url}?key={self.api_key}",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload)
            )
            
            # Parse response
            if response.status_code == 200:
                response_data = response.json()
                if "candidates" in response_data and len(response_data["candidates"]) > 0:
                    candidate = response_data["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        bot_response = candidate["content"]["parts"][0]["text"]
                        
                        # Add bot response to conversation history
                        self.conversation_history.append({
                            "role": "system",
                            "content": bot_response
                        })
                        
                        return bot_response
            
            # Handle API errors
            error_message = f"Error from Gemini API: {response.status_code} - {response.text}"
            print(error_message)
            return "I'm sorry, I encountered an error. Please try again later."
            
        except Exception as e:
            print(f"Exception in chatbot: {str(e)}")
            return "I'm sorry, I encountered an unexpected error. Please try again."
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = [
            {"role": "system", "content": self.system_context}
        ]