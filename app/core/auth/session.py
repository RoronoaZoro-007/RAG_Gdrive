"""
Session state management for Streamlit application.
Handles initialization and updates of session state variables.
"""

import streamlit as st
from config.constants import SESSION_DEFAULTS

class SessionManager:
    """Manages Streamlit session state for the application."""
    
    @staticmethod
    def initialize_session():
        """Initialize session state with default values."""
        for key, default_value in SESSION_DEFAULTS.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @staticmethod
    def update_session(key: str, value):
        """Update a specific session state variable."""
        st.session_state[key] = value
    
    @staticmethod
    def get_session(key: str):
        """Get a specific session state variable."""
        return st.session_state.get(key)
    
    @staticmethod
    def clear_session():
        """Clear all session state variables."""
        for key in SESSION_DEFAULTS.keys():
            if key in st.session_state:
                del st.session_state[key]
    
    @staticmethod
    def is_authenticated() -> bool:
        """Check if user is authenticated."""
        return st.session_state.get('authenticated', False)
    
    @staticmethod
    def get_user_email() -> str:
        """Get the current user's email."""
        user_info = st.session_state.get('user_info')
        return user_info.get('email', 'unknown') if user_info else 'unknown'
    
    @staticmethod
    def get_credentials():
        """Get the current user's credentials."""
        return st.session_state.get('credentials')
    
    @staticmethod
    def get_chat_history():
        """Get the current chat history."""
        return st.session_state.get('chat_history', [])
    
    @staticmethod
    def add_to_chat_history(role: str, content: str):
        """Add a message to the chat history."""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        st.session_state.chat_history.append({
            "role": role,
            "content": content
        })
    
    @staticmethod
    def clear_chat_history():
        """Clear the chat history."""
        st.session_state.chat_history = [] 