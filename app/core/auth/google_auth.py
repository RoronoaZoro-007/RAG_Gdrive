"""
Google OAuth authentication management.
Handles Google Drive authentication flow and user credentials.
"""

import streamlit as st
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from config.settings import settings
from config.constants import SCOPES

class GoogleAuthManager:
    """Manages Google OAuth authentication flow."""
    
    def __init__(self):
        """Initialize the authentication manager."""
        self.client_config = settings.get_google_client_config()
    
    def create_oauth_flow(self) -> Flow:
        """
        Create OAuth flow with proper configuration.
        
        Returns:
            Flow: Configured OAuth flow object
            
        Raises:
            ValueError: If OAuth credentials are missing
        """
        if not settings.GOOGLE_CLIENT_ID:
            raise ValueError("GOOGLE_OAUTH_CLIENT_ID not found in environment variables")
        
        if not settings.GOOGLE_CLIENT_SECRET:
            raise ValueError("GOOGLE_OAUTH_CLIENT_SECRET not found in environment variables")
        
        # Configure flow with insecure transport for local development
        flow = Flow.from_client_config(
            self.client_config, 
            SCOPES,
            redirect_uri="http://localhost:8501"
        )
        
        return flow
    
    def get_google_oauth_url(self) -> str:
        """
        Generate Google OAuth URL for authentication.
        
        Returns:
            str: OAuth authorization URL or None if failed
        """
        try:
            flow = self.create_oauth_flow()
            
            auth_url, _ = flow.authorization_url(
                access_type='offline',
                include_granted_scopes='true',
                prompt='consent'
            )
            
            # Store flow in session for later use
            st.session_state.oauth_flow = flow
            
            return auth_url
            
        except Exception as e:
            st.error(f"Error generating OAuth URL: {str(e)}")
            return None
    
    def authenticate_with_google(self, callback_url: str) -> tuple:
        """
        Authenticate user with Google using callback URL.
        
        Args:
            callback_url (str): OAuth callback URL with authorization code
            
        Returns:
            tuple: (credentials, user_info) or (None, None) if failed
        """
        try:
            # Get or create flow
            if 'oauth_flow' not in st.session_state:
                flow = self.create_oauth_flow()
                st.session_state.oauth_flow = flow
            else:
                flow = st.session_state.oauth_flow
            
            # Validate callback URL format
            if not isinstance(callback_url, str) or not callback_url.startswith('http://localhost:8501'):
                st.error("Invalid callback URL received.")
                return None, None
                
            if 'code=' not in callback_url:
                st.error("No authorization code found. Please try signing in again.")
                return None, None
            
            # Exchange authorization code for credentials
            # Suppress scope mismatch warnings
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                flow.fetch_token(authorization_response=callback_url)
            
            credentials = flow.credentials
            
            # Get user info
            service = build('oauth2', 'v2', credentials=credentials)
            user_info = service.userinfo().get().execute()
            print(f"🔍 [GOOGLE_AUTH] Retrieved user info: {user_info}")
            
            return credentials, user_info
            
        except Exception as e:
            st.error(f"Authentication failed: {str(e)}")
            return None, None 