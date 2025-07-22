"""
Settings management for the SheetsLoader application.
Handles environment variable loading and configuration validation.
"""

import os
from dotenv import load_dotenv
from typing import Optional

class Settings:
    """Central configuration management for the application."""
    
    def __init__(self):
        """Initialize settings by loading environment variables."""
        self.load_environment()
        self.validate_config()
    
    def load_environment(self):
        """Load environment variables from .env file."""
        load_dotenv()
        
        # Google API configuration
        self.GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        self.GOOGLE_CLIENT_ID = os.getenv('GOOGLE_OAUTH_CLIENT_ID')
        self.GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_OAUTH_CLIENT_SECRET')
        
        # OpenSearch configuration
        self.OPENSEARCH_URL = os.getenv('OPENSEARCH_URL', 'http://localhost:9200')
        self.OPENSEARCH_USERNAME = os.getenv('OPENSEARCH_USERNAME', '')
        self.OPENSEARCH_PASSWORD = os.getenv('OPENSEARCH_PASSWORD', '')
        
        # LlamaIndex configuration
        self.LLAMA_CLOUD_API_KEY = os.getenv('LLAMA_CLOUD_API_KEY')
        
        # Development settings
        self.OAUTHLIB_INSECURE_TRANSPORT = os.getenv('OAUTHLIB_INSECURE_TRANSPORT', '1')
        
        # Set environment variable for local development
        os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = self.OAUTHLIB_INSECURE_TRANSPORT
    
    def validate_config(self):
        """Validate that required configuration is present."""
        # Only validate if we're not in a test environment
        if os.getenv('TESTING'):
            return
            
        missing_vars = []
        
        if not self.GOOGLE_API_KEY:
            missing_vars.append("GOOGLE_API_KEY or GEMINI_API_KEY")
        
        if not self.GOOGLE_CLIENT_ID:
            missing_vars.append("GOOGLE_OAUTH_CLIENT_ID")
        
        if not self.GOOGLE_CLIENT_SECRET:
            missing_vars.append("GOOGLE_OAUTH_CLIENT_SECRET")
        
        if missing_vars:
            print(f"⚠️ Warning: Missing environment variables: {', '.join(missing_vars)}")
            print("⚠️ Authentication and some features may not work without these variables.")
            # Don't raise an error, just warn
    
    def get_google_client_config(self) -> dict:
        """Get Google OAuth client configuration dictionary."""
        return {
            "web": {
                "client_id": self.GOOGLE_CLIENT_ID,
                "client_secret": self.GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "redirect_uris": ["http://localhost:8501"]
            }
        }

# Global settings instance
settings = Settings() 