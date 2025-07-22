"""
Google Drive client for API operations.
Handles Google Drive service creation and management.
"""

from googleapiclient.discovery import build

class DriveClient:
    """Google Drive API client for file operations."""
    
    def __init__(self):
        """Initialize the Drive client."""
        pass
    
    def create_service(self, credentials):
        """
        Create Google Drive service instance.
        
        Args:
            credentials: Google OAuth credentials
            
        Returns:
            Resource: Google Drive service instance
        """
        try:
            service = build('drive', 'v3', credentials=credentials)
            return service
        except Exception as e:
            print(f"Error creating Drive service: {str(e)}")
            return None 