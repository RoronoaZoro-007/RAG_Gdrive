"""
File scanning for Google Drive.
Handles discovery and listing of files in Google Drive.
"""

from googleapiclient.discovery import build

class FileScanner:
    """Handles scanning and discovery of files in Google Drive."""
    
    def scan_files(self, credentials, mime_types, file_type_name="files"):
        """
        Generic function to scan Google Drive for files with specified MIME types.
        
        Args:
            credentials: Google OAuth credentials
            mime_types (str or list): MIME type(s) to search for
            file_type_name (str): Name for logging purposes
            
        Returns:
            list: List of file information dictionaries
        """
        try:
            service = build('drive', 'v3', credentials=credentials)
            results = []
            
            # Handle multiple MIME types (for sheets)
            if isinstance(mime_types, str):
                mime_types = [mime_types]
            
            for mime_type in mime_types:
                query = f"mimeType='{mime_type}' and trashed=false"
                page_token = None
                
                while True:
                    response = service.files().list(
                        q=query,
                        spaces='drive',
                        fields='nextPageToken, files(id, name, mimeType, size, createdTime, modifiedTime, viewedByMeTime, owners, permissions, webViewLink, webContentLink, thumbnailLink, parents, trashed, starred, shared, viewedByMe, capabilities, exportLinks, appProperties, properties, imageMediaMetadata, videoMediaMetadata)',
                        pageToken=page_token
                    ).execute()
                    
                    results.extend(response.get('files', []))
                    page_token = response.get('nextPageToken', None)
                    
                    if page_token is None:
                        break
            
            return results
            
        except Exception as e:
            print(f"Error scanning Drive for {file_type_name}: {str(e)}")
            return [] 