"""
File downloading from Google Drive.
Handles downloading of various file types from Google Drive.
"""

import os
from googleapiclient.http import MediaIoBaseDownload

class FileDownloader:
    """Handles downloading of files from Google Drive."""
    
    def download_file(self, service, file_id: str, file_name: str, download_dir: str, file_type="file", export_mime_type=None) -> str:
        """
        Generic function to download files from Google Drive.
        
        Args:
            service: Google Drive service instance
            file_id (str): Google Drive file ID
            file_name (str): Name of the file
            download_dir (str): Directory to download to
            file_type (str): Type of file for naming
            export_mime_type (str, optional): MIME type for export
            
        Returns:
            str: Path to downloaded file or None if failed
        """
        try:
            # Determine request type
            if export_mime_type:
                # Export file (for Google Docs/Sheets)
                request = service.files().export_media(
                    fileId=file_id,
                    mimeType=export_mime_type
                )
            else:
                # Direct download
                request = service.files().get_media(fileId=file_id)
            
            # Determine file path
            if file_type == "sheet" and not file_name.endswith('.xlsx'):
                file_path = os.path.join(download_dir, f"{file_id}_{file_name}.xlsx")
            else:
                file_path = os.path.join(download_dir, f"{file_id}_{file_name}")
            
            # Download to file
            with open(file_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                    if status and file_type == "google_doc":
                        print(f"Download {int(status.progress() * 100)}%")
            
            return file_path
            
        except Exception as e:
            print(f"Failed to download {file_name}: {str(e)}")
            return None
    
    def download_pdf(self, service, file_id: str, file_name: str, download_dir: str) -> str:
        """
        Download a single PDF file from Google Drive.
        
        Args:
            service: Google Drive service instance
            file_id (str): Google Drive file ID
            file_name (str): Name of the PDF file
            download_dir (str): Directory to download to
            
        Returns:
            str: Path to downloaded PDF or None if failed
        """
        return self.download_file(service, file_id, file_name, download_dir, "pdf")
    
    def download_sheet(self, service, file_id: str, file_name: str, download_dir: str) -> str:
        """
        Download a sheet file from Google Drive.
        
        Args:
            service: Google Drive service instance
            file_id (str): Google Drive file ID
            file_name (str): Name of the sheet file
            download_dir (str): Directory to download to
            
        Returns:
            str: Path to downloaded sheet or None if failed
        """
        # For Excel files, download directly; for Google Sheets, export as Excel
        export_mime_type = None
        if not file_name.endswith(('.xlsx', '.xls')):
            export_mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        
        return self.download_file(service, file_id, file_name, download_dir, "sheet", export_mime_type) 