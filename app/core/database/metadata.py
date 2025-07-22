"""
Metadata management for file storage and retrieval.
Handles file metadata operations including storage, retrieval, and cleanup.
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Optional, Tuple
from config.constants import VECTORSTORES_DIR, OPENSEARCH_INDEX, SHEETS_INDEX
from core.database.opensearch import OpenSearchClient

class MetadataManager:
    """Manages file metadata storage and retrieval operations."""
    
    def __init__(self):
        """Initialize metadata manager."""
        self.opensearch_client = OpenSearchClient()
    
    def get_metadata_path(self, user_email: str) -> str:
        """
        Get the path to the user's enhanced metadata file.
        
        Args:
            user_email (str): User's email address
            
        Returns:
            str: Path to the metadata file
        """
        print(f"🔍 [METADATA] Getting metadata path for user: {user_email}")
        user_hash = hashlib.md5(user_email.encode()).hexdigest()
        metadata_path = os.path.join("vectorstores", f"user_{user_hash}_enhanced_metadata.json")
        print(f"🔍 [METADATA] Metadata path: {metadata_path}")
        return metadata_path
    
    def save_metadata(self, file_metadata: Dict, user_email: str):
        """
        Save comprehensive metadata about processed files to both JSON and OpenSearch.
        
        Args:
            file_metadata (dict): File metadata to save
            user_email (str): User's email for data isolation
        """
        print(f"💾 [METADATA] Saving metadata for user: {user_email}")
        # Save to JSON file
        metadata_path = self.get_metadata_path(user_email)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        # Load existing metadata if it exists
        existing_metadata = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    existing_metadata = json.load(f)
            except Exception as e:
                print(f"Failed to load existing metadata: {e}")
        
        # Update with new metadata
        existing_metadata.update(file_metadata)
        
        # Save updated metadata to JSON
        with open(metadata_path, 'w') as f:
            json.dump(existing_metadata, f, indent=2)
        
        print(f"✅ Enhanced metadata saved to {metadata_path}")
        
        # Also save to OpenSearch using spaCy handler
        try:
            from core.retrieval.query_classifier import get_spacy_metadata_handler
            spacy_handler = get_spacy_metadata_handler()
            success = spacy_handler.store_metadata_in_opensearch(file_metadata, user_email)
            if success:
                print(f"✅ Metadata also stored in OpenSearch for user {user_email}")
            else:
                print(f"⚠️ Failed to store metadata in OpenSearch for user {user_email}")
        except Exception as e:
            print(f"⚠️ Error storing metadata in OpenSearch: {str(e)}")
    
    def load_metadata(self, user_email: str) -> Dict:
        """
        Load comprehensive metadata about processed files.
        
        Args:
            user_email (str): User's email to load metadata for
            
        Returns:
            dict: Loaded metadata or empty dict if not found
        """
        print(f"🔄 [METADATA] Loading metadata for user: {user_email}")
        metadata_path = self.get_metadata_path(user_email)
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Failed to load enhanced metadata: {e}")
                return {}
        print(f"⚠️ Enhanced metadata not found for user: {user_email}")
        return {}
    
    def delete_file(self, file_name: str, user_email: str) -> bool:
        """
        Delete a file entry from the enhanced metadata JSON file.
        
        Args:
            file_name (str): Name of file to delete
            user_email (str): User's email for data access
            
        Returns:
            bool: True if file deleted successfully
        """
        print(f"🗑️ [METADATA] Deleting file from enhanced metadata for user: {user_email}")
        try:
            metadata_path = self.get_metadata_path(user_email)
            if not os.path.exists(metadata_path):
                print(f"⚠️ [METADATA] Metadata file does not exist: {metadata_path}")
                return False
            
            # Load existing metadata
            with open(metadata_path, 'r') as f:
                enhanced_metadata = json.load(f)
            
            # Find and delete the file entry
            file_hash_to_delete = None
            for file_hash, metadata in enhanced_metadata.items():
                if metadata.get('file_name') == file_name:
                    file_hash_to_delete = file_hash
                    break
            
            if file_hash_to_delete:
                del enhanced_metadata[file_hash_to_delete]
                
                # Save updated metadata
                with open(metadata_path, 'w') as f:
                    json.dump(enhanced_metadata, f, indent=2)
                
                print(f"✅ [METADATA] Deleted {file_name} from enhanced metadata")
                return True
            else:
                print(f"⚠️ [METADATA] File {file_name} not found in enhanced metadata")
                return False
                
        except Exception as e:
            print(f"❌ [METADATA] Error deleting file from enhanced metadata: {str(e)}")
            return False
    
    def check_file_modification(self, file_info: Dict, user_email: str) -> Tuple[bool, Optional[Dict]]:
        """
        Check if a file has been modified since last processing.
        
        Args:
            file_info (dict): Current file information from Google Drive
            user_email (str): User's email
            
        Returns:
            tuple: (is_modified, existing_metadata) where is_modified is bool and existing_metadata is dict or None
        """
        print(f"🔍 [METADATA] Checking modification for file: {file_info.get('name', 'unknown')} for user: {user_email}")
        try:
            
            file_id = file_info['id']
            file_name = file_info['name']
            current_modified_time = file_info.get('modifiedTime', '')
            
            # Load enhanced metadata
            enhanced_metadata = self.load_metadata(user_email)
            
            # Find existing metadata for this file by file_id
            existing_metadata = None
            existing_file_hash = None
            
            for file_hash, metadata in enhanced_metadata.items():
                if metadata.get('file_id') == file_id:
                    existing_metadata = metadata
                    existing_file_hash = file_hash
                    break
            
            if existing_metadata:
                stored_modified_time = existing_metadata.get('modified_time', '')
                
                # Compare modification times
                if current_modified_time != stored_modified_time:
                    print(f"🔄 [METADATA] File '{file_name}' has been modified:")
                    print(f"   Old modified time: {stored_modified_time}")
                    print(f"   New modified time: {current_modified_time}")
                    return True, existing_metadata
                else:
                    print(f"✅ [METADATA] File '{file_name}' is up to date")
                    return False, existing_metadata
            else:
                print(f"🆕 [METADATA] File '{file_name}' is new (not previously processed)")
                return False, None
                
        except Exception as e:
            print(f"❌ [METADATA] Error checking file modification: {str(e)}")
            return False, None
    
    def cleanup_modified_file(self, file_name: str, user_email: str, file_id: str = None) -> bool:
        """
        Clean up a modified file by deleting its chunks from vector databases and metadata.
        
        Args:
            file_name (str): Name of the file to cleanup
            user_email (str): User's email
            file_id (str, optional): File ID for more precise metadata cleanup
            
        Returns:
            bool: True if cleanup was successful
        """
        print(f"🧹 [METADATA] Cleaning up modified file: {file_name} for user: {user_email}")
        try:
            print(f"🧹 [METADATA] Cleaning up modified file: {file_name}")
            
            from config.constants import OPENSEARCH_INDEX, SHEETS_INDEX
            
            query = {
                "bool": {
                    "must": [
                        {"match": {"metadata.source": file_name}}
                    ]
                }
            }

            # Delete chunks from PDF/Docs index
            pdf_cleanup_success = self.delete_from_opensearch(OPENSEARCH_INDEX, query, operation_name=f"chunks for file '{file_name}'")
            
            # Delete chunks from sheets index
            sheets_cleanup_success = self.delete_from_opensearch(SHEETS_INDEX, query, operation_name=f"chunks for file '{file_name}'")
            
            # Delete from enhanced metadata JSON file
            enhanced_metadata_cleanup_success = self.delete_file_from_enhanced_metadata(file_name, user_email)
            
            # Delete from OpenSearch metadata index
            metadata_index_cleanup_success = False
            if file_id:
                # Use file_id for more precise cleanup
                metadata_index_cleanup_success = self.delete_metadata_from_opensearch_by_file_id(file_id, user_email)
            else:
                # Fallback to filename-based cleanup
                metadata_index_cleanup_success = self.delete_metadata_from_opensearch_by_filename(file_name, user_email)
            
            if pdf_cleanup_success or sheets_cleanup_success or enhanced_metadata_cleanup_success or metadata_index_cleanup_success:
                print(f"✅ [METADATA] Cleanup completed for {file_name}")
                return True
            else:
                print(f"⚠️ [METADATA] No cleanup actions were performed for {file_name}")
                return False
                
        except Exception as e:
            print(f"❌ [METADATA] Error during file cleanup: {str(e)}")
            return False
    
    def extract_metadata(self, file_info: Dict, file_type: str, additional_info: Dict = None) -> Dict:
        """
        Extract comprehensive metadata from a file.
        
        Args:
            file_info (dict): Basic file information from Google Drive
            file_type (str): Type of file (pdf, google_doc, sheet)
            additional_info (dict): Additional processing information
            
        Returns:
            dict: Comprehensive file metadata
        """
        print(f"📄 [METADATA] Extracting metadata for file: {file_info.get('name', 'unknown')} for user: {file_info.get('owners', ['unknown'])[0]['emailAddress']}")
        metadata = {
            "file_id": file_info['id'],
            "file_name": file_info['name'],
            "file_type": file_type,
            "mime_type": file_info.get('mimeType', ''),
            "size": file_info.get('size', '0'),
            "created_time": file_info.get('createdTime', ''),
            "modified_time": file_info.get('modifiedTime', ''),
            "last_opened_time": file_info.get('viewedByMeTime', ''),
            "processed_time": datetime.now().isoformat(),
            "owners": file_info.get('owners', []),
            "permissions": file_info.get('permissions', []),
            "web_view_link": file_info.get('webViewLink', ''),
            "web_content_link": file_info.get('webContentLink', ''),
            "thumbnail_link": file_info.get('thumbnailLink', ''),
            "parents": file_info.get('parents', []),
            "trashed": file_info.get('trashed', False),
            "starred": file_info.get('starred', False),
            "shared": file_info.get('shared', False),
            "viewed_by_me_time": file_info.get('viewedByMeTime', ''),
            "viewed_by_me": file_info.get('viewedByMe', False),
            "capabilities": file_info.get('capabilities', {}),
            "export_links": file_info.get('exportLinks', {}),
            "app_properties": file_info.get('appProperties', {}),
            "properties": file_info.get('properties', {}),
            "image_media_metadata": file_info.get('imageMediaMetadata', {}),
            "video_media_metadata": file_info.get('videoMediaMetadata', {}),
            "processing_status": "completed",
            "processing_errors": [],
            # Add revision information fields
            "revisions": [],
            "revision_count": 0
        }
        
        # Add file-type specific metadata
        if file_type == 'pdf':
            metadata.update({
                "page_count": additional_info.get('page_count', 0),
                "text_chunks": additional_info.get('text_chunks', 0),
                "image_chunks": additional_info.get('image_chunks', 0),
                "total_chunks": additional_info.get('total_chunks', 0),
                "extracted_links": additional_info.get('extracted_links', []),
                "has_images": additional_info.get('has_images', False),
                "file_size_mb": round(int(metadata['size']) / (1024 * 1024), 2) if metadata['size'] != '0' else 0
            })
        elif file_type == 'google_doc':
            metadata.update({
                "word_count": additional_info.get('word_count', 0),
                "character_count": additional_info.get('character_count', 0),
                "text_chunks": additional_info.get('text_chunks', 0),
                "total_chunks": additional_info.get('text_chunks', 0),
                "file_size_mb": round(int(metadata['size']) / (1024 * 1024), 2) if metadata['size'] != '0' else 0
            })
        elif file_type == 'sheet':
            metadata.update({
                "sheet_count": additional_info.get('sheet_count', 0),
                "row_count": additional_info.get('row_count', 0),
                "column_count": additional_info.get('column_count', 0),
                "text_chunks": additional_info.get('text_chunks', 0),
                "total_chunks": additional_info.get('text_chunks', 0),
                "file_size_mb": round(int(metadata['size']) / (1024 * 1024), 2) if metadata['size'] != '0' else 0
            })
        
        return metadata

    def delete_file_from_enhanced_metadata(self, file_name: str, user_email: str) -> bool:
        """
        Delete a file entry from the enhanced metadata JSON file.
        
        Args:
            file_name (str): Name of file to delete
            user_email (str): User's email for data access
            
        Returns:
            bool: True if file deleted successfully
        """
        print(f"🗑️ [METADATA] Deleting file from enhanced metadata for user: {user_email}")
        try:
            print(f"🗑️ [METADATA] Deleting file from enhanced metadata: {file_name}")
            
            metadata_path = self.get_metadata_path(user_email)
            if not os.path.exists(metadata_path):
                print(f"⚠️ [METADATA] Metadata file does not exist: {metadata_path}")
                return False
            
            # Load existing metadata
            with open(metadata_path, 'r') as f:
                enhanced_metadata = json.load(f)
            
            # Find and delete the file entry
            file_hash_to_delete = None
            for file_hash, metadata in enhanced_metadata.items():
                if metadata.get('file_name') == file_name:
                    file_hash_to_delete = file_hash
                    break
            
            if file_hash_to_delete:
                del enhanced_metadata[file_hash_to_delete]
                
                # Save updated metadata
                with open(metadata_path, 'w') as f:
                    json.dump(enhanced_metadata, f, indent=2)
                
                print(f"✅ [METADATA] Deleted {file_name} from enhanced metadata")
                return True
            else:
                print(f"⚠️ [METADATA] File {file_name} not found in enhanced metadata")
                return False
                
        except Exception as e:
            print(f"❌ [METADATA] Error deleting file from enhanced metadata: {str(e)}")
            return False

    def delete_from_opensearch(self, index_name: str, query: Dict, user_email: str = None, operation_name: str = "documents") -> bool:
        """
        Generic function to delete documents from OpenSearch based on query.
        
        Args:
            index_name (str): OpenSearch index name
            query (dict): Query to match documents for deletion
            user_email (str, optional): User's email for data isolation
            operation_name (str): Name for logging purposes
            
        Returns:
            bool: True if deletion was successful
        """
        print(f"🗑️ [METADATA] Deleting {operation_name} from {index_name} for user: {user_email}")
        try:
            
            # Determine which client to use
            if index_name == "file_metadata":
                # Use spaCy metadata handler client
                from core.retrieval.query_classifier import get_spacy_metadata_handler
                spacy_handler = get_spacy_metadata_handler()
                if not spacy_handler or not spacy_handler.opensearch_client:
                    print("❌ [METADATA] spaCy metadata handler or OpenSearch client not available")
                    return False
                client = spacy_handler.opensearch_client
            else:
                # Use regular OpenSearch client
                from core.database.opensearch import OpenSearchClient
                opensearch_client = OpenSearchClient()
                if not opensearch_client.client:
                    print("❌ [METADATA] Failed to create OpenSearch client")
                    return False
                client = opensearch_client.client
            
            # Check if index exists
            if not client.indices.exists(index=index_name):
                print(f"❌ [METADATA] Index {index_name} does not exist")
                return False
            
            # Execute deletion
            response = client.delete_by_query(
                index=index_name,
                body={"query": query}
            )
            
            deleted_count = response.get('deleted', 0)
            print(f"✅ [METADATA] Deleted {deleted_count} {operation_name} from {index_name}")
            return True
            
        except Exception as e:
            print(f"❌ [METADATA] Error deleting {operation_name} from OpenSearch: {str(e)}")
            return False

    def delete_metadata_from_opensearch_by_filename(self, file_name: str, user_email: str) -> bool:
        """
        Delete metadata from OpenSearch metadata index for a specific file.
        
        Args:
            file_name (str): Name of file to delete metadata for
            user_email (str): User's email for data access
            
        Returns:
            bool: True if metadata deleted successfully
        """
        print(f"🗑️ [METADATA] Deleting metadata for file: {file_name} for user: {user_email}")
        query = {
            "bool": {
                "must": [
                    {"term": {"user_email": user_email}},
                    {"match": {"file_name": file_name}}
                ]
            }
        }
        return self.delete_from_opensearch("file_metadata", query, user_email, f"metadata entries for file '{file_name}'")

    def delete_metadata_from_opensearch_by_file_id(self, file_id: str, user_email: str) -> bool:
        """
        Delete metadata from OpenSearch metadata index for a specific file by file_id.
        
        Args:
            file_id (str): Google Drive file ID
            user_email (str): User's email for data access
            
        Returns:
            bool: True if metadata deleted successfully
        """
        print(f"🗑️ [METADATA] Deleting metadata for file_id: {file_id} for user: {user_email}")
        query = {
            "bool": {
                "must": [
                    {"term": {"user_email": user_email}},
                    {"term": {"file_id": file_id}}
                ]
            }
        }
        return self.delete_from_opensearch("file_metadata", query, user_email, f"metadata entries for file_id '{file_id}'")
    
    def cleanup_deleted_files_from_metadata(self, drive_files: list, user_email: str) -> int:
        """
        Clean up files that exist in metadata but are no longer present in Google Drive.
        
        Args:
            drive_files (list): List of file info dictionaries from Google Drive
            user_email (str): User's email
            
        Returns:
            int: Number of files cleaned up
        """
        print(f"🔍 [METADATA] Checking for files that have been deleted from Google Drive for user: {user_email}")
        try:
            
            # Load enhanced metadata
            enhanced_metadata = self.load_metadata(user_email)
            if not enhanced_metadata:
                print("✅ [METADATA] No metadata found, nothing to clean up")
                return 0
            
            # Get all file IDs from Google Drive
            drive_file_ids = {file['id'] for file in drive_files}
            
            # Find files in metadata that are no longer in Drive
            files_to_cleanup = []
            for file_hash, metadata in enhanced_metadata.items():
                file_id = metadata.get('file_id')
                file_name = metadata.get('file_name')
                
                if file_id and file_id not in drive_file_ids:
                    files_to_cleanup.append({
                        'file_id': file_id,
                        'file_name': file_name,
                        'file_hash': file_hash
                    })
            
            if not files_to_cleanup:
                print("✅ [METADATA] All files in metadata are still present in Google Drive")
                return 0
            
            print(f"🗑️ [METADATA] Found {len(files_to_cleanup)} files that have been deleted from Google Drive")
            
            # Clean up each deleted file
            cleaned_count = 0
            for file_info in files_to_cleanup:
                file_id = file_info['file_id']
                file_name = file_info['file_name']
                file_hash = file_info['file_hash']
                
                print(f"🧹 [METADATA] Cleaning up deleted file: {file_name} (ID: {file_id}) for user: {user_email}")

                query = {
                    "bool": {
                        "must": [
                            {"match": {"metadata.source": file_name}}
                        ]
                    }
                }
                
                from config.constants import OPENSEARCH_INDEX, SHEETS_INDEX
                
                # Delete chunks from PDF/Docs index
                pdf_cleanup_success = self.delete_from_opensearch(OPENSEARCH_INDEX, query, operation_name=f"chunks for file '{file_name}'")
                
                # Delete chunks from sheets index
                sheets_cleanup_success = self.delete_from_opensearch(SHEETS_INDEX, query, operation_name=f"chunks for file '{file_name}'")
                
                # Delete from enhanced metadata JSON file
                enhanced_metadata_cleanup_success = self.delete_file_from_enhanced_metadata(file_name, user_email)
                
                # Delete from OpenSearch metadata index
                metadata_index_cleanup_success = self.delete_metadata_from_opensearch_by_file_id(file_id, user_email)
                
                if pdf_cleanup_success or sheets_cleanup_success or enhanced_metadata_cleanup_success or metadata_index_cleanup_success:
                    print(f"✅ [METADATA] Cleanup completed for deleted file: {file_name}")
                    cleaned_count += 1
                else:
                    print(f"⚠️ [METADATA] No cleanup actions were performed for deleted file: {file_name}")
            
            print(f"✅ [METADATA] Cleaned up {cleaned_count} deleted files from all indexes and metadata")
            return cleaned_count
            
        except Exception as e:
            print(f"❌ [METADATA] Error cleaning up deleted files: {str(e)}")
            return 0
    
    def process_revisions(self, service, file_info: Dict, user_email: str):
        """
        Process file revisions for enhanced metadata.
        
        Args:
            service: Google Drive service instance
            file_info (dict): File information
            user_email (str): User's email
        """
        print(f"📝 [METADATA] Processing revisions for file: {file_info.get('name', 'unknown')} for user: {user_email}")
        try:
            file_id = file_info['id']
            file_name = file_info['name']
            
            # Get revisions for the file
            revisions = service.revisions().list(fileId=file_id).execute()
            
            if revisions.get('revisions'):
                print(f"📝 Found {len(revisions['revisions'])} revisions for {file_name}")
                
                # Store revision information in metadata
                revision_data = []
                for revision in revisions['revisions']:
                    revision_info = {
                        'revision_id': revision['id'],
                        'revision_modified_time': revision.get('modifiedTime', ''),
                        'revision_size': revision.get('size', '0'),
                        'revision_keep_forever': revision.get('keepForever', False),
                        'revision_original_filename': revision.get('originalFilename', ''),
                        'revision_mime_type': revision.get('mimeType', ''),
                        'last_modifying_user': revision.get('lastModifyingUser', {})
                    }
                    revision_data.append(revision_info)
                
                # Update metadata with revision information
                file_hash = hashlib.md5(f"{file_id}_{file_info.get('modifiedTime', '')}".encode()).hexdigest()
                enhanced_metadata = self.load_metadata(user_email)
                
                if file_hash in enhanced_metadata:
                    enhanced_metadata[file_hash]['revisions'] = revision_data
                    enhanced_metadata[file_hash]['revision_count'] = len(revision_data)
                    self.save_metadata({file_hash: enhanced_metadata[file_hash]}, user_email)
                    
        except Exception as e:
            print(f"⚠️ Error processing revisions for {file_name}: {str(e)}") 