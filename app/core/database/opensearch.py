"""
OpenSearch client and operations.
Handles all OpenSearch database operations for vector storage and metadata.
"""

import streamlit as st
from opensearchpy import OpenSearch
from config.settings import settings
from config.constants import OPENSEARCH_INDEX, SHEETS_INDEX, METADATA_INDEX

class OpenSearchClient:
    """OpenSearch client for vector database operations."""
    
    def __init__(self):
        """Initialize OpenSearch client connection."""
        self.client = self._create_client()
    
    def _create_client(self) -> OpenSearch:
        """
        Create OpenSearch client connection.
        
        Returns:
            OpenSearch: Configured client or None if connection fails
        """
        try:
            # For local OpenSearch without authentication
            if not settings.OPENSEARCH_USERNAME and not settings.OPENSEARCH_PASSWORD:
                client = OpenSearch(
                    hosts=[settings.OPENSEARCH_URL],
                    use_ssl=False,
                    verify_certs=False,
                    ssl_show_warn=False
                )
            else:
                # For OpenSearch with authentication
                client = OpenSearch(
                    hosts=[settings.OPENSEARCH_URL],
                    http_auth=(settings.OPENSEARCH_USERNAME, settings.OPENSEARCH_PASSWORD),
                    use_ssl=settings.OPENSEARCH_URL.startswith('https'),
                    verify_certs=False,
                    ssl_show_warn=False
                )
            return client
        except Exception as e:
            st.error(f"Failed to connect to OpenSearch: {str(e)}")
            return None
    
    def create_index(self, index_name: str, dimension: int = 768) -> bool:
        """
        Create OpenSearch index with proper mapping for vector search.
        
        Args:
            index_name (str): Name of the index to create
            dimension (int): Vector dimension for embeddings
            
        Returns:
            bool: True if index created successfully
        """
        try:
            if not self.client:
                return False
            
            # Delete existing index if it exists
            if self.client.indices.exists(index=index_name):
                print(f"🗑️ Deleting existing index: {index_name}")
                self.client.indices.delete(index=index_name)
            
            # Define the index mapping for vector search
            index_mapping = {
                "mappings": {
                    "properties": {
                        "vector_field": {
                            "type": "knn_vector",
                            "dimension": dimension,  # Gemini embedding dimension
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "lucene"
                            }
                        },
                        "text": {
                            "type": "text"
                        },
                        "metadata": {
                            "type": "object"
                        }
                    }
                },
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": 100
                    }
                }
            }
            
            # Create the index
            self.client.indices.create(
                index=index_name,
                body=index_mapping
            )
            
            print(f"✅ Created OpenSearch index: {index_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating OpenSearch index: {str(e)}")
            return False
    
    def check_documents(self, index_name: str = OPENSEARCH_INDEX) -> bool:
        """
        Check if there are existing documents in OpenSearch.
        
        Args:
            index_name (str): Name of the index to check
            
        Returns:
            bool: True if documents exist in the index
        """
        try:
            if not self.client:
                return False
            
            # Check if index exists
            if not self.client.indices.exists(index=index_name):
                return False
            
            # Count documents in the index
            count_response = self.client.count(index=index_name)
            document_count = count_response.get('count', 0)
            
            return document_count > 0
            
        except Exception as e:
            print(f"Error checking existing documents: {str(e)}")
            return False
    
    def get_count(self, index_name: str = OPENSEARCH_INDEX) -> int:
        """
        Get the number of documents in OpenSearch.
        
        Args:
            index_name (str): Name of the index to count documents in
            
        Returns:
            int: Number of documents in the index
        """
        try:
            print(f"🔍 Checking document count in OpenSearch index: {index_name}")
            if not self.client:
                print("❌ Failed to create OpenSearch client")
                return 0
            
            # Check if index exists
            if not self.client.indices.exists(index=index_name):
                print(f"❌ Index {index_name} does not exist")
                return 0
            
            # Count documents in the index
            count_response = self.client.count(index=index_name)
            document_count = count_response.get('count', 0)
            print(f"📊 Found {document_count} documents in {index_name}")
            
            return document_count
            
        except Exception as e:
            print(f"❌ Error getting document count: {str(e)}")
            return 0
    
    def delete_by_query(self, index_name: str, query: dict, user_email: str = None, operation_name: str = "documents") -> bool:
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
        try:
            if not self.client:
                print("❌ OpenSearch client not available")
                return False
            
            # Check if index exists
            if not self.client.indices.exists(index=index_name):
                print(f"❌ Index {index_name} does not exist")
                return False
            
            # Execute deletion
            response = self.client.delete_by_query(
                index=index_name,
                body={"query": query}
            )
            
            deleted_count = response.get('deleted', 0)
            print(f"✅ Deleted {deleted_count} {operation_name} from {index_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting {operation_name} from OpenSearch: {str(e)}")
            return False
    
    def delete_metadata_by_filename(self, file_name: str, user_email: str) -> bool:
        """
        Delete metadata from OpenSearch metadata index for a specific file.
        
        Args:
            file_name (str): Name of file to delete metadata for
            user_email (str): User's email for data access
            
        Returns:
            bool: True if metadata deleted successfully
        """
        query = {
            "bool": {
                "must": [
                    {"term": {"user_email": user_email}},
                    {"match": {"file_name": file_name}}
                ]
            }
        }
        return self.delete_by_query(METADATA_INDEX, query, user_email, f"metadata entries for file '{file_name}'")
    
    def delete_metadata_by_file_id(self, file_id: str, user_email: str) -> bool:
        """
        Delete metadata from OpenSearch metadata index for a specific file by file_id.
        
        Args:
            file_id (str): Google Drive file ID
            user_email (str): User's email for data access
            
        Returns:
            bool: True if metadata deleted successfully
        """
        query = {
            "bool": {
                "must": [
                    {"term": {"user_email": user_email}},
                    {"term": {"file_id": file_id}}
                ]
            }
        }
        return self.delete_by_query(METADATA_INDEX, query, user_email, f"metadata entries for file_id '{file_id}'") 