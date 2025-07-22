"""
Sheet processing using LlamaIndex.
Handles Google Sheets and Excel file processing.
"""

import os
import hashlib
import tempfile
import traceback
from datetime import datetime
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings, VectorStoreIndex, StorageContext, PromptTemplate
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.opensearch import OpensearchVectorStore, OpensearchVectorClient
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_parse import LlamaParse
from config.settings import settings
from config.constants import SHEETS_INDEX, LLAMA_MODEL
from core.database.metadata import MetadataManager
from core.drive.downloader import FileDownloader
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class SheetProcessor:
    """Handles processing of Google Sheets and Excel files using LlamaIndex."""
    
    def __init__(self):
        """Initialize the sheet processor."""
        self.metadata_manager = MetadataManager()
        self.file_downloader = FileDownloader()
        
        # Thread-local storage for components
        self._local = threading.local()
    
    def _get_parser(self):
        """Get thread-local parser instance."""
        if not hasattr(self._local, 'parser'):
            self._local.parser = self.initialize_parser()
        return self._local.parser
    
    def _get_llm(self):
        """Get thread-local LLM instance."""
        if not hasattr(self._local, 'llm'):
            self._local.llm = self.initialize_llm()
        return self._local.llm
    
    def _get_embeddings(self):
        """Get thread-local embeddings instance."""
        if not hasattr(self._local, 'embeddings'):
            self._local.embeddings = self.initialize_embeddings()
        return self._local.embeddings
    
    @staticmethod
    def _get_api_key(env_var: str, component_name: str) -> str:
        """
        Get API key from environment variable.
        
        Args:
            env_var (str): Environment variable name
            component_name (str): Component name for error messages
            
        Returns:
            str: API key value
            
        Raises:
            ValueError: If API key not found
        """
        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(f"{env_var} not found in environment variables")
        return api_key
    
    @staticmethod
    def initialize_parser():
        """
        Initialize the LlamaParse parser for sheets processing.
        
        Returns:
            LlamaParse: Configured parser instance or None if failed
        """
        try:
            llama_key = SheetProcessor._get_api_key("LLAMA_CLOUD_API_KEY", "LlamaParse")
            return LlamaParse(api_key=llama_key, result_type="markdown")
        except Exception as e:
            print(f"Error initializing LlamaParse: {str(e)}")
            return None
    
    @staticmethod
    def initialize_llm():
        """
        Initialize the Google GenAI LLM for LlamaIndex.
        
        Returns:
            GoogleGenAI: Configured LLM instance or None if failed
        """
        try:
            google_api_key = SheetProcessor._get_api_key("GOOGLE_API_KEY", "LlamaIndex LLM")
            return GoogleGenAI(model=LLAMA_MODEL, api_key=google_api_key, temperature=0.3)
        except Exception as e:
            print(f"Error initializing LlamaIndex LLM: {str(e)}")
            return None
    
    @staticmethod
    def initialize_embeddings():
        """
        Initialize the Google GenAI Embeddings for LlamaIndex.
        
        Returns:
            GoogleGenAIEmbedding: Configured embeddings instance or None if failed
        """
        try:
            google_api_key = SheetProcessor._get_api_key("GOOGLE_API_KEY", "LlamaIndex embeddings")
            return GoogleGenAIEmbedding(model_name="embedding-001", api_key=google_api_key)
        except Exception as e:
            print(f"Error initializing LlamaIndex embeddings: {str(e)}")
            return None
    
    def create_sheets_index(self, all_documents: list):
        """
        Create and return the LlamaIndex index for sheets processing.
        
        Args:
            all_documents (list): List of LlamaIndex document objects
            
        Returns:
            VectorStoreIndex: Configured index for sheet queries
        """
        try:
            print("🔧 Initializing LlamaIndex components for sheets...")
            
            # Initialize components
            llm = self.initialize_llm()
            if not llm:
                raise Exception("Failed to initialize LlamaIndex LLM")
            
            embeddings = self.initialize_embeddings()
            if not embeddings:
                raise Exception("Failed to initialize LlamaIndex embeddings")
            
            Settings.llm = llm
            Settings.embed_model = embeddings
            
            print(f"📊 Processing {len(all_documents)} pre-processed sheet documents...")
            
            if not all_documents:
                raise ValueError("No documents were provided for processing")
            
            print(f"📊 Total documents to process: {len(all_documents)}")
            
            # OpenSearch configuration for sheets
            sheets_endpoint = settings.OPENSEARCH_URL
            sheets_idx = SHEETS_INDEX
            text_field = "content"
            embedding_field = "embedding"
            
            # Create OpenSearch client
            from opensearchpy import OpenSearch
            client = OpenSearch(
                hosts=[sheets_endpoint],
                http_auth=(settings.OPENSEARCH_USERNAME, settings.OPENSEARCH_PASSWORD) if settings.OPENSEARCH_USERNAME else None,
                use_ssl=sheets_endpoint.startswith('https'),
                verify_certs=False,
                ssl_show_warn=False
            )
            
            # Check if sheets index exists and has correct mapping
            index_exists = client.indices.exists(index=sheets_idx)
            should_recreate_index = False
            
            if index_exists:
                try:
                    current_mapping = client.indices.get_mapping(index=sheets_idx)
                    properties = current_mapping[sheets_idx]['mappings']['properties']
                    
                    if embedding_field not in properties or properties[embedding_field]['type'] != 'knn_vector':
                        should_recreate_index = True
                except Exception as e:
                    should_recreate_index = True
            
            if should_recreate_index or not index_exists:
                # Delete existing index if it exists
                if index_exists:
                    client.indices.delete(index=sheets_idx)
                
                # Create new index with proper mapping
                index_mapping = {
                    "mappings": {
                        "properties": {
                            embedding_field: {
                                "type": "knn_vector",
                                "dimension": 768,
                                "method": {
                                    "name": "hnsw",
                                    "space_type": "cosinesimil",
                                    "engine": "lucene"
                                }
                            },
                            text_field: {"type": "text"},
                            "metadata": {"type": "object"}
                        }
                    },
                    "settings": {
                        "index": {
                            "knn": True,
                            "knn.algo_param.ef_search": 100
                        }
                    }
                }
                
                client.indices.create(index=sheets_idx, body=index_mapping)
                print(f"✅ Created/updated OpenSearch index: {sheets_idx}")
            
            # Create vector store
            from llama_index.vector_stores.opensearch import OpensearchVectorClient
            vector_client = OpensearchVectorClient(
                sheets_endpoint,
                sheets_idx,
                768,
                embedding_field=embedding_field,
                text_field=text_field
            )
            vector_store = OpensearchVectorStore(vector_client)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create index
            index = VectorStoreIndex.from_documents(
                all_documents,
                storage_context=storage_context,
                embed_model=embeddings
            )
            
            print(f"✅ Successfully created LlamaIndex for {len(all_documents)} sheet documents")
            return index
            
        except Exception as e:
            print(f"❌ Error creating sheets index: {str(e)}")
            return None
    
    def _process_single_sheet(self, sheet_file, service, user_email):
        """Process a single sheet file (for parallel processing)."""
        try:
            file_id = sheet_file['id']
            file_name = sheet_file['name']
            
            print(f"🔄 [PARALLEL] Starting sheet processing: {file_name}")
            
            # Check if file has been modified
            is_modified, existing_metadata = self.metadata_manager.check_file_modification(sheet_file, user_email)
            
            if is_modified:
                print(f"🔄 [PARALLEL] File '{file_name}' has been modified, cleaning up old data...")
                self.metadata_manager.cleanup_modified_file(file_name, user_email, file_id)
            elif existing_metadata:
                print(f"⏭️ [PARALLEL] Skipping already processed Sheet: {file_name}")
                return None, "skipped"
            
            # Download and process sheet file immediately
            with tempfile.TemporaryDirectory() as temp_dir:
                sheet_path = self.file_downloader.download_sheet(service, file_id, file_name, temp_dir)
                if not sheet_path:
                    print(f"❌ [PARALLEL] Failed to download sheet: {file_name}")
                    return None, "failed"
                
                # Process with LlamaParse immediately while file exists
                try:
                    parser = self._get_parser()
                    if parser:
                        sheet_documents = parser.load_data(sheet_path)
                        if sheet_documents:
                            # Add proper metadata to each sheet document
                            for doc in sheet_documents:
                                doc.metadata.update({
                                    'source': file_name,
                                    'file_id': file_id,
                                    'file_type': 'sheet',
                                    'processed_time': datetime.now().isoformat()
                                })
                            
                            print(f"✅ [PARALLEL] Successfully processed sheet: {file_name} ({len(sheet_documents)} documents)")
                            
                            # Extract enhanced metadata for sheets
                            additional_info = {
                                "sheet_count": 1,  # Default, could be enhanced with actual sheet count
                                "row_count": 0,  # Could be enhanced with actual row count
                                "column_count": 0,  # Could be enhanced with actual column count
                                "text_chunks": len(sheet_documents),
                                "total_chunks": len(sheet_documents)
                            }
                            
                            # Store enhanced metadata
                            file_hash = hashlib.md5(f"{file_id}_{sheet_file.get('modifiedTime', '')}".encode()).hexdigest()
                            enhanced_metadata = self.metadata_manager.extract_metadata(sheet_file, 'sheet', additional_info)
                            self.metadata_manager.save_metadata({file_hash: enhanced_metadata}, user_email)
                            
                            # Process revisions (metadata only)
                            self.metadata_manager.process_revisions(service, sheet_file, user_email)
                            
                            return sheet_documents, "success"
                        else:
                            print(f"⚠️ [PARALLEL] No documents extracted from {file_name}")
                            return None, "failed"
                    else:
                        print(f"❌ [PARALLEL] Failed to initialize parser for {file_name}")
                        return None, "failed"
                except Exception as e:
                    print(f"❌ [PARALLEL] Error processing {file_name} with LlamaParse: {str(e)}")
                    return None, "failed"
                    
        except Exception as e:
            print(f"❌ [PARALLEL] Error processing sheet {sheet_file.get('name', 'unknown')}: {str(e)}")
            return None, "failed"

    def process_sheets(self, credentials, user_email: str, sheet_files: list):
        """
        Process sheet files using LlamaIndex with parallel processing.
        
        Args:
            credentials: Google OAuth credentials
            user_email (str): User's email for data isolation
            sheet_files (list): List of sheet files to process
            
        Returns:
            VectorStoreIndex: Created index or None if failed
        """
        try:
            from googleapiclient.discovery import build
            service = build('drive', 'v3', credentials=credentials)
            
            all_sheet_documents = []
            processed_count = 0
            failed_count = 0
            skipped_count = 0
            modified_count = 0
            
            if not sheet_files:
                print("📊 No sheet files to process")
                return None
            
            print(f"🚀 [PARALLEL] Starting parallel sheet processing with {len(sheet_files)} files...")
            
            # Parallel processing of sheets
            with ThreadPoolExecutor(max_workers=min(4, len(sheet_files))) as executor:
                # Submit all sheet processing tasks
                sheet_futures = {
                    executor.submit(self._process_single_sheet, sheet_file, service, user_email): sheet_file 
                    for sheet_file in sheet_files
                }
                
                # Collect results as they complete
                for future in as_completed(sheet_futures):
                    sheet_file = sheet_futures[future]
                    try:
                        result, status = future.result()
                        if status == "success":
                            all_sheet_documents.extend(result)
                            processed_count += 1
                        elif status == "skipped":
                            skipped_count += 1
                        else:
                            failed_count += 1
                    except Exception as e:
                        print(f"❌ [PARALLEL] Exception in sheet processing: {str(e)}")
                        failed_count += 1
            
            print(f"✅ [PARALLEL] Sheet processing completed: {processed_count} processed, {failed_count} failed, {skipped_count} skipped")
            
            # Create sheets index
            if all_sheet_documents:
                print(f"✅ Successfully processed {processed_count} sheet files ({failed_count} failed, {modified_count} modified)")
                print("📊 Creating LlamaIndex for sheets...")
                
                sheets_index = self.create_sheets_index(all_sheet_documents)
                if sheets_index:
                    print("✅ Sheets index created successfully!")
                    return sheets_index
                else:
                    print("❌ Failed to create sheets index")
                    return None
            else:
                print("⚠️ No sheet documents to process")
                return None
                
        except Exception as e:
            print(f"Error processing sheets: {str(e)}")
            return None 