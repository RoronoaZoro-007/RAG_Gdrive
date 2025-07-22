"""
Document processing for PDFs and Google Docs.
Handles text extraction, image processing, and chunking.
"""

import tempfile
import hashlib
import re
import fitz  # PyMuPDF
import streamlit as st
import os
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from config.settings import settings
from config.constants import OPENSEARCH_INDEX, SHEETS_INDEX
from core.database.metadata import MetadataManager
from core.database.opensearch import OpenSearchClient
from core.processing.image_processor import ImageProcessor

from core.drive.downloader import FileDownloader
from core.drive.scanner import FileScanner
from core.processing.sheet_processor import SheetProcessor
from datetime import datetime

class DocumentProcessor:
    """Handles processing of PDF and Google Doc documents."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.metadata_manager = MetadataManager()
        self.image_processor = ImageProcessor()
        self.file_downloader = FileDownloader()
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=settings.GOOGLE_API_KEY
        )
    
    def extract_text_with_links(self, pdf_path: str) -> list:
        """
        Extract text and hyperlinks from a PDF.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            list: List of dictionaries with page text and links
        """
        try:
            doc = fitz.open(pdf_path)
            extracted_data = []
            
            print(f"📖 PDF has {len(doc)} pages")

            for page_num, page in enumerate(doc):
                page_text = page.get_text("text")
                links = []

                # Extract clickable links
                for link in page.get_links():
                    if "uri" in link:
                        links.append(link["uri"])
                        page_text += f"\n[🔗 Link: {link['uri']}]"

                # Extract plain text URLs using regex
                url_pattern = r"https?://\S+"  
                text_links = re.findall(url_pattern, page_text)
                links.extend(text_links)

                # Remove duplicates
                links = list(set(links))
                
                # Debug: Show text length for each page
                text_length = len(page_text.strip())
                print(f"📄 Page {page_num + 1}: {text_length} characters, {len(links)} links")

                extracted_data.append({
                    "page": page_num + 1,
                    "text": page_text,
                    "links": links
                })
            
            doc.close()
            return extracted_data
            
        except Exception as e:
            print(f"❌ Error extracting text from PDF: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def process_pdf_with_images(self, pdf_path: str, file_name: str, file_id: str, file_info: dict = None, user_email: str = None) -> list:
        """
        Process PDF with text extraction and image analysis.
        
        Args:
            pdf_path (str): Path to the PDF file
            file_name (str): Name of the PDF file
            file_id (str): Google Drive file ID
            file_info (dict, optional): File metadata for enhanced storage
            user_email (str, optional): User's email for metadata storage
            
        Returns:
            list: List of Document objects or None if failed
        """
        try:
            # Extract text and links from PDF
            extracted_pages = self.extract_text_with_links(pdf_path)
            
            if not extracted_pages:
                print(f"⚠️ No text extracted from {file_name}")
                return None
            
            # Process images in PDF
            image_processor = ImageProcessor()
            image_descriptions = image_processor.process_pdf_images(pdf_path)
            
            # Create documents for text content
            documents = []
            text_doc_count = 0
            
            for page_data in extracted_pages:
                text = page_data["text"]
                links = page_data["links"]
                page_num = page_data["page"] - 1  # Convert to 0-based index
                
                if text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={
                            'source': file_name,
                            'type': 'text',
                            'file_id': file_id,
                            'file_type': 'pdf',
                            'page_number': page_data["page"],
                            'total_pages': len(extracted_pages),
                            'processed_time': datetime.now().isoformat(),
                            'extracted_links': links
                        }
                    )
                    documents.append(doc)
                    text_doc_count += 1
            
            # Create documents for image descriptions
            image_doc_count = 0
            for img_desc in image_descriptions:
                doc = Document(
                    page_content=img_desc,
                    metadata={
                        'source': file_name,
                        'type': 'image_description',
                        'file_id': file_id,
                        'file_type': 'pdf',
                        'processed_time': datetime.now().isoformat()
                    }
                )
                documents.append(doc)
                image_doc_count += 1
            
            # Collect all links from all pages
            all_links = []
            for page_data in extracted_pages:
                all_links.extend(page_data["links"])
            
            # Store enhanced metadata if file_info is provided
            if file_info and user_email:
                additional_info = {
                    "page_count": len(extracted_pages),
                    "text_chunks": text_doc_count,
                    "image_chunks": image_doc_count,
                    "total_chunks": len(documents),
                    "extracted_links": list(set(all_links)),  # Remove duplicates
                    "has_images": image_doc_count > 0
                }
                
                # Store enhanced metadata
                file_hash = hashlib.md5(f"{file_id}_{file_info.get('modifiedTime', '')}".encode()).hexdigest()
                enhanced_metadata = self.metadata_manager.extract_metadata(file_info, 'pdf', additional_info)
                
                # Save enhanced metadata with correct user_email
                self.metadata_manager.save_metadata({file_hash: enhanced_metadata}, user_email)
            
            return documents
            
        except Exception as e:
            print(f"❌ Error processing PDF {file_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def download_google_doc_content(self, service, file_id: str, file_name: str, file_info: dict = None, user_email: str = None) -> Document:
        """
        Download Google Doc content as text.
        
        Args:
            service: Google Drive service instance
            file_id (str): Google Drive file ID
            file_name (str): Name of the Google Doc
            file_info (dict, optional): File metadata for enhanced storage
            user_email (str, optional): User's email for metadata storage
            
        Returns:
            Document: LangChain document object or None if failed
        """
        try:
            # Download content to memory
            request = service.files().export_media(
                fileId=file_id,
                mimeType='text/plain'
            )
            
            import io
            from googleapiclient.http import MediaIoBaseDownload
            
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            
            while done is False:
                status, done = downloader.next_chunk()
                if status:
                    print(f"Download {int(status.progress() * 100)}%")
            
            content = fh.getvalue().decode('utf-8')
            
            # Create a document object similar to PDF processing
            doc = Document(
                page_content=content,
                metadata={
                    'source': file_name,
                    'type': 'text',
                    'file_id': file_id,
                    'file_type': 'google_doc',
                    'processed_time': datetime.now().isoformat()
                }
            )
            
            # Extract comprehensive metadata for storage
            if file_info and user_email:
                # Count words and characters
                word_count = len(content.split())
                character_count = len(content)
                
                additional_info = {
                    "word_count": word_count,
                    "character_count": character_count,
                    "text_chunks": 1,  # Google Docs are typically single chunks
                    "total_chunks": 1
                }
                
                # Store enhanced metadata
                file_hash = hashlib.md5(f"{file_id}_{file_info.get('modifiedTime', '')}".encode()).hexdigest()
                enhanced_metadata = self.metadata_manager.extract_metadata(file_info, 'google_doc', additional_info)
                
                # Save enhanced metadata with correct user_email
                self.metadata_manager.save_metadata({file_hash: enhanced_metadata}, user_email)
            
            return doc
            
        except Exception as e:
            print(f"Error downloading Google Doc {file_name}: {str(e)}")
            return None
    
    def process_all_documents(self, credentials, user_email: str, pdf_files: list, doc_files: list, sheet_files: list) -> bool:
        """
        Process all PDF, Google Doc, and Sheet files using the unified approach.
        
        Args:
            credentials: Google OAuth credentials
            user_email (str): User's email for data isolation
            pdf_files (list): List of PDF files
            doc_files (list): List of Google Doc files
            sheet_files (list): List of Sheet files
            
        Returns:
            bool: True if processing was successful
        """
        try:
            print(f"🔍 [DEBUG] process_all_documents called with user_email: {user_email}")
            print(f"🔍 [DEBUG] PDF files: {len(pdf_files)}, Doc files: {len(doc_files)}, Sheet files: {len(sheet_files)}")
            
            # Build Drive service
            from googleapiclient.discovery import build
            service = build('drive', 'v3', credentials=credentials)
            
            # Process files
            documents = []
            processed_count = 0
            failed_count = 0
            skipped_count = 0
            modified_count = 0
            
            print(f"🔍 Processing {len(pdf_files)} PDF files, {len(doc_files)} Google Doc files, and {len(sheet_files)} Sheet files")
            
            # Process PDFs
            for pdf_file in pdf_files:
                file_id = pdf_file['id']
                file_name = pdf_file['name']
                
                # Check if file has been modified
                is_modified, existing_metadata = self.metadata_manager.check_file_modification(pdf_file, user_email)
                
                if is_modified:
                    print(f"🔄 File '{file_name}' has been modified, cleaning up old data...")
                    self.metadata_manager.cleanup_modified_file(file_name, user_email, file_id)
                    modified_count += 1
                elif existing_metadata:
                    print(f"⏭️ Skipping already processed PDF: {file_name}")
                    skipped_count += 1
                    continue
                
                try:
                    # Download PDF
                    with tempfile.TemporaryDirectory() as temp_dir:
                        pdf_path = self.file_downloader.download_pdf(service, file_id, file_name, temp_dir)
                        if not pdf_path:
                            failed_count += 1
                            continue
                        
                        # Extract text and images with enhanced metadata
                        pdf_docs = self.process_pdf_with_images(pdf_path, file_name, file_id, pdf_file, user_email)
                        if pdf_docs:
                            documents.extend(pdf_docs)
                            processed_count += 1
                            print(f"✅ Successfully processed {file_name}: {len(pdf_docs)} documents")
                        else:
                            failed_count += 1
                            
                except Exception as e:
                    print(f"Error processing PDF {file_name}: {str(e)}")
                    failed_count += 1
            
            # Process Google Docs
            for doc_file in doc_files:
                file_id = doc_file['id']
                file_name = doc_file['name']
                
                # Check if file has been modified
                is_modified, existing_metadata = self.metadata_manager.check_file_modification(doc_file, user_email)
                
                if is_modified:
                    print(f"🔄 File '{file_name}' has been modified, cleaning up old data...")
                    self.metadata_manager.cleanup_modified_file(file_name, user_email, file_id)
                    modified_count += 1
                elif existing_metadata:
                    print(f"⏭️ Skipping already processed Google Doc: {file_name}")
                    skipped_count += 1
                    continue
                
                try:
                    # Download Google Doc content with enhanced metadata
                    doc = self.download_google_doc_content(service, file_id, file_name, doc_file, user_email)
                    if doc:
                        documents.append(doc)
                        processed_count += 1
                        print(f"✅ Successfully processed {file_name}: 1 document")
                        # Process revisions (metadata only)
                        self.metadata_manager.process_revisions(service, doc_file, user_email)
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    print(f"Error processing Google Doc {file_name}: {str(e)}")
                    failed_count += 1
            
            # Process documents with LangChain (PDFs and Google Docs)
            if documents:
                print(f"✅ Successfully processed {processed_count} files ({failed_count} failed, {modified_count} modified)")
                print("🧠 Creating embeddings for documents...")
                
                # Apply semantic chunking to documents
                text_splits = []
                from core.processing.chunking import semantic_chunk_text
                
                for doc in documents:
                    print(f"Semantically chunking document: {doc.metadata.get('source', 'unknown')}")
                    chunks = semantic_chunk_text(doc.page_content, self.embeddings)
                    for i, chunk in enumerate(chunks):
                        chunk_doc = Document(
                            page_content=chunk,
                            metadata=doc.metadata.copy()
                        )
                        chunk_doc.metadata['chunk_id'] = i
                        chunk_doc.metadata['total_chunks'] = len(chunks)
                        chunk_doc.metadata['chunking_method'] = 'semantic'
                        text_splits.append(chunk_doc)
                
                if text_splits:
                    print(f"📊 Created {len(text_splits)} semantic chunks from {len(documents)} documents")
                    
                    # Create vector store for PDFs/Docs
                    pdf_vectorstore = OpenSearchVectorSearch(
                        embedding_function=self.embeddings,
                        opensearch_url=settings.OPENSEARCH_URL,
                        index_name=OPENSEARCH_INDEX,
                        vector_field="vector_field",
                        http_auth=(settings.OPENSEARCH_USERNAME, settings.OPENSEARCH_PASSWORD) if settings.OPENSEARCH_USERNAME else None,
                        use_ssl=settings.OPENSEARCH_URL.startswith('https'),
                        verify_certs=False,
                        engine="lucene"
                    )

                    # Add documents to vector store
                    print(f"Adding {len(text_splits)} documents to PDF vector store")
                    pdf_vectorstore.add_documents(text_splits)
                    
                    # Store the vector store in session state for later use
                    st.session_state.pdf_vectorstore = pdf_vectorstore
                    print("✅ PDF vector store stored in session state")
                    
                    return True
                else:
                    print("⚠️ No document chunks created")
                    return False
            else:
                print("⚠️ No documents to process")
                return False
                
        except Exception as e:
            print(f"Error processing documents: {str(e)}")
            return False

def process_all_user_documents_unified(credentials, user_email):
    """
    Process all PDF, Google Doc, and Sheet files using the unified approach.
    
    Args:
        credentials: Google OAuth credentials
        user_email (str): User's email for data isolation
    """
    try:
        print(f"🔍 [PROCESSING] process_all_user_documents_unified called with user_email: {user_email}")
        
        # Create OpenSearch client
        opensearch_client = OpenSearchClient()
        if not opensearch_client.client:
            raise Exception("Failed to create OpenSearch client")

        # Check if indices exist, if not create them
        from config.constants import OPENSEARCH_INDEX, SHEETS_INDEX
        if not opensearch_client.client.indices.exists(index=OPENSEARCH_INDEX):
            opensearch_client.create_index(OPENSEARCH_INDEX)

        if not opensearch_client.client.indices.exists(index=SHEETS_INDEX):
            opensearch_client.create_index(SHEETS_INDEX)

        # Scan for PDFs, Google Docs, and Sheets
        st.session_state.processing_status = "📁 Scanning your Google Drive for PDF, Google Doc, and Sheet files..."
        
        file_scanner = FileScanner()
        pdf_files = file_scanner.scan_files(credentials, "application/pdf", "PDFs")
        doc_files = file_scanner.scan_files(credentials, "application/vnd.google-apps.document", "Google Docs")
        mime_types = [
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.google-apps.spreadsheet"
        ]
        sheet_files = file_scanner.scan_files(credentials, mime_types, "sheets")
        
        all_files = pdf_files + doc_files + sheet_files
        st.session_state.processing_status = f"📊 Found {len(pdf_files)} PDF files, {len(doc_files)} Google Doc files, and {len(sheet_files)} Sheet files in your Drive"
        
        # Clean up files that have been deleted from Google Drive
        metadata_manager = MetadataManager()
        file_downloader = FileDownloader()
        print(f"🔍 [PROCESSING] Cleaning up deleted files for user: {user_email}")
        deleted_files_cleaned = metadata_manager.cleanup_deleted_files_from_metadata(all_files, user_email)
        
        if not all_files:
            st.warning("No PDF, Google Doc, or Sheet files found in your Google Drive.")
            
            # Check if there are existing documents in OpenSearch
            existing_doc_count = opensearch_client.get_count(OPENSEARCH_INDEX)
            if existing_doc_count > 0:
                st.info(f"📚 Found {existing_doc_count} existing documents in your knowledge base.")
                st.success("✅ You can start asking questions about your existing documents!")
                
                # Initialize unified retriever from existing documents
                from core.retrieval.unified_retriever import initialize_unified_retriever_from_existing
                if initialize_unified_retriever_from_existing():
                    st.session_state.processed_pdfs = True
                    st.session_state.processing_status = "✅ Ready to chat with existing documents!"
                else:
                    st.error("❌ Failed to initialize chat interface with existing documents.")
            else:
                st.session_state.processing_status = "📚 No documents found. Please add documents to your Google Drive."
            
            return

        # Build Drive service
        from googleapiclient.discovery import build
        service = build('drive', 'v3', credentials=credentials)
        
        # Process files
        documents = []
        all_sheet_documents = []
        processed_count = 0
        failed_count = 0
        skipped_count = 0
        modified_count = 0
        
        print(f"🔍 Processing {len(pdf_files)} PDF files, {len(doc_files)} Google Doc files, and {len(sheet_files)} Sheet files")
        
        # Process PDFs and Google Docs
        document_processor = DocumentProcessor()
        print(f"🔍 [DEBUG] About to call process_all_documents with {len(pdf_files)} PDFs, {len(doc_files)} docs, {len([])} sheets")
        success = document_processor.process_all_documents(credentials, user_email, pdf_files, doc_files, [])
        print(f"🔍 [DEBUG] process_all_documents returned: {success}")
        
        if success:
            st.session_state.processed_pdfs = True
            st.success("✅ PDFs and Google Docs processed successfully!")
        else:
            print(f"❌ [DEBUG] process_all_documents failed")
        
        # Process sheets with LlamaIndex (individual processing like in original)
        all_sheet_documents = []
        if sheet_files:
            st.info("📊 Processing sheets with LlamaIndex...")
            
            try:
                # Process each sheet individually like in the original
                for sheet_file in sheet_files:
                    file_id = sheet_file['id']
                    file_name = sheet_file['name']
                    
                    # Check if file has been modified
                    print(f"🔍 [PROCESSING] Checking sheet modification for user: {user_email}")
                    is_modified, existing_metadata = metadata_manager.check_file_modification(sheet_file, user_email)
                    
                    if is_modified:
                        print(f"🔄 File '{file_name}' has been modified, cleaning up old data...")
                        metadata_manager.cleanup_modified_file(file_name, user_email, file_id)
                        modified_count += 1
                    elif existing_metadata:
                        print(f"⏭️ Skipping already processed Sheet: {file_name}")
                        skipped_count += 1
                        continue
                    
                    try:
                        st.session_state.processing_status = f"📊 Processing Sheet: {file_name}"
                        
                        # Download and process sheet file immediately
                        with tempfile.TemporaryDirectory() as temp_dir:
                            sheet_path = file_downloader.download_sheet(service, file_id, file_name, temp_dir)
                            if sheet_path:
                                # Process with LlamaParse immediately while file exists
                                try:
                                    from llama_parse import LlamaParse
                                    from config.settings import settings
                                    parser = LlamaParse(api_key=settings.LLAMA_CLOUD_API_KEY)
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
                                            
                                            all_sheet_documents.extend(sheet_documents)
                                            processed_count += 1
                                            print(f"✅ Successfully processed {file_name}: {len(sheet_documents)} documents")
                                            
                                            # Extract enhanced metadata for sheets
                                            additional_info = {
                                                "sheet_count": 1,
                                                "row_count": 0,
                                                "column_count": 0,
                                                "text_chunks": len(sheet_documents),
                                                "total_chunks": len(sheet_documents)
                                            }
                                            
                                            # Store enhanced metadata
                                            file_hash = hashlib.md5(f"{file_id}_{sheet_file.get('modifiedTime', '')}".encode()).hexdigest()
                                            enhanced_metadata = metadata_manager.extract_metadata(sheet_file, 'sheet', additional_info)
                                            print(f"🔍 [PROCESSING] Saving sheet metadata for user: {user_email}")
                                            metadata_manager.save_metadata({file_hash: enhanced_metadata}, user_email)
                                            
                                            # Process revisions (metadata only)
                                            metadata_manager.process_revisions(service, sheet_file, user_email)
                                        else:
                                            print(f"⚠️ No documents extracted from {file_name}")
                                            failed_count += 1
                                    else:
                                        print(f"❌ Failed to initialize parser for {file_name}")
                                        failed_count += 1
                                except Exception as e:
                                    print(f"❌ Error processing {file_name} with LlamaParse: {str(e)}")
                                    failed_count += 1
                            else:
                                print(f"❌ Failed to download {file_name}")
                                failed_count += 1
                                
                    except Exception as e:
                        print(f"Error processing Sheet {file_name}: {str(e)}")
                        failed_count += 1
                
                # Create sheets index if we have documents
                if all_sheet_documents:
                    sheet_processor = SheetProcessor()
                    sheets_index = sheet_processor.create_sheets_index(all_sheet_documents)
                    if sheets_index:
                        st.session_state.processed_sheets = True
                        st.session_state.sheets_index = sheets_index  # Store in session state
                        print("✅ Sheets index stored in session state")
                        st.success("✅ Sheets index created successfully!")
                    else:
                        st.error("❌ Failed to create sheets index")
                else:
                    st.warning("⚠️ No sheet documents were processed successfully")
                    
            except Exception as e:
                st.error(f"❌ Error creating sheets index: {str(e)}")
        
        # Create unified retriever
        if st.session_state.processed_pdfs or st.session_state.processed_sheets:
            try:
                # Initialize LLM using LlamaIndex
                llm = SheetProcessor.initialize_llm()
                if not llm:
                    raise Exception("Failed to initialize LlamaIndex LLM")
                
                # Initialize embeddings using LangChain version (like original)
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                from config.settings import settings
                
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=settings.GOOGLE_API_KEY
                )
                if not embeddings:
                    raise Exception("Failed to initialize embeddings")
                
                # Get PDF vector store if PDFs were processed
                pdf_vectorstore = None
                if st.session_state.processed_pdfs:
                    # Try to get from session state first (if it was just created)
                    pdf_vectorstore = st.session_state.get('pdf_vectorstore')
                    
                    if not pdf_vectorstore:
                        # If not in session state, create a new connection to existing index
                        from langchain_community.vectorstores import OpenSearchVectorSearch
                        from config.constants import OPENSEARCH_INDEX
                        
                        pdf_vectorstore = OpenSearchVectorSearch(
                            embedding_function=embeddings,
                            opensearch_url=settings.OPENSEARCH_URL,
                            index_name=OPENSEARCH_INDEX,
                            vector_field="vector_field",
                            http_auth=(settings.OPENSEARCH_USERNAME, settings.OPENSEARCH_PASSWORD) if settings.OPENSEARCH_USERNAME else None,
                            use_ssl=settings.OPENSEARCH_URL.startswith('https'),
                            verify_certs=False,
                            engine="lucene"
                        )
                        print("✅ PDF vector store connected to existing index")
                    else:
                        print("✅ PDF vector store retrieved from session state")
                
                # Get sheets index if sheets were processed
                sheets_index = None
                if st.session_state.processed_sheets:
                    # The sheets index should already be created and stored in session state
                    sheets_index = st.session_state.get('sheets_index')
                    if sheets_index:
                        print("✅ Sheets index retrieved from session state")
                
                # Create unified retriever with proper vector stores
                from core.retrieval.unified_retriever import UnifiedRetriever
                unified_retriever = UnifiedRetriever(pdf_vectorstore, sheets_index, embeddings)
                
                # Update session state
                st.session_state.unified_retriever = unified_retriever
                st.session_state.llm = llm
                st.session_state.start_chatting = True
                
                st.success("✅ Unified retriever created successfully!")
                
            except Exception as e:
                st.error(f"❌ Error creating unified retriever: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Update session state
        status_message = "✅ Processing complete! You can now ask questions about your documents and sheets."
        if deleted_files_cleaned > 0:
            status_message += f" (Cleaned up {deleted_files_cleaned} deleted files)"
        st.session_state.processing_status = status_message
        
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        st.session_state.processing_status = f"❌ Error: {str(e)}" 