"""
Main Streamlit UI for the SheetsLoader application.
Contains all UI components and the main application logic.
"""

import streamlit as st
import tempfile
import os
from config.settings import settings
from config.constants import STATUS_MESSAGES, OPENSEARCH_INDEX, SHEETS_INDEX
from core.auth.session import SessionManager
from core.auth.google_auth import GoogleAuthManager
from core.database.opensearch import OpenSearchClient
from core.database.metadata import MetadataManager
from core.processing.document_processor import DocumentProcessor
from core.processing.sheet_processor import SheetProcessor
from core.processing.image_processor import ImageProcessor
from core.processing.chunking import SemanticChunker
from core.retrieval.unified_retriever import UnifiedRetriever
from core.retrieval.query_classifier import QueryClassifier, get_spacy_metadata_handler
from core.retrieval.response_generator import ResponseGenerator
from core.drive.client import DriveClient
from core.drive.scanner import FileScanner
from core.drive.downloader import FileDownloader

def main():
    st.set_page_config(
        page_title="Unified AI Document Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize session state
    SessionManager.initialize_session()
    
    st.title("🤖 Unified AI Document Assistant")
    st.markdown("Connect your Google Drive to process PDFs, Google Docs, and Sheets with unified AI analysis")
    
    # Check if user is authenticated
    if not st.session_state.authenticated:
        st.markdown("### 🔐 Authentication Required")
        st.markdown("Please authenticate with Google to access your Drive documents.")
        
        # Google OAuth button
        if st.button("🔑 Connect Google Drive", type="primary"):
            try:
                auth_manager = GoogleAuthManager()
                auth_url = auth_manager.get_google_oauth_url()
                if auth_url:
                    st.markdown(f"**Click the link below to authenticate:**")
                    st.markdown(f"[🔗 Authenticate with Google]({auth_url})")
                    st.info("After authentication, you'll be redirected back to this app.")
                else:
                    st.error("Failed to generate authentication URL. Please check your Google OAuth configuration.")
            except ValueError as e:
                st.error(f"❌ Configuration Error: {str(e)}")
                st.info("Please check your `.env` file and ensure all required environment variables are set.")
            except Exception as e:
                st.error(f"❌ Authentication Error: {str(e)}")
                st.info("Please check your Google OAuth configuration and try again.")
        
        # Handle OAuth callback
        query_params = st.query_params
        if 'code' in query_params and not st.session_state.authenticated:
            st.info("🔄 Processing authentication...")
            
            # Extract the authorization code
            code = query_params['code']
            
            # Construct the callback URL properly
            callback_url = f"http://localhost:8501?code={code}"
            if 'state' in query_params:
                callback_url += f"&state={query_params['state']}"
            if 'scope' in query_params:
                callback_url += f"&scope={query_params['scope']}"
            
            # Process authentication
            with st.spinner("Completing sign in..."):
                auth_manager = GoogleAuthManager()
                # Authenticate with Google
                credentials, user_info = auth_manager.authenticate_with_google(callback_url)
                print(f"🔍 [UI] Authentication completed, user_info: {user_info}")
                
                if credentials and user_info:
                    st.session_state.credentials = credentials
                    st.session_state.user_info = user_info
                    print(f"🔍 [UI] User info stored in session: {user_info}")
                    st.session_state.authenticated = True
                    st.success("✅ Successfully authenticated with Google!")
                    # Clear query params and redirect to clean URL
                    st.query_params.clear()
                    st.rerun()
                else:
                    st.error("❌ Authentication failed. Please try again.")
                    st.query_params.clear()
        
        st.markdown("---")
        st.markdown("### 📋 Setup Instructions")
        st.markdown("""
        1. **Google Cloud Setup**: Create a Google Cloud project and enable the Google Drive API
        2. **OAuth Configuration**: Set up OAuth 2.0 credentials for a web application
        3. **Environment Variables**: Add your Google OAuth credentials to your `.env` file:
           ```
           GOOGLE_OAUTH_CLIENT_ID=your_client_id
           GOOGLE_OAUTH_CLIENT_SECRET=your_client_secret
           GOOGLE_API_KEY=your_gemini_api_key
           LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key
           ```
        4. **OpenSearch**: Ensure OpenSearch is running and accessible
        """)
        
        return
    
    # User is authenticated
    user_email = st.session_state.user_info.get('email', 'unknown')
    print(f"🔍 [UI] User email from session: {user_email}")
    st.success(f"👋 Welcome, {user_email}!")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("📁 Document Processing")
        
        # Force reprocess option
        force_reprocess = st.checkbox("🔄 Force Reprocess All Documents", 
                                    help="Clear existing data and reprocess all documents")
        
        if force_reprocess:
            if st.button("🗑️ Clear All Data", type="secondary"):
                # Clear enhanced metadata
                metadata_manager = MetadataManager()
                enhanced_metadata_path = metadata_manager.get_metadata_path(user_email)
                if os.path.exists(enhanced_metadata_path):
                    os.remove(enhanced_metadata_path)
                
                # Clear OpenSearch indices
                opensearch_client = OpenSearchClient()
                if opensearch_client.client:
                    if opensearch_client.client.indices.exists(index=OPENSEARCH_INDEX):
                        opensearch_client.client.indices.delete(index=OPENSEARCH_INDEX)
                    if opensearch_client.client.indices.exists(index=SHEETS_INDEX):
                        opensearch_client.client.indices.delete(index=SHEETS_INDEX)
                
                # Reset session state
                st.session_state.processed_pdfs = False
                st.session_state.processed_sheets = False
                st.session_state.unified_retriever = None
                st.session_state.start_chatting = False
                st.success("✅ All data cleared!")
                
                st.rerun()
        
        # Process documents button
        if st.button("🚀 Process Documents", type="primary"):
            with st.spinner("Processing your Google Drive..."):
                try:
                    print(f"🔍 [UI] Processing documents for user: {user_email}")
                    # Import the processing function
                    from core.processing.document_processor import process_all_user_documents_unified
                    process_all_user_documents_unified(st.session_state.credentials, user_email)
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
        
        # Show processing status
        if st.session_state.processing_status:
            st.info(st.session_state.processing_status)
        
        # Show document counts
        opensearch_client = OpenSearchClient()
        pdf_count = opensearch_client.get_count(OPENSEARCH_INDEX)
        sheets_count = opensearch_client.get_count(SHEETS_INDEX)
        
        if pdf_count > 0:
            st.success(f"📄 {pdf_count} PDF/Doc chunks in knowledge base")
        if sheets_count > 0:
            st.success(f"📊 {sheets_count} sheet chunks in knowledge base")
        
        # Show enhanced metadata information
        metadata_manager = MetadataManager()
        print(f"🔍 [UI] Loading metadata for user: {user_email}")
        enhanced_metadata = metadata_manager.load_metadata(user_email)
        if enhanced_metadata:
            with st.expander("📋 File Metadata Summary", expanded=False):
                total_files = len(enhanced_metadata)
                file_types = {}
                total_size_mb = 0
                total_chunks = 0
                
                for file_hash, file_data in enhanced_metadata.items():
                    file_type = file_data.get('file_type', 'unknown')
                    file_types[file_type] = file_types.get(file_type, 0) + 1
                    total_size_mb += file_data.get('file_size_mb', 0)
                    total_chunks += file_data.get('total_chunks', 0)
                
                st.write(f"**Total Files:** {total_files}")
                st.write(f"**Total Size:** {total_size_mb:.2f} MB")
                st.write(f"**Total Chunks:** {total_chunks}")
                
                st.write("**File Types:**")
                for file_type, count in file_types.items():
                    st.write(f"  • {file_type.title()}: {count}")
                
                # Show recent files
                recent_files = sorted(
                    enhanced_metadata.items(),
                    key=lambda x: x[1].get('processed_time', ''),
                    reverse=True
                )[:5]
                
                st.write("**Recently Processed:**")
                for file_hash, file_data in recent_files:
                    st.write(f"  • {file_data.get('file_name', 'Unknown')} ({file_data.get('file_type', 'unknown')})")
        
        # Debug information
        with st.expander("🔧 Debug Info"):
            st.write(f"processed_pdfs: {st.session_state.processed_pdfs}")
            st.write(f"processed_sheets: {st.session_state.processed_sheets}")
            st.write(f"start_chatting: {st.session_state.start_chatting}")
            st.write(f"has_unified_retriever: {st.session_state.unified_retriever is not None}")
            st.write(f"has_llm: {st.session_state.llm is not None}")
            st.write(f"pdf_chunks: {pdf_count}")
            st.write(f"sheet_chunks: {sheets_count}")
            st.write(f"enhanced_metadata_files: {len(enhanced_metadata) if enhanced_metadata else 0}")
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.markdown("""
        **Unified Approach:**
        - **Single LLM Call**: Retrieves chunks from both sources and generates response in one call
        - **Better Integration**: Sees all relevant information simultaneously
        - **More Efficient**: Fewer API calls, lower latency, lower cost
        - **Better Quality**: More coherent responses with better cross-references
        
        **Supports:** PDF files, Google Docs, Excel files, and Google Sheets
        """)
        

    
    # Main content area
    if not st.session_state.processed_pdfs and not st.session_state.processed_sheets and not st.session_state.start_chatting:
        # Check if there are existing documents in OpenSearch
        opensearch_client = OpenSearchClient()
        has_existing_docs = opensearch_client.check_documents(OPENSEARCH_INDEX) or opensearch_client.check_documents(SHEETS_INDEX)
        
        if has_existing_docs:
            st.info("📚 Found existing documents in your knowledge base!")
            st.markdown("You can start asking questions about your previously processed documents.")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🚀 Start Asking Questions", type="primary", use_container_width=True):
                    with st.spinner("Initializing unified chat interface..."):
                        # Import the initialization function
                        from core.retrieval.unified_retriever import initialize_unified_retriever_from_existing
                        if initialize_unified_retriever_from_existing():
                            st.success("✅ Unified chat interface ready! You can now ask questions about your documents and sheets.")
                            st.rerun()
                        else:
                            st.error("❌ Failed to initialize unified chat interface.")
            
            st.markdown("---")
            st.markdown("**Or** click 'Process Documents' in the sidebar to scan for new documents and sheets in your Google Drive.")
        else:
            st.info("📚 No documents or sheets processed yet. Click 'Process Documents' in the sidebar to get started.")
        return
    
    # Chat interface
    if st.session_state.processed_pdfs or st.session_state.start_chatting or st.session_state.processed_sheets:
        st.header("💬 Ask Questions About Your Documents & Sheets")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            print(f"🔍 [UI] Chat input received for user: {user_email}")
            print(f"📝 [UI] Query: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            
            # Add user message to chat history
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response using unified system
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                try:
                    print(f"🔍 [UI] Starting response generation...")
                    
                    # Check if we have unified retriever and LLM
                    has_unified_retriever = st.session_state.unified_retriever is not None
                    has_llm = st.session_state.llm is not None
                    
                    print(f"📊 [UI] Component status - has_unified_retriever: {has_unified_retriever}, has_llm: {has_llm}")
                    
                    if has_unified_retriever and has_llm:
                        print(f"✅ [UI] Both retriever and LLM available, proceeding with query processing")
                        
                        # Classify the query first for debugging
                        print(f"🔍 [UI] Starting query classification...")
                        from core.retrieval.query_classifier import classify_query
                        classification = classify_query(prompt)
                        query_type = classification.get('classification', 'content')
                        confidence = classification.get('confidence', 0.0)
                        
                        print(f"✅ [UI] Query classified as: {query_type} (confidence: {confidence:.2f})")
                        
                        # Show classification info in debug mode
                        with st.expander("🔍 Query Classification Info", expanded=False):
                            st.write(f"**Query Type:** {query_type}")
                            st.write(f"**Confidence:** {confidence:.2f}")
                            st.write(f"**Reasoning:** {classification.get('reasoning', 'No reasoning')}")
                        
                        # Use unified query system with user email
                        print(f"🔍 [UI] Calling unified query system for user: {user_email}")
                        from core.retrieval.response_generator import query_unified_system
                        answer = query_unified_system(
                            prompt, 
                            st.session_state.unified_retriever, 
                            st.session_state.llm,
                            st.session_state.messages,
                            user_email
                        )
                        
                        print(f"✅ [UI] Response generated for user: {user_email}, length: {len(answer)} characters")
                        
                        # Display response
                        message_placeholder.markdown(answer)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        print(f"❌ [UI] Missing components - has_unified_retriever: {has_unified_retriever}, has_llm: {has_llm}")
                        message_placeholder.error("❌ Unified retriever not initialized. Please process documents and sheets first.")
                except Exception as e:
                    print(f"❌ [UI] Error during response generation: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    message_placeholder.error(f"❌ Error generating response: {str(e)}")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main() 