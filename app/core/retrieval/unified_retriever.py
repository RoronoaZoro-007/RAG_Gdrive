"""
Unified retriever for document and sheet content.
Combines semantic search across multiple document types.
"""

from config.constants import DEFAULT_K_PDFS, DEFAULT_K_SHEETS, OPENSEARCH_INDEX, SHEETS_INDEX
from core.database.opensearch import OpenSearchClient
from config import settings
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class UnifiedRetriever:
    """
    Unified retriever that fetches relevant chunks from both PDFs/Docs and Sheets.
    Combines semantic search across multiple document types.
    """
    
    def __init__(self, pdf_vectorstore, sheets_index, embeddings):
        """
        Initialize unified retriever with vector stores and embeddings.
        
        Args:
            pdf_vectorstore: OpenSearch vector store for PDFs/Docs
            sheets_index: LlamaIndex for sheets
            embeddings: Embedding model for semantic search
        """
        self.pdf_vectorstore = pdf_vectorstore
        self.sheets_index = sheets_index
        self.embeddings = embeddings
        
        # Initialize OpenSearch client
        self.opensearch_client = OpenSearchClient()
        if not self.opensearch_client.client:
            print("⚠️ Warning: OpenSearch client initialization failed")
            self.opensearch_client = None
        
        # Thread-local storage for embeddings
        self._local = threading.local()
    
    def _get_embeddings(self):
        """Get thread-local embeddings instance."""
        if not hasattr(self._local, 'embeddings'):
            self._local.embeddings = self.embeddings
        return self._local.embeddings
    
    def _retrieve_pdf_chunks(self, enhanced_query, k_pdfs, is_sheet_query):
        """Retrieve PDF chunks (for parallel processing)."""
        try:
            print(f"🔄 [PARALLEL] Starting PDF chunk retrieval...")
            pdf_chunks = []
            
            if not self.pdf_vectorstore:
                print("❌ [PARALLEL] PDF vectorstore not available")
                return pdf_chunks
            
            # Use hybrid search approach to avoid dominance by single documents
            print("🔍 [PARALLEL] Using hybrid search approach for PDFs...")
            
            # Check if OpenSearch client is available
            if not hasattr(self, 'opensearch_client') or not self.opensearch_client:
                print("❌ [PARALLEL] OpenSearch client not available, falling back to simple search")
                # Fallback to simple semantic search
                pdf_docs = self.pdf_vectorstore.similarity_search(enhanced_query, k=k_pdfs)
                for doc in pdf_docs:
                    chunk = {
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'source_type': 'pdf_doc'
                    }
                    pdf_chunks.append(chunk)
                print(f"📄 [PARALLEL] Retrieved {len(pdf_chunks)} PDF chunks using fallback search")
                return pdf_chunks
            
            print(f"✅ [PARALLEL] OpenSearch client available for PDF retrieval")
            
            # Extract key terms from the query
            query_terms = enhanced_query.lower().split()
            meaningful_terms = [term for term in query_terms if len(term) > 3 and term not in ['the', 'and', 'for', 'with', 'this', 'that', 'file', 'containing', 'has', 'includes', 'what', 'does', 'tell', 'me', 'about']]
            
            # Look for potential file names in the query
            potential_file_names = []
            for i, term in enumerate(query_terms):
                if term in ['pdf', 'document', 'file'] and i + 1 < len(query_terms):
                    potential_file_names.append(query_terms[i + 1])
                elif term.endswith('.pdf') or term.endswith('.doc'):
                    potential_file_names.append(term)
            
            # Extract potential document names from query using intelligent parsing
            # Look for patterns that might indicate document names
            query_words = enhanced_query.lower().split()
            
            # Check for file extensions
            for word in query_words:
                if word.endswith('.pdf') or word.endswith('.doc') or word.endswith('.docx'):
                    potential_file_names.append(word)
            
            # Check for quoted strings (potential file names)
            import re
            quoted_names = re.findall(r'"([^"]+)"', enhanced_query)
            potential_file_names.extend(quoted_names)
            
            # Look for capitalized phrases that might be document names
            # Split by common separators and look for title-case patterns
            title_patterns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', enhanced_query)
            for pattern in title_patterns:
                if len(pattern.split()) >= 2:  # At least 2 words for a document name
                    potential_file_names.append(pattern)
            
            # Remove duplicates and filter out common words
            common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall'}
            potential_file_names = [name for name in potential_file_names if name.lower() not in common_words]
            potential_file_names = list(set(potential_file_names))  # Remove duplicates
            
            print(f"🎯 [PARALLEL] Potential file names to search: {potential_file_names}")
            
            # Build hybrid search query
            search_queries = []
            
            # 1. Semantic search with query text
            search_queries.append({
                'match': {
                    'text': {
                        'query': enhanced_query,
                        'boost': 1.0
                    }
                }
            })
            
            # 2. Phrase matching for exact terms
            search_queries.append({
                'match_phrase': {
                    'text': {
                        'query': enhanced_query,
                        'boost': 2.0
                    }
                }
            })
            
            # 3. Boost specific file names if mentioned (using fuzzy matching)
            for file_name in potential_file_names:
                search_queries.append({
                    'fuzzy': {
                        'metadata.source': {
                            'value': file_name,
                            'fuzziness': 'AUTO',
                            'boost': 5.0  # High boost for file name matches
                        }
                    }
                })
                search_queries.append({
                    'match_phrase': {
                        'metadata.source': {
                            'query': file_name,
                            'boost': 10.0  # Very high boost for exact file name matches
                        }
                    }
                })
            
            # 4. Add fuzzy matching for document names in content
            for file_name in potential_file_names:
                search_queries.append({
                    'fuzzy': {
                        'text': {
                            'value': file_name,
                            'fuzziness': 'AUTO',
                            'boost': 3.0
                        }
                    }
                })
            
            # 5. Add meaningful term searches
            for term in meaningful_terms:
                search_queries.append({
                    'match': {
                        'text': {
                            'query': term,
                            'boost': 1.5
                        }
                    }
                })
            
            # 6. Add k-NN vector search for semantic similarity
            embedding = self._get_embeddings().embed_query(enhanced_query)
            search_queries.append({
                'knn': {
                    'vector_field': {
                        'vector': embedding,
                        'k': k_pdfs * 2  # Get more candidates for diversity
                    }
                }
            })
            
            # Execute hybrid search
            hybrid_results = self.opensearch_client.client.search(
                index=OPENSEARCH_INDEX,
                body={
                    'query': {
                        'bool': {
                            'should': search_queries,
                            'minimum_should_match': 1
                        }
                    },
                    'size': k_pdfs * 3,  # Get more results for diversity
                    'collapse': {
                        'field': 'metadata.source.keyword'  # Collapse by source to ensure diversity
                    }
                }
            )
            
            # Process results and ensure diversity
            seen_sources = set()
            source_counts = {}
            
            for hit in hybrid_results['hits']['hits']:
                source = hit['_source']
                source_name = source['metadata']['source']
                
                # Count how many chunks we have from this source
                current_count = source_counts.get(source_name, 0)
                
                # Limit to 3 chunks per source to ensure diversity
                if current_count < 3:
                    chunk = {
                        'content': source['text'],
                        'metadata': source['metadata'],
                        'source_type': 'pdf_doc'
                    }
                    pdf_chunks.append(chunk)
                    source_counts[source_name] = current_count + 1
                    seen_sources.add(source_name)
                    
                    # Stop if we have enough total chunks
                    if len(pdf_chunks) >= k_pdfs:
                        break
            
            # If we still don't have enough results, try additional search strategies
            if len(pdf_chunks) < k_pdfs // 2:
                print("🔍 [PARALLEL] Trying additional search strategies for PDFs...")
                
                # Strategy 1: Fall back to semantic search
                try:
                    pdf_docs = self.pdf_vectorstore.similarity_search(enhanced_query, k=k_pdfs)
                    for doc in pdf_docs:
                        chunk = {
                            'content': doc.page_content,
                            'metadata': doc.metadata,
                            'source_type': 'pdf_doc'
                        }
                        if not any(existing['content'] == chunk['content'] for existing in pdf_chunks):
                            pdf_chunks.append(chunk)
                except Exception as e:
                    print(f"⚠️ [PARALLEL] Semantic search fallback failed: {str(e)}")
                
                # Strategy 2: If we have potential file names but no results, try direct file search
                if potential_file_names and len(pdf_chunks) < k_pdfs // 2:
                    print("🔍 [PARALLEL] Trying direct file name search for PDFs...")
                    for file_name in potential_file_names:
                        try:
                            file_results = self.opensearch_client.client.search(
                                index=OPENSEARCH_INDEX,
                                body={
                                    'query': {
                                        'bool': {
                                            'should': [
                                                {'fuzzy': {'metadata.source': {'value': file_name, 'fuzziness': 'AUTO'}}},
                                                {'match': {'metadata.source': file_name}},
                                                {'wildcard': {'metadata.source': f'*{file_name}*'}}
                                            ],
                                            'minimum_should_match': 1
                                        }
                                    },
                                    'size': 5
                                }
                            )
                            
                            for hit in file_results['hits']['hits']:
                                source = hit['_source']
                                chunk = {
                                    'content': source['text'],
                                    'metadata': source['metadata'],
                                    'source_type': 'pdf_doc'
                                }
                                if not any(existing['content'] == chunk['content'] for existing in pdf_chunks):
                                    pdf_chunks.append(chunk)
                                    
                                if len(pdf_chunks) >= k_pdfs:
                                    break
                        except Exception as e:
                            print(f"⚠️ [PARALLEL] File name search failed for '{file_name}': {str(e)}")
            
            print(f"📄 [PARALLEL] Retrieved {len(pdf_chunks)} PDF/Doc chunks from {len(seen_sources)} different sources")
            print(f"📋 [PARALLEL] Sources found: {list(seen_sources)}")
            print(f"📊 [PARALLEL] Source counts: {source_counts}")
            return pdf_chunks
            
        except Exception as e:
            print(f"❌ [PARALLEL] Error retrieving PDF chunks: {str(e)}")
            # Fallback to simple semantic search
            try:
                pdf_docs = self.pdf_vectorstore.similarity_search(enhanced_query, k=k_pdfs)
                pdf_chunks = [
                    {
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'source_type': 'pdf_doc'
                    }
                    for doc in pdf_docs
                ]
                return pdf_chunks
            except Exception as fallback_error:
                print(f"❌ [PARALLEL] Fallback search also failed: {str(fallback_error)}")
                return []
    
    def _retrieve_sheet_chunks(self, enhanced_query, k_sheets, is_sheet_query):
        """Retrieve sheet chunks (for parallel processing)."""
        try:
            print(f"🔄 [PARALLEL] Starting sheet chunk retrieval...")
            sheets_chunks = []
            
            if not self.sheets_index:
                print("❌ [PARALLEL] Sheets index not available")
                return sheets_chunks
            
            print(f"📊 [PARALLEL] Sheets index available: {type(self.sheets_index)}")
            
            # For sheet-specific queries, also try direct OpenSearch search
            if is_sheet_query:
                print("🔍 [PARALLEL] Using direct OpenSearch search for sheet data...")
                
                # Extract key terms from the query for targeted search
                query_terms = enhanced_query.lower().split()
                # Filter out common words and keep meaningful terms
                meaningful_terms = [term for term in query_terms if len(term) > 3 and term not in ['the', 'and', 'for', 'with', 'this', 'that', 'file', 'containing', 'has', 'includes']]
                
                # Search for sheet content using hybrid approach
                search_queries = [
                    {'match': {'content': enhanced_query}},
                    {'match_phrase': {'content': enhanced_query}}
                ]
                
                # Add targeted searches for meaningful terms
                for term in meaningful_terms:
                    search_queries.append({'match': {'content': term}})
                
                sheet_results = self.opensearch_client.client.search(
                    index=SHEETS_INDEX,
                    body={
                        'query': {
                            'bool': {
                                'should': search_queries,
                                'minimum_should_match': 1
                            }
                        },
                        'size': k_sheets
                    }
                )
                
                # Add direct search results
                for hit in sheet_results['hits']['hits']:
                    source = hit['_source']
                    chunk = {
                        'content': source['content'],
                        'metadata': source.get('metadata', {}),
                        'source_type': 'sheet'
                    }
                    # Avoid duplicates
                    if not any(existing['content'] == chunk['content'] for existing in sheets_chunks):
                        sheets_chunks.append(chunk)
            
            # Also use LlamaIndex query engine for semantic search
            # Set the LLM explicitly to avoid OpenAI fallback
            from llama_index.core import Settings
            from core.processing.sheet_processor import SheetProcessor
            
            # Initialize LLM for LlamaIndex
            llm_for_sheets = SheetProcessor.initialize_llm()
            if llm_for_sheets:
                Settings.llm = llm_for_sheets
            
            query_engine = self.sheets_index.as_query_engine(
                response_mode="tree_summarize",
                similarity_top_k=k_sheets
            )
            sheets_response = query_engine.query(enhanced_query)
            
            # Extract source nodes (chunks) from response
            if hasattr(sheets_response, 'source_nodes'):
                for node in sheets_response.source_nodes:
                    chunk = {
                        'content': node.node.text,
                        'metadata': node.node.metadata,
                        'source_type': 'sheet'
                    }
                    # Avoid duplicates
                    if not any(existing['content'] == chunk['content'] for existing in sheets_chunks):
                        sheets_chunks.append(chunk)
            
            print(f"📊 [PARALLEL] Retrieved {len(sheets_chunks)} sheet chunks")
            return sheets_chunks
            
        except Exception as e:
            print(f"❌ [PARALLEL] Error retrieving sheet chunks: {str(e)}")
            return []

    def retrieve_relevant_chunks(self, query, k_pdfs=DEFAULT_K_PDFS, k_sheets=DEFAULT_K_SHEETS, chat_history=None):
        """
        Retrieve relevant chunks from both PDFs/Docs and Sheets using parallel processing.
        
        Args:
            query (str): User's question
            k_pdfs (int): Number of PDF chunks to retrieve
            k_sheets (int): Number of sheet chunks to retrieve
            chat_history (list, optional): Previous conversation context
            
        Returns:
            tuple: (pdf_chunks, sheets_chunks) lists of relevant content
        """
        try:
            print(f"🔍 [UNIFIED_RETRIEVER] Starting parallel chunk retrieval for query: '{query[:50]}...'")
            print(f"📊 [UNIFIED_RETRIEVER] Parameters - k_pdfs: {k_pdfs}, k_sheets: {k_sheets}")
            
            # Create context-aware query if history is available
            enhanced_query = query
            if chat_history and len(chat_history) > 0:
                # Use last few exchanges to create a more context-aware query
                recent_history = chat_history[-6:]  # Last 6 messages (3 exchanges)
                context_parts = []
                for message in recent_history:
                    if message["role"] == "user":
                        context_parts.append(f"User asked: {message['content']}")
                    else:
                        context_parts.append(f"Assistant responded: {message['content'][:100]}...")  # Truncate long responses
                
                context_summary = "\n".join(context_parts[-4:])  # Use last 4 context parts
                enhanced_query = f"Context from recent conversation:\n{context_summary}\n\nCurrent question: {query}"
                print(f"🔍 Using enhanced query with conversation context")
            
            # Detect if this is a sheet-specific query (generic approach)
            sheet_keywords = ['sheet', 'spreadsheet', 'excel', 'table', 'data']
            is_sheet_query = any(keyword in query.lower() for keyword in sheet_keywords)
            
            # Adjust retrieval parameters based on query type
            if is_sheet_query:
                print("📊 Detected sheet-specific query, prioritizing sheet retrieval...")
                k_sheets = max(k_sheets, 10)  # Increase sheet retrieval
                k_pdfs = min(k_pdfs, 5)  # Reduce PDF retrieval for sheet queries
            
            # Parallel retrieval from both stores
            print(f"🚀 [PARALLEL] Starting parallel retrieval from PDF and sheet stores...")
            
            pdf_chunks = []
            sheets_chunks = []
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both retrieval tasks
                pdf_future = executor.submit(self._retrieve_pdf_chunks, enhanced_query, k_pdfs, is_sheet_query)
                sheets_future = executor.submit(self._retrieve_sheet_chunks, enhanced_query, k_sheets, is_sheet_query)
                
                # Collect results as they complete
                try:
                    pdf_chunks = pdf_future.result()
                    print(f"✅ [PARALLEL] PDF retrieval completed: {len(pdf_chunks)} chunks")
                except Exception as e:
                    print(f"❌ [PARALLEL] PDF retrieval failed: {str(e)}")
                
                try:
                    sheets_chunks = sheets_future.result()
                    print(f"✅ [PARALLEL] Sheet retrieval completed: {len(sheets_chunks)} chunks")
                except Exception as e:
                    print(f"❌ [PARALLEL] Sheet retrieval failed: {str(e)}")
            
            # If no chunks were retrieved, try a simple fallback
            if len(pdf_chunks) == 0 and len(sheets_chunks) == 0:
                print("⚠️ [PARALLEL] No chunks retrieved, trying simple fallback search...")
                
                # Try simple semantic search on PDFs
                if self.pdf_vectorstore:
                    try:
                        pdf_docs = self.pdf_vectorstore.similarity_search(query, k=5)
                        for doc in pdf_docs:
                            chunk = {
                                'content': doc.page_content,
                                'metadata': doc.metadata,
                                'source_type': 'pdf_doc'
                            }
                            pdf_chunks.append(chunk)
                        print(f"📄 [PARALLEL] Fallback: Retrieved {len(pdf_chunks)} PDF chunks")
                    except Exception as e:
                        print(f"❌ [PARALLEL] Fallback PDF search failed: {str(e)}")
                
                # Try simple search on sheets
                if self.sheets_index:
                    try:
                        # Set the LLM explicitly for fallback too
                        from llama_index.core import Settings
                        from core.processing.sheet_processor import SheetProcessor
                        
                        llm_for_sheets = SheetProcessor.initialize_llm()
                        if llm_for_sheets:
                            Settings.llm = llm_for_sheets
                        
                        query_engine = self.sheets_index.as_query_engine(similarity_top_k=5)
                        sheets_response = query_engine.query(query)
                        if hasattr(sheets_response, 'source_nodes'):
                            for node in sheets_response.source_nodes:
                                chunk = {
                                    'content': node.node.text,
                                    'metadata': node.node.metadata,
                                    'source_type': 'sheet'
                                }
                                sheets_chunks.append(chunk)
                        print(f"📊 [PARALLEL] Fallback: Retrieved {len(sheets_chunks)} sheet chunks")
                    except Exception as e:
                        print(f"❌ [PARALLEL] Fallback sheet search failed: {str(e)}")
            
            print(f"📊 [PARALLEL] Final result: {len(pdf_chunks)} PDF chunks, {len(sheets_chunks)} sheet chunks")
            return pdf_chunks, sheets_chunks
            
        except Exception as e:
            print(f"❌ [PARALLEL] Error in unified retrieval: {str(e)}")
            return [], []
    
    def initialize_from_existing(self):
        """Initialize unified retriever from existing documents."""
        try:
            from core.processing.sheet_processor import SheetProcessor
            from core.retrieval.response_generator import ResponseGenerator
            
            # Check if we have existing documents
            if not self.opensearch_client.check_documents():
                return False
            
            # Initialize LLM
            llm = SheetProcessor.initialize_llm()
            if not llm:
                return False
            
            # Initialize embeddings
            embeddings = SheetProcessor.initialize_embeddings()
            if not llm:
                return False
            # Create vector store
            from langchain_community.vectorstores import OpenSearchVectorSearch
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
            
            # Update retriever
            self.pdf_vectorstore = pdf_vectorstore
            self.embeddings = embeddings
            
            return True
            
        except Exception as e:
            print(f"❌ Error initializing from existing: {str(e)}")
            return False

def initialize_unified_retriever_from_existing():
    """Initialize unified retriever from existing documents in OpenSearch."""
    try:
        print(f"🚀 [INIT_UNIFIED_RETRIEVER] Starting initialization from existing documents...")
        
        from core.processing.sheet_processor import SheetProcessor
        from core.database.opensearch import OpenSearchClient
        from config.constants import OPENSEARCH_INDEX, SHEETS_INDEX
        from config.settings import settings
        import streamlit as st
        
        # Check if we have existing documents
        print(f"🔍 [INIT_UNIFIED_RETRIEVER] Checking for existing documents...")
        opensearch_client = OpenSearchClient()
        has_pdfs = opensearch_client.check_documents(OPENSEARCH_INDEX)
        has_sheets = opensearch_client.check_documents(SHEETS_INDEX)
        
        print(f"📊 [INIT_UNIFIED_RETRIEVER] Document status - has_pdfs: {has_pdfs}, has_sheets: {has_sheets}")
        
        if not has_pdfs and not has_sheets:
            print("❌ No existing documents found")
            return False
        
        # Initialize LLM
        print(f"🔍 [INIT_UNIFIED_RETRIEVER] Initializing LLM...")
        llm = SheetProcessor.initialize_llm()
        if not llm:
            print("❌ [INIT_UNIFIED_RETRIEVER] Failed to initialize LLM")
            return False
        print(f"✅ [INIT_UNIFIED_RETRIEVER] LLM initialized: {type(llm)}")
        
        # Initialize embeddings using LangChain version (like original)
        print(f"🔍 [INIT_UNIFIED_RETRIEVER] Initializing embeddings...")
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from config.settings import settings
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=settings.GOOGLE_API_KEY
        )
        if not embeddings:
            print("❌ [INIT_UNIFIED_RETRIEVER] Failed to initialize embeddings")
            return False
        print(f"✅ [INIT_UNIFIED_RETRIEVER] Embeddings initialized: {type(embeddings)}")
        
        # Create vector store for PDFs if they exist
        pdf_vectorstore = None
        if has_pdfs:
            print(f"🔍 [INIT_UNIFIED_RETRIEVER] Creating PDF vector store...")
            from langchain_community.vectorstores import OpenSearchVectorSearch
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
            print(f"✅ [INIT_UNIFIED_RETRIEVER] PDF vector store created: {type(pdf_vectorstore)}")
        
        # Create sheets index if sheets exist
        sheets_index = None
        if has_sheets:
            print(f"🔍 [INIT_UNIFIED_RETRIEVER] Creating sheets index...")
            try:
                from llama_index.vector_stores.opensearch import OpensearchVectorStore, OpensearchVectorClient
                from llama_index.core import VectorStoreIndex, StorageContext
                
                # Create vector client first (like in original test.py)
                vector_client = OpensearchVectorClient(
                    settings.OPENSEARCH_URL,
                    SHEETS_INDEX,
                    768,
                    embedding_field="embedding",
                    text_field="content"
                )
                
                # Create OpenSearch vector store for LlamaIndex
                opensearch_vector_store = OpensearchVectorStore(vector_client)
                
                # Create storage context
                storage_context = StorageContext.from_defaults(vector_store=opensearch_vector_store)
                
                # Create index
                sheets_index = VectorStoreIndex.from_vector_store(
                    vector_store=opensearch_vector_store,
                    embed_model=SheetProcessor.initialize_embeddings()  # Use LlamaIndex embeddings for sheets
                )
                
                print("✅ Sheets index initialized successfully")
                
            except Exception as e:
                print(f"⚠️ Warning: Could not initialize sheets index: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Create unified retriever
        print(f"🔍 [INIT_UNIFIED_RETRIEVER] Creating unified retriever...")
        unified_retriever = UnifiedRetriever(pdf_vectorstore, sheets_index, embeddings)
        
        # Update session state
        print(f"🔍 [INIT_UNIFIED_RETRIEVER] Updating session state...")
        st.session_state.unified_retriever = unified_retriever
        st.session_state.llm = llm
        st.session_state.processed_pdfs = has_pdfs
        st.session_state.processed_sheets = has_sheets
        st.session_state.start_chatting = True
        
        print("✅ [INIT_UNIFIED_RETRIEVER] Unified retriever initialized from existing documents")
        return True
        
    except Exception as e:
        print(f"❌ [INIT_UNIFIED_RETRIEVER] Error initializing unified retriever from existing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False 