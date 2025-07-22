"""
Response generation for unified document queries.
Handles response creation from retrieved chunks and metadata.
"""

from core.database.metadata import MetadataManager

class ResponseGenerator:
    """Handles response generation for unified document queries."""
    
    def __init__(self):
        """Initialize response generator."""
        self.metadata_manager = MetadataManager()
    
    def create_mixed_response(self, pdf_chunks, sheets_chunks, query, llm, chat_history=None, user_email=None):
        """
        Create a response that includes metadata information.
        
        Args:
            pdf_chunks: Retrieved PDF chunks
            sheets_chunks: Retrieved sheet chunks
            query: User query
            llm: Language model
            chat_history: Previous chat messages
            user_email (str, optional): User's email for metadata access
        """
        print(f"🔍 [RESPONSE] Creating mixed response for user: {user_email}")
        
        # Load enhanced metadata for additional context
        enhanced_metadata = self.metadata_manager.load_metadata(user_email) if user_email else {}
        print(f"🔍 [RESPONSE] Loaded metadata for user: {user_email}, entries: {len(enhanced_metadata)}")
        
        # Collect unique file sources from chunks
        file_sources = set()
        for chunk in pdf_chunks + sheets_chunks:
            if isinstance(chunk, dict):
                source = chunk.get('metadata', {}).get('source', 'Unknown')
                file_sources.add(source)
        
        # Get metadata for each file source
        file_metadata_list = []
        for source in file_sources:
            # Find metadata for this file
            for file_hash, metadata in enhanced_metadata.items():
                if metadata.get('file_name') == source:
                    file_metadata_list.append(metadata)
                    break
        
        # Create response with file metadata
        if file_metadata_list:
            response = f"**Files containing relevant content:**\n\n"
            for i, file_meta in enumerate(file_metadata_list, 1):
                file_name = file_meta.get('file_name', 'Unknown')
                file_type = file_meta.get('file_type', 'Unknown')
                web_link = file_meta.get('web_view_link', '')
                file_size = file_meta.get('file_size_mb', 0)
                
                response += f"{i}. **{file_name}** ({file_type.title()})\n"
                if web_link:
                    response += f"   🔗 [View in Google Drive]({web_link})\n"
                response += f"   📏 Size: {file_size:.2f} MB\n\n"
        else:
            response = "No files found containing the requested content."
        
        return response
        
    def create_unified_response(self, pdf_chunks, sheets_chunks, query, llm, chat_history=None, user_email=None):
        """
        Create a unified response using all retrieved chunks in a single LLM call.
        
        Args:
            pdf_chunks (list): Retrieved PDF document chunks
            sheets_chunks (list): Retrieved sheet chunks
            query (str): User's original question
            llm: Language model instance
            chat_history (list, optional): Previous conversation context
            user_email (str, optional): User's email for metadata access
            
        Returns:
            str: Comprehensive response combining all relevant information
        """
        try:
            print(f"🤖 [CREATE_UNIFIED_RESPONSE] Starting response generation for query: '{query[:50]}...'")
            print(f"📊 [CREATE_UNIFIED_RESPONSE] Input - pdf_chunks: {len(pdf_chunks)}, sheets_chunks: {len(sheets_chunks)}, has_llm: {llm is not None}")
            
            if llm is None:
                print(f"❌ [CREATE_UNIFIED_RESPONSE] LLM is None, cannot generate response")
                return "❌ Error: Language model not initialized"
            
            # Load enhanced metadata for web links
            enhanced_metadata = {}
            if user_email:
                enhanced_metadata = self.metadata_manager.load_metadata(user_email)
                print(f"🔍 [CREATE_UNIFIED_RESPONSE] Loaded metadata for user: {user_email}, entries: {len(enhanced_metadata)}")
            
            # Prepare context from PDF chunks
            pdf_context = ""
            document_links = {}  # Store document links for later inclusion
            
            if pdf_chunks:
                pdf_context = "**DOCUMENTS & PDFS:**\n"
                for i, chunk in enumerate(pdf_chunks, 1):
                    if isinstance(chunk, dict):
                        content = chunk.get('content', '')
                        metadata = chunk.get('metadata', {})
                        source = metadata.get('source', 'Unknown')
                        page = metadata.get('page_number', 'N/A')
                        content_type = metadata.get('type', 'text')
                        
                        # Get web link for this document
                        web_link = ""
                        for file_hash, file_meta in enhanced_metadata.items():
                            if file_meta.get('file_name') == source:
                                web_link = file_meta.get('web_view_link', '')
                                if web_link:
                                    document_links[source] = web_link
                                break
                    else:
                        content = str(chunk)
                        source = 'Unknown'
                        page = 'N/A'
                        content_type = 'text'
                    pdf_context += f"\n--- Chunk {i} (Source: {source}, Page: {page}, Type: {content_type}) ---\n"
                    pdf_context += content + "\n"
            
            # Prepare context from sheet chunks
            sheets_context = ""
            if sheets_chunks:
                sheets_context = "\n**SPREADSHEETS & EXCEL FILES:**\n"
                for i, chunk in enumerate(sheets_chunks, 1):
                    if isinstance(chunk, dict):
                        content = chunk.get('content', '')
                        metadata = chunk.get('metadata', {})
                        source = metadata.get('source', 'Unknown')
                    else:
                        content = str(chunk)
                        source = 'Unknown'
                    sheets_context += f"\n--- Chunk {i} (Source: {source}) ---\n"
                    sheets_context += content + "\n"
            
            # Prepare conversation history
            history_context = ""
            if chat_history and len(chat_history) > 0:
                history_context = "\n**CONVERSATION HISTORY:**\n"
                # Include last 5 exchanges for context (to avoid token limits)
                recent_history = chat_history[-10:]  # Last 10 messages (5 exchanges)
                for message in recent_history:
                    role = "User" if message["role"] == "user" else "Assistant"
                    history_context += f"{role}: {message['content']}\n"
            
            # Create unified prompt with strict guidelines
                    unified_prompt = f"""# 🤖 UNIFIED AI DOCUMENT ASSISTANT

You are an advanced AI assistant with comprehensive access to information from both documents/PDFs and spreadsheets/Excel files. Your role is to provide accurate, well-structured, and contextually relevant responses based solely on the provided information.

## 📋 CORE RESPONSIBILITIES

### 🎯 Primary Objectives
- **Answer questions accurately** using only the provided context
- **Cite sources properly** by mentioning document names and content types
- **Maintain conversation flow** by referencing previous exchanges appropriately
- **Provide comprehensive insights** by combining text and visual information

### 🔍 Information Sources
- **Text Content**: Extracted from PDFs, Google Docs, and spreadsheets
- **Visual Content**: Image descriptions, diagrams, charts, and visual elements
- **Structured Data**: Numerical data, tables, and spreadsheet information
- **Metadata**: File sources, page numbers, and processing timestamps

## 📊 CONTEXT INFORMATION

### 📄 DOCUMENTS & PDFS
{pdf_context}

### 📈 SPREADSHEETS & EXCEL FILES
{sheets_context}

### 💬 CONVERSATION HISTORY
{history_context}

## 🎨 VISUAL CONTENT GUIDELINES

### 📎 Document Reference Guidelines
- **Always mention document names** when citing information from specific files
- **Use exact document names** as they appear in the source metadata
- **Include file extensions** when referencing documents (e.g., "RAG test.pdf", "Data.xlsx")
- **Reference page numbers** when available for PDF documents

### 🔍 Image-Related Keywords to Monitor
**Visual Elements:**
- image, picture, photo, screenshot, diagram, chart, graph, logo, icon

**Descriptive Terms:**
- visual, appearance, look like, show, display, depict, illustrate

**Interface Elements:**
- dashboard, interface, UI, layout, design, mockup, wireframe

**Technical Specifications:**
- color, style, format, size, dimensions, resolution

### 📸 Image Processing Protocol
1. **Priority Check**: When image-related keywords are detected, prioritize image descriptions
2. **Comprehensive Analysis**: Use both text and visual information for complete answers
3. **Source Attribution**: Always mention whether information came from text or images
4. **Visual Description**: When images are referenced, describe their content based on available descriptions
5. **Fallback Handling**: If visual content is requested but not found, clearly state this

## 🔗 URL AND LINK HANDLING

### 📎 Link Management
- **Always include URLs** when present in the context
- **Format as clickable links** using proper markdown syntax
- **Reference both section names and links** when discussing specific content
- **Maintain link integrity** throughout the response

## 📝 RESPONSE STRUCTURE GUIDELINES

### 🎯 Answer Format
1. **Direct Response**: Provide a clear, concise answer to the user's question
2. **Source Citation**: Reference document names and content types (text/image) - always mention the document name when citing information
3. **Context Integration**: Use conversation history for follow-up questions
4. **Data Presentation**: Use bullet points or numbered lists for multiple items
5. **Professional Tone**: Maintain helpful and informative communication style

### 📊 Data Analysis Approach
- **Numerical Context**: Provide trends and context for numerical data
- **Insight Generation**: Offer analysis and explanations, not just raw data
- **Cell References**: Clearly reference specific cells, rows, or columns when applicable
- **Trend Identification**: Highlight patterns and relationships in the data

## ⚠️ STRICT COMPLIANCE RULES

### 🚫 Prohibited Actions
- **Never answer from personal knowledge** - use only provided context
- **No speculation** about information not present in the context
- **No repetition** of previously discussed information unless specifically requested
- **No inclusion** of irrelevant details from the chunks

### ✅ Required Actions
- **Eliminate duplicates** and redundant information
- **Be concise and precise** in all responses
- **Cite sources exactly once** for each piece of information
- **Clearly state** when information is not available in the context

## 🎯 CURRENT QUERY

**USER QUESTION:** {query}

## 📤 RESPONSE GENERATION

Based on the comprehensive context provided above, generate a well-structured response that:

1. **Directly addresses** the user's question
2. **Integrates information** from all relevant sources (text, images, spreadsheets)
3. **Maintains conversation continuity** using the provided history
4. **Follows all formatting and citation guidelines**
5. **Provides actionable insights** when applicable

**ANSWER:**"""

            # Get response from LLM using complete method for raw strings
            if llm is None:
                return "❌ Error: Language model not initialized"
            
            print(f"🤖 Calling LLM with prompt length: {len(unified_prompt)}")
            response = llm.complete(unified_prompt)
            print(f"✅ LLM response received: {len(response.text)} characters")
            
            # Add document links to the response if available
            if document_links:
                # Extract document names mentioned in the response
                mentioned_docs = []
                response_lower = response.text.lower()
                
                for doc_name in document_links.keys():
                    # Check if document name is mentioned in the response
                    # Remove file extensions for better matching
                    doc_name_clean = doc_name.lower()
                    if doc_name_clean.endswith('.pdf'):
                        doc_name_clean = doc_name_clean[:-4]
                    elif doc_name_clean.endswith('.doc') or doc_name_clean.endswith('.docx'):
                        doc_name_clean = doc_name_clean[:-4] if doc_name_clean.endswith('.docx') else doc_name_clean[:-3]
                    elif doc_name_clean.endswith('.xlsx') or doc_name_clean.endswith('.xls'):
                        doc_name_clean = doc_name_clean[:-5] if doc_name_clean.endswith('.xlsx') else doc_name_clean[:-4]
                    
                    # Check for exact match or partial match
                    # Also check for quoted document names
                    if (doc_name_clean in response_lower or 
                        doc_name.lower() in response_lower or
                        f'"{doc_name}"' in response.text or
                        f"'{doc_name}'" in response.text):
                        mentioned_docs.append(doc_name)
                
                # Only add links for documents that are actually mentioned
                if mentioned_docs:
                    print(f"🔗 [RESPONSE] Adding links for {len(mentioned_docs)} mentioned documents: {mentioned_docs}")
                    response.text += "\n\n## 📎 Document Links\n"
                    for doc_name in mentioned_docs:
                        link = document_links[doc_name]
                        response.text += f"- **{doc_name}**: [View in Google Drive]({link})\n"
                else:
                    print(f"🔗 [RESPONSE] No document links added - no documents mentioned in response")
                    print(f"🔗 [RESPONSE] Available documents: {list(document_links.keys())}")
            
            return response.text
            
        except Exception as e:
            print(f"❌ Error creating unified response: {str(e)}")
            return f"An error occurred while processing your query: {str(e)}"
    
    def query_system(self, prompt, unified_retriever, llm, chat_history=None, user_email=None):
        """
        Query the unified system with a prompt.
        
        Args:
            prompt (str): User's query
            unified_retriever: UnifiedRetriever instance
            llm: Language model instance
            chat_history (list, optional): Previous conversation
            user_email (str, optional): User's email
            
        Returns:
            str: Generated response
        """
        try:
            # Retrieve relevant chunks
            pdf_chunks, sheets_chunks = unified_retriever.retrieve_relevant_chunks(
                prompt, chat_history=chat_history
            )
            
            # Generate response
            response = self.create_unified_response(
                pdf_chunks, sheets_chunks, prompt, llm, chat_history
            )
            
            return response
            
        except Exception as e:
            print(f"❌ Error querying unified system: {str(e)}")
            return f"An error occurred while processing your query: {str(e)}"

def query_unified_system(prompt, unified_retriever, llm, chat_history=None, user_email=None):
    """
    Query the unified system with enhanced metadata support.
    
    Args:
        prompt (str): User's query
        unified_retriever: UnifiedRetriever instance
        llm: Language model
        chat_history (list, optional): Previous conversation
        user_email (str, optional): User's email for metadata queries
        
    Returns:
        str: Generated response
    """
    print(f"🚀 [QUERY_UNIFIED_SYSTEM] Processing query for user: {user_email}")
    print(f"📊 [QUERY_UNIFIED_SYSTEM] Query: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    print(f"📊 [QUERY_UNIFIED_SYSTEM] Chat history: {chat_history is not None}, user_email: {user_email}")
    try:
        print(f"🚀 [QUERY_UNIFIED_SYSTEM] Starting unified system query for: '{prompt[:50]}...'")
        print(f"📊 [QUERY_UNIFIED_SYSTEM] Parameters - has_retriever: {unified_retriever is not None}, has_llm: {llm is not None}, has_history: {chat_history is not None}, user_email: {user_email}")
        
        print("🔍 Querying unified system...")
        
        # Classify the query
        print("🎯 Classifying query...")
        from core.retrieval.query_classifier import classify_query
        classification = classify_query(prompt)
        query_type = classification.get('classification', 'content')
        confidence = classification.get('confidence', 0.0)
        reasoning = classification.get('reasoning', 'No reasoning provided')
        
        print(f"📊 Query classified as: {query_type} (confidence: {confidence:.2f})")
        print(f"💭 Reasoning: {reasoning}")
        
        # Route query based on classification
        if query_type == 'metadata':
            print("📋 Handling metadata query...")
            if user_email:
                from core.retrieval.query_classifier import get_spacy_metadata_handler
                spacy_handler = get_spacy_metadata_handler()
                response = spacy_handler.handle_metadata_query(prompt, user_email)
                print(f"✅ [QUERY_UNIFIED_SYSTEM] Metadata query response generated for user: {user_email}")
            else:
                response = "❌ User email not available for metadata query processing."
        else:
            print("📄 Handling content query...")
            # Retrieve relevant chunks from both sources with history context
            pdf_chunks, sheets_chunks = unified_retriever.retrieve_relevant_chunks(prompt, k_pdfs=20, k_sheets=10, chat_history=chat_history)
            
            print(f"📄 Retrieved {len(pdf_chunks)} PDF chunks and {len(sheets_chunks)} sheet chunks")
            
            # Check if this is a mixed query (content search + metadata response)
            mixed_query_indicators = ["which file", "what file", "find the file", "show me the file", "which document", "what document"]
            is_mixed_query = any(indicator in prompt.lower() for indicator in mixed_query_indicators)
            
            # Create mixed response if needed
            if is_mixed_query and user_email:
                print(f"🔍 [QUERY_UNIFIED_SYSTEM] Creating mixed response for user: {user_email}")
                response_generator = ResponseGenerator()
                response = response_generator.create_mixed_response(pdf_chunks, sheets_chunks, prompt, llm, chat_history, user_email)
                print(f"✅ [QUERY_UNIFIED_SYSTEM] Mixed response created for user: {user_email}")
            else:
                # Create unified response in a single LLM call with history
                print("🔄 Creating unified response...")
                response_generator = ResponseGenerator()
                response = response_generator.create_unified_response(pdf_chunks, sheets_chunks, prompt, llm, chat_history, user_email)
                print(f"🔄 Response generated: {len(response)} characters")
        
        print("✅ Unified response generated successfully")
        return response
        
    except Exception as e:
        print(f"❌ Error in unified query system: {str(e)}")
        return f"An error occurred while processing your query: {str(e)}" 