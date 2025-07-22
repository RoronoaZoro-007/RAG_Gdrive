# 🤖 Unified AI Document Assistant

A comprehensive document processing and querying system that combines PDFs, Google Docs, and Sheets with AI-powered analysis using Gemini Vision, spaCy NLP, and vector search.

## 🚀 Features

### 📄 **Multi-Format Document Support**
- **PDFs**: Text extraction + image analysis with Gemini Vision
- **Google Docs**: Direct content extraction and processing
- **Google Sheets/Excel**: Structured data processing with LlamaParse

### 🧠 **Intelligent Query Processing**
- **Query Classification**: Automatically detects content vs metadata queries using spaCy NLP
- **Unified Retrieval**: Combines semantic search across all document types
- **Context-Aware Responses**: Uses conversation history for better answers

### 🔍 **Advanced Search Capabilities**
- **Semantic Chunking**: Intelligent text splitting using embeddings
- **Hybrid Search**: Combines vector similarity + keyword matching
- **Metadata Queries**: File properties, permissions, revision history
- **Content Queries**: Deep document analysis and information extraction

### 🛡️ **Data Management**
- **User Isolation**: Separate data storage per user
- **Incremental Processing**: Only processes new/modified files
- **Cleanup System**: Removes deleted files automatically
- **Revision Tracking**: Monitors file changes and updates

## 🏗️ Architecture

### **Core Components**

#### 1. **Image Processor** (`app/core/processing/image_processor.py`)
- **Gemini Vision Integration**: Analyzes images in PDFs
- **PyMuPDF Processing**: Extracts images and text from PDFs
- **Multi-format Support**: Handles various image formats

#### 2. **spaCy Metadata Handler** (`app/core/retrieval/query_classifier.py`)
- **NLP Classification**: Uses spaCy to classify query types
- **OpenSearch Integration**: Stores and queries file metadata
- **Query Routing**: Directs queries to appropriate handlers

#### 3. **Main Application** (`app/ui.py`)
- **Streamlit Interface**: Web-based user interface
- **Google OAuth**: Secure authentication
- **Unified Processing**: Orchestrates all components

### **Data Flow**

```
Google Drive → Authentication → File Discovery → Processing Pipeline
     ↓
PDFs: Text + Images → Gemini Vision → Vector Embeddings
Docs: Content → Semantic Chunking → Vector Embeddings  
Sheets: Structured Data → LlamaParse → Vector Embeddings
     ↓
OpenSearch Storage → Query Classification → Unified Retrieval → AI Response
```

## 🛠️ Setup Instructions

### **1. Quick Start**

#### Option 1: Automated Setup (Recommended)
```bash
# Make the setup script executable
chmod +x setup_venv.sh

# Run the automated setup
./setup_venv.sh

# Activate the virtual environment
source venv/bin/activate

# Run the application
streamlit run main.py
```

#### Option 2: Manual Setup
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm

# Run the application
streamlit run main.py
```

### **2. API Keys & Configuration**

Create a `.env` file with your API keys:

```env
# Google OAuth (for Drive access)
GOOGLE_OAUTH_CLIENT_ID=your_client_id
GOOGLE_OAUTH_CLIENT_SECRET=your_client_secret

# Google AI (for Gemini)
GOOGLE_API_KEY=your_gemini_api_key

# LlamaCloud (for sheet parsing)
LLAMA_CLOUD_API_KEY=your_llama_cloud_key

# OpenSearch (optional - defaults to localhost)
OPENSEARCH_URL=http://localhost:9200
OPENSEARCH_USERNAME=your_username
OPENSEARCH_PASSWORD=your_password
```

### **3. Google Cloud Setup**

1. **Create Google Cloud Project**
2. **Enable APIs**:
   - Google Drive API
   - Google OAuth2 API
3. **Create OAuth Credentials**:
   - Application type: Web application
   - Authorized redirect URIs: `http://localhost:8501`

### **4. Start OpenSearch**

```bash
# Using Docker
docker run -d --name opensearch -p 9200:9200 -p 9600:9600 opensearchproject/opensearch:latest

# Or install locally
# Follow OpenSearch installation guide
```

### **5. System Requirements**

- **Python**: 3.13 or later
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: At least 2GB free space
- **Network**: Internet connection for API access

## 🚀 Usage

### **1. Start the Application**

```bash
streamlit run main.py
```

### **2. Authentication**

1. Click "Connect Google Drive"
2. Complete Google OAuth flow
3. Grant necessary permissions

### **3. Process Documents**

1. Click "Process Documents" in sidebar
2. System scans your Google Drive
3. Processes PDFs, Docs, and Sheets
4. Creates vector embeddings and metadata

### **4. Ask Questions**

#### **Content Queries**
- "What does the quarterly report say about revenue?"
- "Find information about product specifications"
- "Summarize the key points from the meeting notes"

#### **Metadata Queries**
- "Show me all PDF files from last week"
- "Who has access to the budget spreadsheet?"
- "What are the largest files in my Drive?"
- "Show revision history for the project plan"

#### **Mixed Queries**
- "Which file contains information about marketing strategy?"
- "Find documents about Q4 planning"

## 🔧 Key Functions Explained

### **Image Processing**
- `ImageProcessor.init_gemini()`: Initialize Gemini Vision API
- `ImageProcessor.extract_images_from_pdf()`: Extract images from PDFs
- `ImageProcessor.process_pdf_images()`: Generate descriptions for all images

### **Query Classification**
- `QueryClassifier.classify_query()`: Determine if query is content or metadata
- `QueryClassifier.handle_metadata_query()`: Process metadata-specific questions
- `QueryClassifier._classify_metadata_query_type()`: Identify specific metadata query type

### **Document Processing**
- `DocumentProcessor.process_all_user_documents_unified()`: Main processing pipeline
- `semantic_chunk_text()`: Intelligent text splitting
- `SheetProcessor.create_sheets_index()`: Process spreadsheets with LlamaIndex

### **Data Management**
- `MetadataManager.save_metadata()`: Store comprehensive file info
- `MetadataManager.cleanup_modified_file()`: Remove old data when files change
- `MetadataManager.check_file_modification()`: Detect file changes

### **Search & Retrieval**
- `UnifiedRetriever`: Combines search across all document types
- `UnifiedRetriever.retrieve_relevant_chunks()`: Get relevant content from multiple sources
- `ResponseGenerator.create_unified_response()`: Generate comprehensive answers

## 📊 System Capabilities

### **Document Types Supported**
- **PDFs**: Text + images + links + metadata
- **Google Docs**: Full content + revision history
- **Google Sheets**: Structured data + formulas + formatting
- **Excel Files**: Direct processing + data extraction

### **Query Types Handled**
- **Content Analysis**: Deep document understanding
- **Metadata Queries**: File properties and organization
- **Temporal Queries**: Time-based file filtering
- **Relationship Queries**: Permissions and sharing
- **Property Queries**: File sizes, types, counts

### **AI Features**
- **Gemini Vision**: Image analysis and description
- **Semantic Search**: Context-aware content retrieval
- **Conversation Memory**: Maintains context across queries
- **Hybrid Search**: Combines multiple search strategies
