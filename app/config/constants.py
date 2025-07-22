"""
Constants for the SheetsLoader application.
Contains all hardcoded values and configuration constants.
"""

# OpenSearch configuration
OPENSEARCH_INDEX = "document_embeddings"
SHEETS_INDEX = "sheets_embeddings"
METADATA_INDEX = "file_metadata"

# Google OAuth configuration
SCOPES = [
    'openid',
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.activity',
    'https://www.googleapis.com/auth/drive.readonly',
    'https://www.googleapis.com/auth/drive.activity.readonly',
    'https://www.googleapis.com/auth/drive.metadata',
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile'
]

# MIME types for file scanning
PDF_MIME_TYPE = "application/pdf"
GOOGLE_DOC_MIME_TYPE = "application/vnd.google-apps.document"
SHEET_MIME_TYPES = [
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.google-apps.spreadsheet"
]

# File processing constants
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 300
DEFAULT_INITIAL_THRESHOLD = 0.6
DEFAULT_APPENDING_THRESHOLD = 0.8

# Retrieval constants
DEFAULT_K_PDFS = 20
DEFAULT_K_SHEETS = 10

# Session state defaults
SESSION_DEFAULTS = {
    'authenticated': False,
    'user_info': None,
    'credentials': None,
    'processed_pdfs': False,
    'processed_sheets': False,
    'chat_history': [],
    'unified_retriever': None,
    'llm': None,
    'processing_status': "",
    'start_chatting': False
}

# File paths
VECTORSTORES_DIR = "vectorstores"
TEMP_DIR_PREFIX = "sheetsloader_"

# Model configurations
GEMINI_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/embedding-001"
LLAMA_MODEL = "gemini-2.5-flash"

# Processing status messages
STATUS_MESSAGES = {
    'scanning': "📁 Scanning your Google Drive for PDF, Google Doc, and Sheet files...",
    'processing_pdf': "📄 Processing PDF: {file_name}",
    'processing_doc': "📝 Processing Google Doc: {file_name}",
    'processing_sheet': "📊 Processing Sheet: {file_name}",
    'creating_embeddings': "🧠 Creating embeddings for documents...",
    'processing_sheets': "📊 Processing sheets with LlamaIndex...",
    'ready': "✅ Processing complete! You can now ask questions about your documents and sheets."
} 