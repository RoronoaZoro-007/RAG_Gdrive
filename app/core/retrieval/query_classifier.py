"""
Query classification using spaCy NLP.
Handles classification of queries as content vs metadata and metadata query processing.
"""

import os
import json
import hashlib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import spacy
from spacy.tokens import Doc
from spacy.language import Language
from spacy.matcher import Matcher
from opensearchpy import OpenSearch

from config.settings import settings
from config.constants import METADATA_INDEX

@dataclass
class QueryClassification:
    """Result of query classification"""
    classification: str  # 'content' or 'metadata'
    confidence: float
    reasoning: str
    query_type: Optional[str] = None  # 'temporal', 'relationship', 'property', etc.

@dataclass
class MetadataQuery:
    """Structured metadata query"""
    query_type: str
    parameters: Dict[str, Any]
    original_query: str

class QueryClassifier:
    """
    spaCy-based metadata classification and handling for document queries.
    Uses NLP to classify queries as content vs metadata and handles metadata searches.
    """
    
    def __init__(self):
        """Initialize spaCy model, OpenSearch client, and pattern matcher."""
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise
        
        # Initialize OpenSearch client
        self.opensearch_client = self._create_opensearch_client()
        
        # Setup patterns and matcher
        self._setup_patterns()
    
    def _create_opensearch_client(self) -> Optional[OpenSearch]:
        """Create OpenSearch client connection for metadata storage and retrieval."""
        try:
            if not settings.OPENSEARCH_USERNAME and not settings.OPENSEARCH_PASSWORD:
                client = OpenSearch(
                    hosts=[settings.OPENSEARCH_URL],
                    use_ssl=False,
                    verify_certs=False,
                    ssl_show_warn=False
                )
            else:
                client = OpenSearch(
                    hosts=[settings.OPENSEARCH_URL],
                    http_auth=(settings.OPENSEARCH_USERNAME, settings.OPENSEARCH_PASSWORD),
                    use_ssl=settings.OPENSEARCH_URL.startswith('https'),
                    verify_certs=False,
                    ssl_show_warn=False
                )
            return client
        except Exception as e:
            return None
    
    def _setup_patterns(self):
        """Setup spaCy patterns for query classification."""
        self.matcher = Matcher(self.nlp.vocab)
        
        # Metadata patterns
        metadata_patterns = [
            # Temporal patterns
            [{"LOWER": {"IN": ["last", "recent", "yesterday", "today", "this"]}}],
            [{"LOWER": "files"}, {"LOWER": "from"}],
            [{"LOWER": "modified"}, {"LOWER": "when"}],
            [{"LOWER": "created"}, {"LOWER": "when"}],
            [{"LOWER": "opened"}, {"LOWER": "when"}],
            [{"LOWER": "viewed"}, {"LOWER": "when"}],
            
            # Relationship patterns
            [{"LOWER": "who"}, {"LOWER": {"IN": ["shared", "owns", "created", "has"]}}],
            [{"LOWER": "shared"}, {"LOWER": "by"}],
            [{"LOWER": "owned"}, {"LOWER": "by"}],
            [{"LOWER": "permissions"}],
            [{"LOWER": "access"}],
            
            # Property patterns
            [{"LOWER": {"IN": ["largest", "smallest", "biggest"]}}],
            [{"LOWER": "how"}, {"LOWER": "many"}],
            [{"LOWER": "count"}, {"LOWER": "of"}],
            [{"LOWER": "file"}, {"LOWER": "type"}],
            [{"LOWER": "file"}, {"LOWER": "size"}],
            [{"LOWER": "number"}, {"LOWER": "of"}],
            
            # General metadata patterns
            [{"LOWER": "show"}, {"LOWER": "all"}],
            [{"LOWER": "list"}, {"LOWER": "files"}],
            [{"LOWER": "metadata"}],
            [{"LOWER": "properties"}],
            [{"LOWER": "structure"}],
            [{"LOWER": "organization"}],
        ]
        
        # Content patterns
        content_patterns = [
            # Content patterns
            [{"LOWER": "what"}, {"LOWER": "does"}],
            [{"LOWER": "what"}, {"LOWER": "is"}],
            [{"LOWER": "what"}, {"LOWER": "are"}],
            [{"LOWER": "explain"}],
            [{"LOWER": "summarize"}],
            [{"LOWER": "find"}, {"LOWER": "information"}],
            [{"LOWER": "key"}, {"LOWER": "points"}],
            [{"LOWER": "topics"}],
            [{"LOWER": "content"}],
            [{"LOWER": "says"}, {"LOWER": "about"}],
            [{"LOWER": "discusses"}],
            [{"LOWER": "mentions"}],
            [{"LOWER": "analyze"}],
            [{"LOWER": "describe"}],
            [{"LOWER": "details"}],
            [{"LOWER": "facts"}],
            [{"LOWER": "data"}],
        ]
        
        # Add patterns to matcher
        for pattern in metadata_patterns:
            self.matcher.add("METADATA", [pattern])
        for pattern in content_patterns:
            self.matcher.add("CONTENT", [pattern])
    
    def classify_query(self, query: str) -> QueryClassification:
        """
        Classify a query as either 'content' or 'metadata' using spaCy NLP.
        
        Args:
            query (str): User's question to classify
            
        Returns:
            QueryClassification: Classification result with confidence and reasoning
        """
        try:
            doc = self.nlp(query.lower())
            
            # Get matches
            matches = self.matcher(doc)
            
            # Count matches by type
            metadata_score = sum(1 for match_id, start, end in matches 
                               if self.nlp.vocab.strings[match_id] == "METADATA")
            content_score = sum(1 for match_id, start, end in matches 
                              if self.nlp.vocab.strings[match_id] == "CONTENT")
            
            # Additional keyword scoring
            metadata_keywords = [
                "file", "files", "document", "documents", "size", "date", "time",
                "when", "who", "shared", "owned", "created", "modified", "opened",
                "viewed", "count", "number", "type", "format", "properties",
                "metadata", "structure", "organization", "permissions", "access"
            ]
            
            content_keywords = [
                "what", "explain", "information", "content", "topic", "topics",
                "summarize", "analyze", "describe", "details", "facts", "data",
                "says", "discusses", "mentions", "key", "points", "conclusion",
                "method", "process", "steps", "procedure", "how", "why"
            ]
            
            # Score based on keyword presence
            for token in doc:
                if token.text in metadata_keywords:
                    metadata_score += 0.3
                if token.text in content_keywords:
                    content_score += 0.3
            
            # Determine classification
            if metadata_score > content_score:
                classification = "metadata"
                confidence = min(metadata_score / (metadata_score + content_score + 1), 0.95)
                reasoning = f"spaCy pattern matching: metadata score {metadata_score:.1f}, content score {content_score:.1f}"
                query_type = self._classify_metadata_query_type(doc)
            else:
                classification = "content"
                confidence = min(content_score / (metadata_score + content_score + 1), 0.95)
                reasoning = f"spaCy pattern matching: content score {content_score:.1f}, metadata score {metadata_score:.1f}"
                query_type = None
            
            # If both scores are 0, use fallback classification
            if metadata_score == 0 and content_score == 0:
                print("⚠️ spaCy returned 0 scores, using fallback classification")
                return self._fallback_classification(query)
            
            return QueryClassification(
                classification=classification,
                confidence=confidence,
                reasoning=reasoning,
                query_type=query_type
            )
            
        except Exception as e:
            return self._fallback_classification(query)
    
    def _classify_metadata_query_type(self, doc: Doc) -> str:
        """Classify the specific type of metadata query (temporal, relationship, etc.)."""
        text = doc.text.lower()
        
        # Revision queries
        revision_keywords = [
            "revision", "version", "change", "changed", "modified", "updated",
            "what changed", "what was changed", "difference", "diff", "compare",
            "from version", "to version", "between versions", "revision history"
        ]
        if any(keyword in text for keyword in revision_keywords):
            return "revision"
        
        # Temporal queries
        temporal_keywords = [
            "last week", "yesterday", "today", "recent", "recently", 
            "this month", "this week", "modified", "opened", "viewed", 
            "created", "when", "date", "time"
        ]
        if any(keyword in text for keyword in temporal_keywords):
            return "temporal"
        
        # Relationship queries
        relationship_keywords = [
            "shared by", "owned by", "created by", "who", "person", 
            "user", "team", "permissions", "access"
        ]
        if any(keyword in text for keyword in relationship_keywords):
            return "relationship"
        
        # Property queries
        property_keywords = [
            "largest", "smallest", "type", "format", "size", "biggest", 
            "how many", "count", "number of", "file type"
        ]
        if any(keyword in text for keyword in property_keywords):
            return "property"
        
        # Content search queries
        content_keywords = [
            "contains", "about", "topic", "subject", "find", "search"
        ]
        if any(keyword in text for keyword in content_keywords):
            return "content_search"
        
        return "general"
    
    def _fallback_classification(self, query: str) -> QueryClassification:
        """Fallback classification using simple keyword matching when spaCy fails."""
        query_lower = query.lower()
        
        metadata_keywords = [
            'how many', 'count', 'number of', 'files', 'documents',
            'file type', 'file size', 'file name', 'file format',
            'when', 'date', 'time', 'modified', 'created', 'processed',
            'recent', 'oldest', 'newest', 'largest', 'smallest',
            'show me all', 'list', 'what files', 'which files',
            'file properties', 'metadata', 'structure', 'organization'
        ]
        
        content_keywords = [
            'what does', 'what is', 'what are', 'find', 'search',
            'information about', 'data about', 'content', 'says',
            'explain', 'describe', 'analyze', 'summarize', 'key points',
            'topics', 'subjects', 'details', 'facts', 'information'
        ]
        
        metadata_score = sum(1 for keyword in metadata_keywords if keyword in query_lower)
        content_score = sum(1 for keyword in content_keywords if keyword in query_lower)
        
        if metadata_score > content_score:
            return QueryClassification(
                classification="metadata",
                confidence=0.7,
                reasoning="Fallback keyword-based classification: query contains metadata-related terms",
                query_type=self._classify_metadata_query_type(self.nlp(query_lower))
            )
        else:
            return QueryClassification(
                classification="content",
                confidence=0.7,
                reasoning="Fallback keyword-based classification: query appears to ask about content"
            )
    
    def handle_metadata_query(self, query: str, user_email: str) -> str:
        """Handle metadata queries using spaCy classification and OpenSearch."""
        try:
            # Classify the query type
            classification = self.classify_query(query)
            
            if classification.classification != "metadata":
                return "This query appears to be about content, not metadata."
            
            # Route to appropriate handler based on query type
            if classification.query_type == "revision":
                return self._handle_revision_query(query, user_email)
            elif classification.query_type == "temporal":
                return self._handle_temporal_query(query, user_email)
            elif classification.query_type == "relationship":
                return self._handle_relationship_query(query, user_email)
            elif classification.query_type == "property":
                return self._handle_property_query(query, user_email)
            else:
                return self._handle_general_metadata_query(query, user_email)
                
        except Exception as e:
            return f"❌ Error processing metadata query: {str(e)}"
    
    def create_metadata_index(self) -> bool:
        """Create the metadata index in OpenSearch with proper mapping."""
        if not self.opensearch_client:
            return False
        
        try:
            # Check if index already exists
            if self.opensearch_client.indices.exists(index=METADATA_INDEX):
                return True
            
            # Define the index mapping
            index_mapping = {
                "mappings": {
                    "properties": {
                        # Basic file info
                        "file_id": {"type": "keyword"},
                        "file_name": {"type": "text"},
                        "file_type": {"type": "keyword"},
                        "mime_type": {"type": "keyword"},
                        
                        # Temporal fields
                        "created_time": {"type": "date"},
                        "modified_time": {"type": "date"},
                        "viewed_by_me_time": {"type": "date"},
                        "processed_time": {"type": "date"},
                        
                        # Size and content info
                        "file_size_mb": {"type": "float"},
                        "page_count": {"type": "integer"},
                        "word_count": {"type": "integer"},
                        "sheet_count": {"type": "integer"},
                        "total_chunks": {"type": "integer"},
                        
                        # Revision-specific fields (for sheets)
                        "revision_id": {"type": "keyword"},
                        "revision_modified_time": {"type": "date"},
                        "revision_size": {"type": "keyword"},
                        "revision_keep_forever": {"type": "boolean"},
                        "revision_original_filename": {"type": "text"},
                        "revision_mime_type": {"type": "keyword"},
                        "last_modifying_user": {
                            "type": "object",
                            "properties": {
                                "displayName": {"type": "text"},
                                "emailAddress": {"type": "keyword"},
                                "permissionId": {"type": "keyword"},
                                "photoLink": {"type": "keyword"}
                            }
                        },
                        
                        # Ownership and permissions
                        "owners": {
                            "type": "nested",
                            "properties": {
                                "displayName": {"type": "text"},
                                "emailAddress": {"type": "keyword"},
                                "permissionId": {"type": "keyword"}
                            }
                        },
                        "permissions": {
                            "type": "nested", 
                            "properties": {
                                "displayName": {"type": "text"},
                                "emailAddress": {"type": "keyword"},
                                "role": {"type": "keyword"},
                                "type": {"type": "keyword"},
                                "permissionId": {"type": "keyword"}
                            }
                        },
                        
                        # File status
                        "trashed": {"type": "boolean"},
                        "starred": {"type": "boolean"},
                        "shared": {"type": "boolean"},
                        
                        # Hierarchical structure
                        "parents": {"type": "keyword"},
                        "folder_path": {"type": "text"},
                        
                        # User context
                        "user_email": {"type": "keyword"},
                        "access_level": {"type": "keyword"},
                        
                        # Additional metadata
                        "web_view_link": {"type": "keyword"},
                        "web_content_link": {"type": "keyword"},
                        "thumbnail_link": {"type": "keyword"},
                        "capabilities": {"type": "object"},
                        "export_links": {"type": "object"},
                        "app_properties": {"type": "object"},
                        "properties": {"type": "object"}
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
            self.opensearch_client.indices.create(
                index=METADATA_INDEX,
                body=index_mapping
            )
            
            return True
            
        except Exception as e:
            return False
    
    def store_metadata_in_opensearch(self, file_metadata: Dict, user_email: str) -> bool:
        """Store file metadata in OpenSearch for querying."""
        if not self.opensearch_client:
            return False
        
        try:
            # Ensure index exists
            if not self.opensearch_client.indices.exists(index=METADATA_INDEX):
                self.create_metadata_index()
            
            success_count = 0
            for file_hash, metadata in file_metadata.items():
                try:
                    # Prepare document for OpenSearch
                    document = {
                        "user_email": user_email,
                        "file_id": metadata.get('file_id', ''),
                        "file_name": metadata.get('file_name', ''),
                        "file_type": metadata.get('file_type', ''),
                        "mime_type": metadata.get('mime_type', ''),
                        "file_size_mb": metadata.get('file_size_mb', 0.0),
                        "page_count": metadata.get('page_count', 0),
                        "word_count": metadata.get('word_count', 0),
                        "sheet_count": metadata.get('sheet_count', 0),
                        "total_chunks": metadata.get('total_chunks', 0),
                        "trashed": metadata.get('trashed', False),
                        "starred": metadata.get('starred', False),
                        "shared": metadata.get('shared', False),
                        "parents": metadata.get('parents', []),
                        "web_view_link": metadata.get('web_view_link', ''),
                        "web_content_link": metadata.get('web_content_link', ''),
                        "thumbnail_link": metadata.get('thumbnail_link', ''),
                        "capabilities": metadata.get('capabilities', {}),
                        "export_links": metadata.get('export_links', {}),
                        "app_properties": metadata.get('app_properties', {}),
                        "properties": metadata.get('properties', {}),
                        
                        # Revision-specific fields
                        "revision_id": metadata.get('revision_id', ''),
                        "revision_size": metadata.get('revision_size', ''),
                        "revision_keep_forever": metadata.get('revision_keep_forever', False),
                        "revision_original_filename": metadata.get('revision_original_filename', ''),
                        "revision_mime_type": metadata.get('revision_mime_type', ''),
                        "last_modifying_user": metadata.get('last_modifying_user', {}),
                        
                        # Nested objects
                        "owners": metadata.get('owners', []),
                        "permissions": metadata.get('permissions', [])
                    }
                    
                    # Handle date fields properly - only include if they have valid values
                    date_fields = {
                        "created_time": metadata.get('created_time'),
                        "modified_time": metadata.get('modified_time'),
                        "viewed_by_me_time": metadata.get('viewed_by_me_time'),
                        "processed_time": metadata.get('processed_time'),
                        "revision_modified_time": metadata.get('revision_modified_time')
                    }
                    
                    for field_name, field_value in date_fields.items():
                        if field_value and field_value.strip():  # Only include non-empty values
                            document[field_name] = field_value
                    
                    # Index the document
                    self.opensearch_client.index(
                        index=METADATA_INDEX,
                        id=file_hash,
                        body=document
                    )
                    
                    success_count += 1
                    
                except Exception as e:
                    continue
            
            return success_count > 0
            
        except Exception as e:
            return False
    
    # Metadata query handlers (simplified versions)
    def _handle_temporal_query(self, query: str, user_email: str) -> str:
        """Handle temporal queries (time-based)"""
        return "Temporal query handling - implementation needed"
    
    def _handle_relationship_query(self, query: str, user_email: str) -> str:
        """Handle relationship queries (people-based)"""
        return "Relationship query handling - implementation needed"
    
    def _handle_property_query(self, query: str, user_email: str) -> str:
        """Handle property queries (file properties)"""
        return "Property query handling - implementation needed"
    
    def _handle_revision_query(self, query: str, user_email: str) -> str:
        """Handle revision-related queries for sheets"""
        return "Revision query handling - implementation needed"
    
    def _handle_general_metadata_query(self, query: str, user_email: str) -> str:
        """Handle general metadata queries using text search"""
        return "General metadata query handling - implementation needed"

# Global instance
spacy_metadata_handler = None

def get_spacy_metadata_handler() -> QueryClassifier:
    """Get or create the global spaCy metadata handler instance"""
    global spacy_metadata_handler
    if spacy_metadata_handler is None:
        spacy_metadata_handler = QueryClassifier()
    return spacy_metadata_handler

def classify_query(query: str) -> dict:
    """
    Classify a query as either 'content' or 'metadata' using spaCy-based approach.
    
    Args:
        query (str): The user's query
        
    Returns:
        dict: Classification result with type and confidence
    """
    try:
        print(f"🔍 [QUERY_CLASSIFIER] Starting classification for query: '{query[:50]}...'")
        
        # Use spaCy-based classification
        spacy_handler = get_spacy_metadata_handler()
        if not spacy_handler:
            print(f"❌ [QUERY_CLASSIFIER] Failed to get spaCy handler, using fallback")
            # Fallback to basic classification
            return fallback_query_classification(query)
        
        print(f"✅ [QUERY_CLASSIFIER] Got spaCy handler, classifying...")
        classification = spacy_handler.classify_query(query)
        
        print(f"✅ [QUERY_CLASSIFIER] Classification result: {classification.classification} (confidence: {classification.confidence:.2f})")
        
        # Convert to the expected format (dictionary)
        result = {
            "classification": classification.classification,
            "confidence": classification.confidence,
            "reasoning": classification.reasoning,
            "query_type": classification.query_type
        }
        
        print(f"✅ [QUERY_CLASSIFIER] Returning result: {result}")
        return result
        
    except Exception as e:
        print(f"❌ [QUERY_CLASSIFIER] Error in classification: {str(e)}")
        # Fallback to original LLM-based classification
        return fallback_query_classification(query)

def fallback_query_classification(query: str) -> dict:
    """
    Fallback classification using keyword matching when LLM classification fails.
    
    Args:
        query (str): User's question to classify
        
    Returns:
        dict: Basic classification result with confidence
    """
    print(f"🔍 [FALLBACK_CLASSIFIER] Starting fallback classification for query: '{query[:50]}...'")
    
    query_lower = query.lower()
    
    # Metadata keywords (queries about file properties, structure, etc.)
    metadata_keywords = [
        'how many', 'count', 'number of', 'file type', 'file size', 'file name', 'file format',
        'when', 'date', 'time', 'modified', 'created', 'processed',
        'recent', 'oldest', 'newest', 'largest', 'smallest',
        'show me all', 'list', 'what files', 'which files',
        'file properties', 'metadata', 'structure', 'organization',
        'file count', 'document count', 'total files', 'file list'
    ]
    
    # Content keywords (queries about actual content, data, information)
    content_keywords = [
        'what does', 'what is', 'what are', 'find', 'search',
        'information about', 'data about', 'content', 'says',
        'explain', 'describe', 'analyze', 'summarize', 'key points',
        'topics', 'subjects', 'details', 'facts', 'information',
        'containing', 'with', 'about', 'reviews', 'products', 'companies',
        'data', 'table', 'sheet', 'spreadsheet'
    ]
    
    metadata_score = sum(1 for keyword in metadata_keywords if keyword in query_lower)
    content_score = sum(1 for keyword in content_keywords if keyword in query_lower)
    
    # Special handling for queries that ask about content within files
    if 'containing' in query_lower or 'with' in query_lower:
        content_score += 2  # Boost content score for "containing" queries
    
    # Special handling for specific content types
    if any(word in query_lower for word in ['reviews', 'products', 'companies', 'data', 'table']):
        content_score += 1  # Boost content score for specific content types
    
    print(f"📊 [FALLBACK_CLASSIFIER] Scores - Metadata: {metadata_score}, Content: {content_score}")
    
    if metadata_score > content_score:
        result = {
            "classification": "metadata",
            "confidence": 0.7,
            "reasoning": "Keyword-based classification: query contains metadata-related terms"
        }
        print(f"✅ [FALLBACK_CLASSIFIER] Classified as METADATA")
    else:
        result = {
            "classification": "content",
            "confidence": 0.7,
            "reasoning": "Keyword-based classification: query appears to ask about content"
        }
        print(f"✅ [FALLBACK_CLASSIFIER] Classified as CONTENT")
    
    print(f"✅ [FALLBACK_CLASSIFIER] Returning result: {result}")
    return result 