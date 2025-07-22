"""
Semantic text chunking for document processing.
Handles intelligent text splitting using embeddings and semantic analysis.
"""

import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.constants import (
    DEFAULT_CHUNK_SIZE, 
    DEFAULT_CHUNK_OVERLAP, 
    DEFAULT_INITIAL_THRESHOLD, 
    DEFAULT_APPENDING_THRESHOLD
)

def semantic_chunk_text(text, embeddings, chunk_size=1500, overlap=300, 
                       initial_threshold=0.6, appending_threshold=0.8):
    """
    Split text semantically using embeddings to find natural breakpoints.
    
    Args:
        text (str): Text to split into chunks
        embeddings: Embedding model for semantic analysis
        chunk_size (int): Target size for each chunk
        overlap (int): Overlap between chunks
        initial_threshold (float): Similarity threshold for chunk boundaries
        appending_threshold (float): Threshold for merging similar chunks
        
    Returns:
        list: List of semantically coherent text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    # Split into sentences first using multiple approaches
    sentences = []
    
    # Try multiple sentence splitting approaches
    # 1. Split by periods followed by space and capital letter
    period_splits = re.split(r'(?<=\.)\s+(?=[A-Z])', text)
    
    # 2. Split by other sentence endings
    sentence_endings = re.split(r'(?<=[.!?])\s+', text)
    
    # Use the approach that gives more reasonable sentence lengths
    if len(period_splits) > len(sentence_endings) and all(len(s) > 10 for s in period_splits):
        sentences = period_splits
    else:
        sentences = sentence_endings
    
    # Clean up sentences
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    if len(sentences) <= 1:
        # Fallback to recursive splitting if no sentence breaks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " ", ""],
            keep_separator=True
        )
        return text_splitter.split_text(text)
    
    # Generate embeddings for all sentences
    print(f"Generating embeddings for {len(sentences)} sentences...")
    sentence_embeddings = embeddings.embed_documents(sentences)
    
    # Calculate cosine similarity between consecutive sentences
    chunks = []
    current_chunk = []
    current_length = 0
    
    for i, (sentence, embedding) in enumerate(zip(sentences, sentence_embeddings)):
        sentence_length = len(sentence)
        
        # Check if adding this sentence would exceed chunk size
        if current_length + sentence_length > chunk_size and current_chunk:
            # Finalize current chunk
            chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap (last few sentences)
            overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
            current_chunk = overlap_sentences.copy()
            current_length = sum(len(s) for s in current_chunk)
        
        # If this is the first sentence or we have a current chunk, check similarity
        if current_chunk:
            # Calculate similarity with the last sentence in current chunk
            last_sentence_idx = i - 1
            if last_sentence_idx >= 0:
                last_embedding = sentence_embeddings[last_sentence_idx]
                similarity = cosine_similarity([last_embedding], [embedding])[0][0]
                
                # If similarity is low, start a new chunk (semantic breakpoint)
                if similarity < initial_threshold and current_length > chunk_size * 0.3:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                    continue
        
        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_length += sentence_length
    
    # Add the final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # Post-process: merge very similar adjacent chunks
    if len(chunks) > 1:
        final_chunks = []
        i = 0
        while i < len(chunks):
            if i < len(chunks) - 1:
                # Check if we can merge with next chunk
                chunk1_embedding = embeddings.embed_documents([chunks[i]])[0]
                chunk2_embedding = embeddings.embed_documents([chunks[i + 1]])[0]
                similarity = cosine_similarity([chunk1_embedding], [chunk2_embedding])[0][0]
                
                if similarity > appending_threshold and len(chunks[i]) + len(chunks[i + 1]) <= chunk_size * 1.2:
                    # Merge chunks
                    merged_chunk = chunks[i] + " " + chunks[i + 1]
                    final_chunks.append(merged_chunk)
                    i += 2  # Skip next chunk
                else:
                    final_chunks.append(chunks[i])
                    i += 1
            else:
                final_chunks.append(chunks[i])
                i += 1
        
        chunks = final_chunks
    
    return chunks

class SemanticChunker:
    """Handles semantic text chunking for document processing."""
    
    def __init__(self, embeddings):
        """
        Initialize the semantic chunker.
        
        Args:
            embeddings: Embedding model for semantic analysis
        """
        self.embeddings = embeddings
    
    def chunk_text(self, text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, 
                   overlap: int = DEFAULT_CHUNK_OVERLAP, 
                   initial_threshold: float = DEFAULT_INITIAL_THRESHOLD, 
                   appending_threshold: float = DEFAULT_APPENDING_THRESHOLD) -> list:
        """
        Split text semantically using embeddings to find natural breakpoints.
        
        Args:
            text (str): Text to split into chunks
            chunk_size (int): Target size for each chunk
            overlap (int): Overlap between chunks
            initial_threshold (float): Similarity threshold for chunk boundaries
            appending_threshold (float): Threshold for merging similar chunks
            
        Returns:
            list: List of semantically coherent text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        # Split into sentences first using multiple approaches
        sentences = []
        
        # Try multiple sentence splitting approaches
        # 1. Split by periods followed by space and capital letter
        period_splits = re.split(r'(?<=\.)\s+(?=[A-Z])', text)
        
        # 2. Split by other sentence endings
        sentence_endings = re.split(r'(?<=[.!?])\s+', text)
        
        # Use the approach that gives more reasonable sentence lengths
        if len(period_splits) > len(sentence_endings) and all(len(s) > 10 for s in period_splits):
            sentences = period_splits
        else:
            sentences = sentence_endings
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        if len(sentences) <= 1:
            # Fallback to recursive splitting if no sentence breaks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " ", ""],
                keep_separator=True
            )
            return text_splitter.split_text(text)
        
        # Generate embeddings for all sentences
        print(f"Generating embeddings for {len(sentences)} sentences...")
        sentence_embeddings = self.embeddings.embed_documents(sentences)
        
        # Calculate cosine similarity between consecutive sentences
        chunks = []
        current_chunk = []
        current_length = 0
        
        for i, (sentence, embedding) in enumerate(zip(sentences, sentence_embeddings)):
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if current_length + sentence_length > chunk_size and current_chunk:
                # Finalize current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap (last few sentences)
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_sentences.copy()
                current_length = sum(len(s) for s in current_chunk)
            
            # If this is the first sentence or we have a current chunk, check similarity
            if current_chunk:
                # Calculate similarity with the last sentence in current chunk
                last_sentence_idx = i - 1
                if last_sentence_idx >= 0:
                    last_embedding = sentence_embeddings[last_sentence_idx]
                    similarity = cosine_similarity([last_embedding], [embedding])[0][0]
                    
                    # If similarity is low, start a new chunk (semantic breakpoint)
                    if similarity < initial_threshold and current_length > chunk_size * 0.3:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = [sentence]
                        current_length = sentence_length
                        continue
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add the final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Post-process: merge very similar adjacent chunks
        if len(chunks) > 1:
            final_chunks = []
            i = 0
            while i < len(chunks):
                if i < len(chunks) - 1:
                    # Check if we can merge with next chunk
                    chunk1_embedding = self.embeddings.embed_documents([chunks[i]])[0]
                    chunk2_embedding = self.embeddings.embed_documents([chunks[i + 1]])[0]
                    similarity = cosine_similarity([chunk1_embedding], [chunk2_embedding])[0][0]
                    
                    if similarity > appending_threshold and len(chunks[i]) + len(chunks[i + 1]) <= chunk_size * 1.2:
                        # Merge chunks
                        merged_chunk = chunks[i] + " " + chunks[i + 1]
                        final_chunks.append(merged_chunk)
                        i += 2  # Skip next chunk
                    else:
                        final_chunks.append(chunks[i])
                        i += 1
                else:
                    final_chunks.append(chunks[i])
                    i += 1
            
            chunks = final_chunks
        
        return chunks 