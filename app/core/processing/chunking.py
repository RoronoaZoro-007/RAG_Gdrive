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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
    
    # Generate embeddings for all sentences using parallel processing
    print(f"🚀 [PARALLEL] Generating embeddings for {len(sentences)} sentences...")
    
    # Thread-local storage for embeddings
    _local = threading.local()
    
    def _get_embeddings():
        """Get thread-local embeddings instance."""
        if not hasattr(_local, 'embeddings'):
            _local.embeddings = embeddings
        return _local.embeddings
    
    def _embed_batch(sentence_batch):
        """Embed a batch of sentences (for parallel processing)."""
        try:
            print(f"🔄 [PARALLEL] Embedding batch of {len(sentence_batch)} sentences")
            batch_embeddings = _get_embeddings().embed_documents(sentence_batch)
            print(f"✅ [PARALLEL] Successfully embedded batch of {len(sentence_batch)} sentences")
            return batch_embeddings
        except Exception as e:
            print(f"❌ [PARALLEL] Error embedding batch: {str(e)}")
            return []
    
    # Split sentences into batches for parallel processing
    batch_size = max(1, len(sentences) // 4)  # 4 parallel workers
    sentence_batches = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
    
    sentence_embeddings = []
    with ThreadPoolExecutor(max_workers=min(4, len(sentence_batches))) as executor:
        # Submit all embedding tasks
        embedding_futures = {
            executor.submit(_embed_batch, batch): batch 
            for batch in sentence_batches
        }
        
        # Collect results as they complete
        for future in as_completed(embedding_futures):
            batch = embedding_futures[future]
            try:
                batch_embeddings = future.result()
                sentence_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"❌ [PARALLEL] Exception in embedding batch: {str(e)}")
    
    print(f"✅ [PARALLEL] Completed embedding generation for {len(sentence_embeddings)} sentences")
    
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
    
    # Post-process: merge very similar adjacent chunks using parallel processing
    if len(chunks) > 1:
        print(f"🚀 [PARALLEL] Starting parallel chunk merging for {len(chunks)} chunks...")
        
        def _merge_chunk_pair(chunk_pair):
            """Merge a pair of chunks if they are similar (for parallel processing)."""
            try:
                i, (chunk1, chunk2) = chunk_pair
                print(f"🔄 [PARALLEL] Checking similarity for chunk pair {i}")
                
                chunk1_embedding = _get_embeddings().embed_documents([chunk1])[0]
                chunk2_embedding = _get_embeddings().embed_documents([chunk2])[0]
                similarity = cosine_similarity([chunk1_embedding], [chunk2_embedding])[0][0]
                
                if similarity > appending_threshold and len(chunk1) + len(chunk2) <= chunk_size * 1.2:
                    merged_chunk = chunk1 + " " + chunk2
                    print(f"✅ [PARALLEL] Merged chunk pair {i} (similarity: {similarity:.3f})")
                    return i, merged_chunk, True
                else:
                    print(f"❌ [PARALLEL] Chunk pair {i} not merged (similarity: {similarity:.3f})")
                    return i, None, False
            except Exception as e:
                print(f"❌ [PARALLEL] Error merging chunk pair {i}: {str(e)}")
                return i, None, False
        
        # Create chunk pairs for parallel processing
        chunk_pairs = [(i, (chunks[i], chunks[i + 1])) for i in range(len(chunks) - 1)]
        
        merged_chunks = {}
        with ThreadPoolExecutor(max_workers=min(4, len(chunk_pairs))) as executor:
            # Submit all merging tasks
            merge_futures = {
                executor.submit(_merge_chunk_pair, pair): pair 
                for pair in chunk_pairs
            }
            
            # Collect results as they complete
            for future in as_completed(merge_futures):
                pair = merge_futures[future]
                try:
                    i, merged_chunk, should_merge = future.result()
                    if should_merge:
                        merged_chunks[i] = merged_chunk
                except Exception as e:
                    print(f"❌ [PARALLEL] Exception in chunk merging: {str(e)}")
        
        # Reconstruct final chunks
        final_chunks = []
        i = 0
        while i < len(chunks):
            if i in merged_chunks:
                final_chunks.append(merged_chunks[i])
                i += 2  # Skip next chunk
            else:
                final_chunks.append(chunks[i])
                i += 1
        
        chunks = final_chunks
        print(f"✅ [PARALLEL] Chunk merging completed: {len(chunks)} final chunks")
    
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