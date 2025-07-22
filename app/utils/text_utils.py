"""
Text utility functions.
Handles common text processing operations.
"""

import re
from typing import List

def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', text)
    
    return text

def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from text.
    
    Args:
        text (str): Text to extract URLs from
        
    Returns:
        List[str]: List of URLs found in the text
    """
    url_pattern = r'https?://\S+'
    urls = re.findall(url_pattern, text)
    return list(set(urls))  # Remove duplicates

def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text (str): Text to truncate
        max_length (int): Maximum length
        suffix (str): Suffix to add if truncated
        
    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def count_words(text: str) -> int:
    """
    Count words in text.
    
    Args:
        text (str): Text to count words in
        
    Returns:
        int: Number of words
    """
    if not text:
        return 0
    
    words = text.split()
    return len(words)

def count_characters(text: str) -> int:
    """
    Count characters in text.
    
    Args:
        text (str): Text to count characters in
        
    Returns:
        int: Number of characters
    """
    if not text:
        return 0
    
    return len(text) 