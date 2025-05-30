from abc import ABC, abstractmethod
from typing import List, Set, Dict, Any
import re
import hashlib

class BaseContentFilter(ABC):
    """Abstract base class for document content filters."""
    
    def __init__(self):
        self.stats = {
            'total_chunks_processed': 0,
            'chunks_filtered_out': 0,
            'chunks_cleaned': 0,
            'boilerplate_removed': 0,
            'duplicates_removed': 0
        }
    
    @abstractmethod
    def should_skip_chunk(self, text: str) -> bool:
        """Determine if chunk should be skipped entirely."""
        pass
    
    @abstractmethod
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        pass
    
    def filter_page_content(self, page_text: str) -> str:
        """Filter content from a single page."""
        if not page_text or len(page_text.strip()) < 100:
            return ""
        
        cleaned_text = self.clean_text(page_text)
        
        if self.should_skip_chunk(cleaned_text):
            return ""
        
        return cleaned_text
    
    def get_content_hash(self, text: str) -> str:
        """
        Generate hash for content deduplication.
        
        Args:
            text: Text content
            
        Returns:
            MD5 hash of normalized content
        """
        # Normalize text for hashing
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        normalized = re.sub(r'[^\w\s]', '', normalized)  # Remove punctuation
        
        # Remove common variations that shouldn't affect deduplication
        normalized = re.sub(r'\b(?:page|pp?\.?)\s*\d+\b', '', normalized)
        normalized = re.sub(r'\b(?:figure|fig\.?|table|tab\.?)\s*\d+\b', '', normalized)
        
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get_filtering_stats(self) -> Dict[str, Any]:
        """Get filtering statistics."""
        total_processed = self.stats['total_chunks_processed']
        filter_rate = (self.stats['chunks_filtered_out'] / total_processed * 100) if total_processed > 0 else 0
        
        return {
            **self.stats,
            'filter_rate_percent': round(filter_rate, 2)
        } 