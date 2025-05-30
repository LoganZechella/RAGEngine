"""
Scientific Content Filter for RAG Engine.
Filters out redundant and low-value content from scientific documents.
Optimized for scientific reports with high redundancy and structured format.
"""

import re
import hashlib
from typing import List, Set, Dict, Any
from loguru import logger
from .content_filter_base import BaseContentFilter


class ScientificContentFilter(BaseContentFilter):
    """
    Filters out redundant and low-value content from scientific documents.
    Designed specifically for scientific reports with predictable structure.
    """
    
    def __init__(self):
        """Initialize the content filter with patterns for scientific documents."""
        super().__init__()
        
        # Patterns for content that should be skipped entirely
        self.skip_patterns = [
            r'^Page \d+ of \d+$',                    # Page numbers
            r'^Experiment \d+ Report - Page \d+',   # Headers
            r'^APPENDIX [A-Z]\d*$',                  # Appendix headers only
            r'^\s*$',                                # Empty lines
            r'^.*?Report.*?Page \d+ of \d+.*?$',    # Report headers
            r'^Table of Contents$',                  # TOC headers
            r'^\d+\s*$',                            # Standalone numbers
        ]
        
        # Boilerplate phrases that indicate low-value content
        self.boilerplate_phrases = {
            "The protocol is attached as Appendix",
            "Results as reported by Mayo",
            "Table R1-", "Figure R1-",  # Table/figure references
            "See Appendix",
            "Refer to Section",
            "As shown in Figure",
            "Page intentionally left blank",
            "End of Report",
            "Confidential and Proprietary"
        }
        
        # Patterns for repetitive headers/footers
        self.header_footer_patterns = [
            r'^.*?Confidential.*?$',
            r'^.*?Proprietary.*?$',
            r'^\w+\s+\d{4}\s*$',  # Month Year
            r'^Report\s+\d+.*?$',
            r'^Version\s+\d+.*?$'
        ]
        
        # Minimum content thresholds
        self.min_chunk_length = 50
        self.min_meaningful_words = 5
        
        # Track filtering statistics
        self.stats = {
            'total_chunks_processed': 0,
            'chunks_filtered_out': 0,
            'chunks_cleaned': 0,
            'boilerplate_removed': 0,
            'duplicates_removed': 0
        }
    
    def should_skip_chunk(self, text: str) -> bool:
        """
        Determine if a chunk should be skipped entirely.
        
        Args:
            text: The text content to evaluate
            
        Returns:
            True if the chunk should be skipped, False otherwise
        """
        if not text or len(text.strip()) < 50:  # Too short to be meaningful
            return True
        
        text_stripped = text.strip()
        
        # Check skip patterns
        for pattern in self.skip_patterns:
            if re.match(pattern, text_stripped, re.IGNORECASE):
                logger.debug(f"Skipping chunk matching pattern: {pattern}")
                return True
        
        # Check for header/footer patterns
        for pattern in self.header_footer_patterns:
            if re.match(pattern, text_stripped, re.IGNORECASE):
                if len(text_stripped) < 200:  # Only skip if it's short
                    logger.debug(f"Skipping header/footer: {text_stripped[:50]}...")
                    return True
        
        # Check boilerplate content
        for phrase in self.boilerplate_phrases:
            if phrase in text and len(text) < 200:
                logger.debug(f"Skipping boilerplate chunk: {phrase}")
                return True
        
        # Skip if mostly numbers/tables without meaningful text
        words = text.split()
        if len(words) > 10:
            numeric_ratio = sum(1 for word in words if re.match(r'^\d+\.?\d*$', word)) / len(words)
            if numeric_ratio > 0.7:  # More than 70% numbers
                logger.debug("Skipping chunk with high numeric content")
                return True
        
        # Skip repetitive content (like repeated headers)
        lines = text.split('\n')
        if len(lines) > 3:
            unique_lines = set(line.strip() for line in lines if line.strip())
            if len(unique_lines) / len([l for l in lines if l.strip()]) < 0.5:
                logger.debug("Skipping repetitive content")
                return True
        
        return False
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text content
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove isolated page numbers
        text = re.sub(r'(?:^|\s)Page \d+(?:\s|$)', ' ', text)
        
        # Remove common artifacts
        text = re.sub(r'(?:^|\s)PAGE \d+(?:\s|$)', ' ', text)
        text = re.sub(r'\bpp?\.\s*\d+\b', '', text)  # Remove page references
        
        # Normalize quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace('–', '-').replace('—', '-')
        
        # Remove multiple consecutive periods/dots (often OCR artifacts)
        text = re.sub(r'\.{3,}', '...', text)
        
        # Clean up spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])\s*([,.!?;:])', r'\1 \2', text)
        
        return text.strip()
    
    def _is_mostly_tabular_without_context(self, text: str) -> bool:
        """
        Check if text is mostly tabular data without meaningful context.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if text is mostly tabular without context
        """
        lines = text.split('\n')
        
        # Count lines that look like table rows (mostly numbers/short values)
        table_like_lines = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line is mostly numbers, pipes, or short values
            words = line.split()
            if len(words) > 0:
                numeric_or_short = sum(1 for word in words 
                                     if word.replace('.', '').replace(',', '').isdigit() 
                                     or len(word) <= 3)
                if numeric_or_short / len(words) > 0.7:  # 70% numeric or very short
                    table_like_lines += 1
        
        # If more than 60% of lines are table-like, consider it tabular
        if len(lines) > 0 and table_like_lines / len(lines) > 0.6:
            # Check if there's meaningful context (sentences with verbs/descriptive words)
            meaningful_content = re.findall(r'\b(?:shows?|indicates?|demonstrates?|reveals?|suggests?|contains?|includes?|represents?)\b', text, re.IGNORECASE)
            if len(meaningful_content) < 2:  # Less than 2 meaningful context words
                return True
        
        return False
    
    def extract_table_summary(self, table_text: str) -> str:
        """
        Create concise summary of tabular data for large tables.
        
        Args:
            table_text: Raw table text
            
        Returns:
            Summarized table content or original if small
        """
        lines = [line.strip() for line in table_text.split('\n') if line.strip()]
        
        if len(lines) <= 5:  # Small table, keep as-is
            return table_text
        
        # Extract header if present
        header = lines[0] if lines else ""
        
        # Count data rows
        data_rows = len(lines) - 1 if len(lines) > 1 else 0
        
        # For large tables, create summary
        if data_rows > 10:
            summary = f"Table: {header}\n"
            summary += f"Contains {data_rows} data entries with experimental results.\n"
            
            # Include first few rows as examples
            if len(lines) > 3:
                summary += "Sample entries:\n"
                for line in lines[1:4]:  # First 3 data rows
                    summary += f"  {line}\n"
                
            if data_rows > 3:
                summary += f"  ... and {data_rows - 3} more entries"
            
            return summary
        
        return table_text  # Keep medium tables as-is
    
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
    
    def is_duplicate_content(self, text: str, seen_hashes: Set[str]) -> bool:
        """
        Check if content is a duplicate based on hash.
        
        Args:
            text: Text content to check
            seen_hashes: Set of previously seen content hashes
            
        Returns:
            True if content is duplicate, False otherwise
        """
        content_hash = self.get_content_hash(text)
        
        if content_hash in seen_hashes:
            self.stats['duplicates_removed'] += 1
            return True
        
        seen_hashes.add(content_hash)
        return False
    
    def filter_page_content(self, page_text: str) -> str:
        """
        Filter and clean content from a single page.
        
        Args:
            page_text: Raw page content
            
        Returns:
            Filtered and cleaned page content, or empty string if page should be skipped
        """
        if not page_text or len(page_text.strip()) < 100:
            return ""
        
        # Clean the page content
        cleaned_text = self.clean_text(page_text)
        
        # Check if page should be skipped after cleaning
        if self.should_skip_chunk(cleaned_text):
            return ""
        
        return cleaned_text
    
    def get_filtering_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the filtering process.
        
        Returns:
            Dictionary with filtering statistics
        """
        total_processed = self.stats['total_chunks_processed']
        if total_processed > 0:
            filter_rate = (self.stats['chunks_filtered_out'] / total_processed) * 100
        else:
            filter_rate = 0
        
        return {
            **self.stats,
            'filter_rate_percent': round(filter_rate, 2),
            'content_reduction_estimate': f"{round(filter_rate + (self.stats['chunks_cleaned'] / max(total_processed, 1)) * 10, 1)}%"
        }
    
    def reset_stats(self):
        """Reset filtering statistics."""
        self.stats = {
            'total_chunks_processed': 0,
            'chunks_filtered_out': 0,
            'chunks_cleaned': 0,
            'boilerplate_removed': 0,
            'duplicates_removed': 0
        }

    def should_merge_chunks(self, chunk1_text: str, chunk2_text: str) -> bool:
        """
        Determine if two consecutive chunks should be merged.
        
        Args:
            chunk1_text: First chunk text
            chunk2_text: Second chunk text
            
        Returns:
            True if chunks should be merged
        """
        # Don't merge if either is too long
        if len(chunk1_text.split()) > 400 or len(chunk2_text.split()) > 400:
            return False
        
        # Merge if second chunk is very short and seems like a continuation
        if len(chunk2_text.split()) < 20:
            # Check if it starts with lowercase (likely continuation)
            first_word = chunk2_text.strip().split()[0] if chunk2_text.strip() else ""
            if first_word and first_word[0].islower():
                return True
            
            # Check if it's just a list item or incomplete sentence
            if re.match(r'^[•\-\*\d+\.]', chunk2_text.strip()):
                return True
        
        return False 