"""
Scientific Content Filter for RAG Engine.
Filters out redundant and low-value content from scientific documents.
Optimized for scientific reports with high redundancy and structured format.
"""

import re
import hashlib
from typing import List, Set, Dict, Any
from loguru import logger


class ScientificContentFilter:
    """
    Filters out redundant and low-value content from scientific documents.
    Designed specifically for scientific reports with predictable structure.
    """
    
    def __init__(self):
        """Initialize the content filter with patterns for scientific documents."""
        
        # Patterns to skip entirely
        self.skip_patterns = [
            r'^Page \d+ of \d+$',                    # Page numbers
            r'^Experiment \d+ Report - Page \d+',   # Report headers
            r'^APPENDIX [A-Z]\d*$',                  # Appendix headers only
            r'^\s*$',                                # Empty lines
            r'^.*?Report.*?Page \d+ of \d+.*?$',    # Report headers with page numbers
            r'^Table of Contents$',                  # TOC headers
            r'^References$',                         # Reference section headers only
            r'^Bibliography$',                       # Bibliography headers only
            r'^\d+\s*$',                            # Standalone numbers
            r'^[A-Z\s]+$',                          # All caps headers without content
        ]
        
        # Boilerplate phrases that indicate low-value content
        self.boilerplate_phrases = {
            "The protocol is attached as Appendix",
            "Results as reported by Mayo",
            "Table R1-", "Figure R1-",              # Table/figure references without context
            "See Appendix for details",
            "Data not shown",
            "Results pending",
            "To be determined",
            "Under investigation",
            "Contact laboratory for details",
        }
        
        # Patterns for header/footer content
        self.header_footer_patterns = [
            r'^.*?Confidential.*?$',
            r'^.*?Internal Use Only.*?$',
            r'^.*?Draft.*?$',
            r'^.*?Version \d+\.\d+.*?$',
            r'^.*?Copyright.*?$',
            r'^.*?All rights reserved.*?$',
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
            text: Text content to evaluate
            
        Returns:
            True if chunk should be skipped, False otherwise
        """
        if not text or len(text.strip()) < self.min_chunk_length:
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
                logger.debug(f"Skipping header/footer content: {pattern}")
                return True
        
        # Check boilerplate content
        for phrase in self.boilerplate_phrases:
            if phrase in text and len(text) < 200:  # Short text with boilerplate
                logger.debug(f"Skipping boilerplate content: {phrase}")
                self.stats['boilerplate_removed'] += 1
                return True
        
        # Check if content is mostly numbers/tables without context
        if self._is_mostly_tabular_without_context(text):
            logger.debug("Skipping tabular content without context")
            return True
        
        # Check word count
        words = text.split()
        if len(words) < self.min_meaningful_words:
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
        
        original_length = len(text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove isolated page numbers
        text = re.sub(r'(?:^|\s)Page \d+(?:\s|$)', ' ', text)
        
        # Remove report headers embedded in text
        text = re.sub(r'Experiment \d+ Report - Page \d+', '', text)
        
        # Normalize quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace('–', '-').replace('—', '-')
        
        # Remove multiple consecutive periods
        text = re.sub(r'\.{3,}', '...', text)
        
        # Clean up spacing around punctuation
        text = re.sub(r'\s+([,.;:!?])', r'\1', text)
        text = re.sub(r'([,.;:!?])\s+', r'\1 ', text)
        
        # Remove header/footer content
        for pattern in self.header_footer_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Final cleanup
        text = text.strip()
        
        if len(text) < original_length * 0.8:  # Significant cleaning occurred
            self.stats['chunks_cleaned'] += 1
        
        return text
    
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
        Create concise summary of tabular data instead of full content.
        
        Args:
            table_text: Full table text
            
        Returns:
            Summarized table description
        """
        lines = [line.strip() for line in table_text.split('\n') if line.strip()]
        
        if not lines:
            return ""
        
        # Extract header if present
        header = lines[0] if lines else ""
        
        # Count data rows
        data_rows = len(lines) - 1 if len(lines) > 1 else 0
        
        # Look for table caption or title
        caption_match = re.search(r'Table\s+\d+[:\-]?\s*(.+)', table_text, re.IGNORECASE)
        caption = caption_match.group(1) if caption_match else ""
        
        # Create summary instead of full table for large tables
        if data_rows > 5:  # Large table
            summary = f"Table: {caption or header}\n"
            summary += f"Contains {data_rows} data entries with experimental results."
            
            # Add key column information if detectable
            if '|' in header or '\t' in header:
                columns = len(header.split('|' if '|' in header else '\t'))
                summary += f" Data organized in {columns} columns."
            
            return summary
        else:
            # Keep small tables as-is but clean them
            return self.clean_text(table_text)
    
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