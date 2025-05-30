import re
from typing import Set
from .content_filter_base import BaseContentFilter

class PolicySOPContentFilter(BaseContentFilter):
    """Content filter optimized for policy and SOP documents."""
    
    def __init__(self):
        super().__init__()
        
        # Patterns for content that should be skipped
        self.skip_patterns = [
            r'^Page \d+ of \d+$',
            r'^Effective Date:?\s*\d{2}\w{3}\d{4}$',
            r'^Version \d+\.?\d*$',
            r'^\s*$',
            r'^Form.*?v\d+$'
        ]
        
        # Boilerplate phrases common in policies/SOPs
        self.boilerplate_phrases = {
            "Page intentionally left blank",
            "Confidential and Proprietary", 
            "All rights reserved",
            "This document is controlled",
            "Printed copies are uncontrolled"
        }
        
        # Patterns for empty form fields and templates
        self.empty_field_patterns = [
            r'_{5,}',  # Long underlines for filling
            r'\.{5,}', # Dotted lines for filling
            r'\[ \]|\□|☐',  # Empty checkboxes
            r'Enter.*?here',
            r'Fill.*?below'
        ]
        
        # Headers/footers specific to policy docs
        self.header_footer_patterns = [
            r'^.*?Diagnostics.*?$',
            r'^.*?Breath.*?$',
            r'^\w+\s+\d{4}\s*$',  # Month Year
            r'^.*?Effective.*?Date.*?$'
        ]
    
    def should_skip_chunk(self, text: str) -> bool:
        """Determine if chunk should be skipped for policy/SOP docs."""
        if not text or len(text.strip()) < 30:
            return True
        
        text_stripped = text.strip()
        
        # Check skip patterns
        for pattern in self.skip_patterns:
            if re.match(pattern, text_stripped, re.IGNORECASE):
                return True
        
        # Skip if mostly empty form fields
        empty_field_matches = sum(1 for pattern in self.empty_field_patterns 
                                if re.search(pattern, text))
        if empty_field_matches > 3:  # Multiple empty fields
            return True
        
        # Skip revision history tables (usually just administrative)
        if re.search(r'Revision History.*?Version.*?Reason.*?Date', text, re.DOTALL | re.IGNORECASE):
            return True
        
        # Skip signature blocks without content
        if re.search(r'Signature.*?Date.*?Signature.*?Date', text, re.DOTALL) and len(text) < 200:
            return True
        
        return False
    
    def clean_text(self, text: str) -> str:
        """Clean policy/SOP specific content."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove form field indicators but keep context
        text = re.sub(r'☐|□|\[ \]', '[checkbox]', text)
        text = re.sub(r'_{5,}', '[field]', text)
        
        # Clean up common artifacts
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'Form.*?v\d+\.?\d*', '', text)
        
        # Normalize quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace('–', '-').replace('—', '-')
        
        return text.strip() 