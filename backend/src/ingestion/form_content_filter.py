import re
from .content_filter_base import BaseContentFilter

class FormContentFilter(BaseContentFilter):
    """Content filter optimized for forms and structured documents."""
    
    def __init__(self):
        super().__init__()
        
        # Form-specific patterns to preserve structure while removing noise
        self.preserve_structure_patterns = [
            r'Section \d+:',
            r'Part [A-Z]:',
            r'\d+\.',  # Numbered items
            r'[A-Z]\.',  # Lettered items
        ]
        
        # Empty table patterns
        self.empty_table_patterns = [
            r'^\s*\|\s*\|\s*\|\s*\|?\s*$',  # Empty table rows
            r'^\s*Name\s*Location\s*Supply.*?Status\s*$'  # Empty header only
        ]
    
    def should_skip_chunk(self, text: str) -> bool:
        """Skip chunks that are mostly empty form structure."""
        if not text or len(text.strip()) < 20:
            return True
        
        # Skip if mostly empty table structure
        lines = text.split('\n')
        empty_table_lines = sum(1 for line in lines 
                              if any(re.match(pattern, line) for pattern in self.empty_table_patterns))
        
        if len(lines) > 0 and empty_table_lines / len(lines) > 0.7:
            return True
        
        # Skip if mostly checkboxes and fields without content
        checkbox_count = len(re.findall(r'☐|□|\[ \]', text))
        field_count = len(re.findall(r'_{3,}', text))
        
        if checkbox_count + field_count > 5 and len(text.split()) < 50:
            return True
        
        return False
    
    def clean_text(self, text: str) -> str:
        """Clean form content while preserving structure."""
        if not text:
            return ""
        
        # Replace checkboxes with readable text
        text = re.sub(r'☐', '[unchecked]', text)
        text = re.sub(r'☑|☒', '[checked]', text)
        
        # Replace long underlines with field markers
        text = re.sub(r'_{3,}', '[input field]', text)
        
        # Clean excessive whitespace while preserving structure
        text = re.sub(r' {3,}', '   ', text)  # Max 3 spaces
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
        
        return text.strip() 