import re
from typing import Dict, List, Tuple
from backend.src.models.content_types import ContentType, DocumentCategory

class DocumentTypeDetector:
    """Analyzes document content to determine appropriate filtering strategy."""
    
    def __init__(self):
        # Policy/SOP indicators
        self.policy_sop_patterns = [
            r'\bPURPOSE\b',
            r'\bSCOPE\b', 
            r'\bDEFINITIONS?\b',
            r'\bPROCEDURE\b',
            r'\bRESPONSIBILITIES\b',
            r'\bREFERENCES\b',
            r'SOP-\d+',
            r'POL-\d+',
            r'Quality Management System',
            r'Effective Date'
        ]
        
        # Form indicators  
        self.form_patterns = [
            r'☐|□|\[ \]',  # Checkboxes
            r'Signature.*?Date',
            r'Form.*?v\d+',
            r'Fill.*?below',
            r'Check.*?applicable',
            r'____+',  # Signature lines
            r'Enter.*?here'
        ]
        
        # Template indicators
        self.template_patterns = [
            r'\[.*?\]',  # Placeholder brackets
            r'<.*?>',    # Placeholder tags
            r'instructions.*?italics',
            r'template.*?adjusted',
            r'Contact.*?obtain.*?number'
        ]
        
        # Scientific indicators
        self.scientific_patterns = [
            r'Experiment \d+',
            r'Table R\d+-',
            r'Figure R\d+-', 
            r'Results as reported',
            r'Appendix [A-Z]\d*',
            r'Statistical analysis',
            r'Methodology'
        ]

    def detect_content_type(self, text: str, title: str = "") -> Tuple[ContentType, float]:
        """Detect the primary content type and confidence score."""
        
        text_sample = text[:5000]  # First 5k chars for analysis
        combined_text = f"{title} {text_sample}".lower()
        
        scores = {
            ContentType.POLICY_SOP: self._score_patterns(combined_text, self.policy_sop_patterns),
            ContentType.FORM: self._score_patterns(combined_text, self.form_patterns),
            ContentType.TEMPLATE: self._score_patterns(combined_text, self.template_patterns),
            ContentType.SCIENTIFIC: self._score_patterns(combined_text, self.scientific_patterns)
        }
        
        # Determine best match
        best_type = max(scores.items(), key=lambda x: x[1])
        
        if best_type[1] < 0.3:  # Low confidence threshold
            return ContentType.GENERAL, best_type[1]
        
        return best_type[0], best_type[1]
    
    def _score_patterns(self, text: str, patterns: List[str]) -> float:
        """Score text against pattern list."""
        matches = 0
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches += 1
        return matches / len(patterns) if patterns else 0.0 