from typing import Optional
from .content_filter_base import BaseContentFilter
from .content_filter import ScientificContentFilter
from .policy_sop_content_filter import PolicySOPContentFilter
from .form_content_filter import FormContentFilter
from .document_type_detector import DocumentTypeDetector
from backend.src.models.content_types import ContentType

class ContentFilterFactory:
    """Factory for creating appropriate content filters based on document type."""
    
    def __init__(self, enable_auto_detection: bool = True):
        self.enable_auto_detection = enable_auto_detection
        self.detector = DocumentTypeDetector() if enable_auto_detection else None
    
    def get_filter(
        self, 
        content_type: Optional[ContentType] = None,
        document_text: str = "",
        document_title: str = ""
    ) -> BaseContentFilter:
        """Get appropriate content filter for document."""
        
        # Auto-detect if no type specified
        if content_type is None and self.enable_auto_detection:
            detected_type, confidence = self.detector.detect_content_type(
                document_text, document_title
            )
            content_type = detected_type
        
        # Default to scientific if no detection
        if content_type is None:
            content_type = ContentType.SCIENTIFIC
        
        # Return appropriate filter
        if content_type == ContentType.POLICY_SOP:
            return PolicySOPContentFilter()
        elif content_type == ContentType.FORM:
            return FormContentFilter()
        elif content_type == ContentType.SCIENTIFIC:
            return ScientificContentFilter()
        else:
            # Default to policy/SOP for general content
            return PolicySOPContentFilter() 