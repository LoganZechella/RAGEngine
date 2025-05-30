from enum import Enum

class ContentType(str, Enum):
    """Types of document content for filtering purposes."""
    SCIENTIFIC = "scientific"
    POLICY_SOP = "policy_sop" 
    FORM = "form"
    TEMPLATE = "template"
    GENERAL = "general"

class DocumentCategory(str, Enum):
    """Document categories for processing optimization."""
    TECHNICAL_REPORT = "technical_report"
    POLICY_DOCUMENT = "policy_document"
    STANDARD_FORM = "standard_form"
    DOCUMENT_TEMPLATE = "document_template"
    MIXED_CONTENT = "mixed_content" 