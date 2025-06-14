"""
Enhanced core data models for the RAG Engine.
These models represent the various data structures used throughout the application.
Optimized for readability and sophisticated knowledge synthesis output.
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    """Types of documents that can be processed."""
    PDF = "pdf"
    HTML = "html"
    TEXT = "text"


class ChunkType(str, Enum):
    """Types of document chunks."""
    TEXT = "text"
    TABLE = "table"
    HEADING = "heading"
    LIST = "list"


class DocumentCollection(str, Enum):
    """Available document collections for organization."""
    CURRENT = "current_documents"
    LEGACY = "legacy_documents"
    SOP_POLICY = "sop_policy"
    RESEARCH = "research_data"
    CLINICAL = "clinical_studies"
    REGULATORY = "regulatory_documents"
    TRAINING = "training_materials"


class AnalysisDepth(str, Enum):
    """Depth levels of analysis performed."""
    BASIC = "basic"
    STANDARD = "standard"
    PHD_LEVEL = "phd_level"
    EXPERT = "expert"
    FALLBACK_TEXT = "fallback_text"
    ERROR = "error"


class EvidenceQuality(str, Enum):
    """Quality assessment of evidence supporting concepts."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    INSUFFICIENT = "insufficient"
    CONFLICTING = "conflicting"


class CollectionMetadata(BaseModel):
    """Metadata for a document collection."""
    collection_name: DocumentCollection
    display_name: str
    description: str
    document_count: int = 0
    last_updated: Optional[datetime] = None
    auto_assignment_rules: List[str] = Field(default_factory=list)


class StructuredTable(BaseModel):
    """Representation of a table extracted from a document."""
    table_id: str
    data: List[List[str]]
    caption: Optional[str] = None
    page_number: Optional[int] = None


class ParsedDocument(BaseModel):
    """A document after parsing, containing text and metadata."""
    document_id: str
    title: Optional[str] = None
    source_path: str
    document_type: DocumentType
    text_content: str
    tables: List[StructuredTable] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parsed_at: datetime = Field(default_factory=datetime.now)


class TextChunk(BaseModel):
    """A chunk of text from a document with metadata."""
    chunk_id: str
    document_id: str
    text: str
    chunk_type: ChunkType = ChunkType.TEXT
    page_number: Optional[int] = None
    section_path: Optional[List[str]] = None  # Hierarchical path of headings
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmbeddedChunk(BaseModel):
    """A text chunk with its vector embedding."""
    chunk_id: str
    document_id: str
    text: str
    embedding_vector: Optional[List[float]] = None  # Can be None if embedding failed
    chunk_type: ChunkType = ChunkType.TEXT
    collection: DocumentCollection = DocumentCollection.CURRENT  # NEW FIELD
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedContext(BaseModel):
    """A context chunk retrieved from the knowledge base."""
    chunk_id: str
    document_id: str
    text: str
    initial_score: float  # Score from initial hybrid search
    rerank_score: Optional[float] = None  # Score after reranking, if performed
    collection: Optional[DocumentCollection] = None  # NEW FIELD
    metadata: Dict[str, Any] = Field(default_factory=dict)
    synthesis_quality_indicators: Optional[Dict[str, Any]] = Field(None, description="Indicators for the quality of the synthesis process.")
    analysis_thoughts: Optional[List[str]] = Field(None, description="The chain of thought and reasoning process from the analysis model.")


class KeyConcept(BaseModel):
    """A key concept identified in the synthesized knowledge."""
    concept: str
    explanation: str
    importance: str
    evidence_quality: Optional[EvidenceQuality] = None
    controversies: Optional[str] = None
    related_concepts: List[str] = Field(default_factory=list)
    confidence_score: Optional[float] = None


class SynthesisInsight(BaseModel):
    """A novel insight generated from cross-referencing information."""
    insight: str
    supporting_evidence: List[str] = Field(default_factory=list)
    confidence_level: Optional[str] = None
    implications: Optional[str] = None


class ResearchGap(BaseModel):
    """An identified gap or limitation in the current knowledge."""
    gap_description: str
    severity: Optional[str] = None  # "critical", "moderate", "minor"
    suggested_investigation: Optional[str] = None


class SynthesizedKnowledge(BaseModel):
    """Enhanced knowledge synthesized by deep analysis of retrieved contexts."""
    
    # Core synthesis results
    summary: str
    key_concepts: List[KeyConcept] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    
    # Advanced analysis fields
    synthesis_insights: List[SynthesisInsight] = Field(default_factory=list)
    research_gaps: List[ResearchGap] = Field(default_factory=list)
    
    # Methodological and theoretical analysis
    methodological_observations: Optional[str] = None
    theoretical_implications: Optional[str] = None
    
    # Quality and confidence metrics
    analysis_depth: AnalysisDepth = AnalysisDepth.STANDARD
    overall_confidence: Optional[float] = None
    completeness_score: Optional[float] = None
    
    # Source tracking
    source_chunk_ids: List[str] = Field(default_factory=list)
    num_source_contexts: int = 0
    
    # Enhanced metadata
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    analysis_model: Optional[str] = None
    query_complexity: Optional[str] = None
    synthesis_quality_indicators: Dict[str, Any] = Field(default_factory=dict)
    
    def get_display_summary(self, max_length: int = 300) -> str:
        """Get a truncated summary for display purposes."""
        if len(self.summary) <= max_length:
            return self.summary
        return self.summary[:max_length-3] + "..."
    
    def get_confidence_indicator(self) -> str:
        """Get a human-readable confidence indicator."""
        if self.overall_confidence is None:
            return "Unknown"
        elif self.overall_confidence >= 0.8:
            return "High"
        elif self.overall_confidence >= 0.6:
            return "Moderate"
        elif self.overall_confidence >= 0.4:
            return "Low"
        else:
            return "Very Low"
    
    def get_analysis_quality_summary(self) -> Dict[str, str]:
        """Get a summary of analysis quality indicators."""
        return {
            "depth": self.analysis_depth.value.replace("_", " ").title(),
            "confidence": self.get_confidence_indicator(),
            "completeness": f"{self.completeness_score:.1%}" if self.completeness_score else "Unknown",
            "num_concepts": str(len(self.key_concepts)),
            "num_insights": str(len(self.synthesis_insights)),
            "num_sources": str(self.num_source_contexts)
        }
    
    def has_advanced_analysis(self) -> bool:
        """Check if advanced analysis fields are populated."""
        return bool(
            self.synthesis_insights or 
            self.research_gaps or 
            self.methodological_observations or 
            self.theoretical_implications
        )

class SearchResult(BaseModel):
    query: str
