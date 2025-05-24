"""
Core data models for the RAG Engine.
These models represent the various data structures used throughout the application.
Extracted from APEGA with generalizations for broader use cases.
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
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedContext(BaseModel):
    """A context chunk retrieved from the knowledge base."""
    chunk_id: str
    document_id: str
    text: str
    initial_score: float  # Score from initial hybrid search
    rerank_score: Optional[float] = None  # Score after reranking, if performed
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SynthesizedKnowledge(BaseModel):
    """Knowledge synthesized by deep analysis of retrieved contexts."""
    summary: str
    key_concepts: List[Dict[str, str]] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)  # Generalized from potential_exam_areas
    source_chunk_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict) 