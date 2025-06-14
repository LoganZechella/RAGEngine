"""
PDF Parser for RAG Engine.
Extracts text, tables, and structure from PDF documents using PyMuPDF and PyMuPDF4LLM.
Extracted from APEGA with full functionality preserved.
"""

import fitz  # PyMuPDF
import pymupdf4llm  # New library for robust PDF to Markdown conversion
import re
from typing import List, Dict, Any, Tuple, Optional
import os
from loguru import logger

from backend.src.models.data_models import ParsedDocument, StructuredTable, DocumentType
from backend.src.models.content_types import ContentType
from .content_filter_factory import ContentFilterFactory


class PdfParser:
    """
    Extracts text, tables, and structural metadata from PDF documents using PyMuPDF4LLM.
    """
    
    def __init__(self, 
                 use_ocr: bool = False, 
                 ocr_language: str = 'eng', 
                 enable_content_filtering: bool = True,
                 content_type: Optional[ContentType] = None,
                 enable_auto_detection: bool = True):
        """
        Initialize the PDF parser.
        
        Args:
            use_ocr: Whether to use OCR (Note: PyMuPDF4LLM handles underlying text extraction)
            ocr_language: Language for OCR (Note: PyMuPDF4LLM handles underlying text extraction)
            enable_content_filtering: Whether to enable optimized content filtering
            content_type: Specific content type for filtering, None for auto-detection
            enable_auto_detection: Whether to enable automatic document type detection
        """
        self.use_ocr = use_ocr
        self.ocr_language = ocr_language
        self.enable_content_filtering = enable_content_filtering
        self.content_type = content_type
        
        # Initialize content filter factory
        if self.enable_content_filtering:
            self.filter_factory = ContentFilterFactory(enable_auto_detection)
            logger.info("Content filtering enabled with automatic document type detection")
        else:
            self.filter_factory = None
            
        # Note: With PyMuPDF4LLM, direct OCR control here might be less critical
        # as it handles text extraction comprehensively.

    def parse_pdf(self, file_path: str) -> ParsedDocument:
        """
        Parse a PDF document using PyMuPDF4LLM to extract content as Markdown.

        Args:
            file_path: Path to the PDF file.

        Returns:
            A ParsedDocument object containing the Markdown content and metadata.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        document_id = os.path.basename(file_path).replace('.pdf', '')
        
        logger.info(f"Starting optimized PDF parsing for: {file_path}")
        try:
            # Use page_chunks=True for better processing control and filtering
            if self.enable_content_filtering:
                markdown_pages = pymupdf4llm.to_markdown(
                    file_path, 
                    page_chunks=True,  # Changed from False for page-level processing
                    show_progress=False
                )
                
                # Get appropriate filter
                content_filter = self.filter_factory.get_filter(
                    content_type=self.content_type,
                    document_text=markdown_pages[0].get('text', '')[:5000] if markdown_pages else '',
                    document_title=document_id
                )
                
                # Filter and process pages
                filtered_content = []
                for page_idx, page in enumerate(markdown_pages):
                    page_text = page.get('text', '')
                    
                    # Skip pages with minimal content
                    if len(page_text.strip()) < 100:
                        continue
                    
                    # Filter the page content
                    cleaned_text = content_filter.filter_page_content(page_text)
                    if cleaned_text:
                        filtered_content.append(cleaned_text)
                
                # Combine filtered pages
                markdown_output = "\n\n".join(filtered_content)
                
                logger.info(f"Content filtering: {len(filtered_content)}/{len(markdown_pages)} pages retained")
            else:
                # Fallback to original method if filtering disabled
                markdown_output = pymupdf4llm.to_markdown(file_path, page_chunks=False)
            
            # Open with fitz for metadata and TOC extraction
            doc = fitz.open(file_path)
            metadata = self._extract_metadata(doc, file_path)
            
            # Extract TOC and add it to metadata for text_chunker compatibility
            toc = self._extract_toc(doc)
            metadata['toc'] = toc  # Put TOC in metadata where text_chunker expects it
            
            # Add page count and filtering stats to metadata
            metadata['page_count'] = len(doc)
            
            if self.enable_content_filtering and self.filter_factory:
                filtering_stats = content_filter.get_filtering_stats()
                metadata['filtering_stats'] = filtering_stats
                metadata['content_type_detected'] = content_filter.__class__.__name__
                metadata['filtered_pages'] = len(filtered_content) if 'filtered_content' in locals() else len(doc)
                logger.info(f"Content filtering stats: {filtering_stats}")
                logger.info(f"Detected content type: {content_filter.__class__.__name__}")
            
            doc.close()

            # Tables are embedded in the markdown output
            # For now, we'll return an empty list as tables are in the markdown
            tables: List[StructuredTable] = []
            
            if self.enable_content_filtering:
                logger.success(f"Successfully parsed and filtered PDF: {file_path}")
            else:
                logger.success(f"Successfully parsed PDF: {file_path}")
            
            return ParsedDocument(
                document_id=document_id,
                title=metadata.get('title', document_id),
                source_path=file_path,
                document_type=DocumentType.PDF,
                text_content=markdown_output,  # Main content is the full Markdown
                tables=tables,  # Empty for now, as tables are in Markdown
                metadata=metadata  # Contains all metadata including TOC
            )

        except Exception as e:
            logger.error(f"PyMuPDF4LLM failed to parse PDF {file_path}: {e}")
            # Return a minimal ParsedDocument on failure to prevent downstream crashes
            return ParsedDocument(
                document_id=document_id,
                title=os.path.basename(file_path),
                source_path=file_path,
                document_type=DocumentType.PDF,
                text_content="",  # Default to empty string
                tables=[],
                metadata={"error": str(e)}
            )

    def _extract_metadata(self, pdf_document: fitz.Document, pdf_path: str) -> Dict[str, Any]:
        """
        Extract metadata from the PDF document.
        
        Args:
            pdf_document: PyMuPDF document object
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary of metadata
        """
        metadata = pdf_document.metadata
        # Ensure basic metadata is present, provide defaults if not
        return {
            "title": metadata.get("title") or os.path.basename(pdf_path),
            "author": metadata.get("author") or "Unknown",
            "subject": metadata.get("subject") or "Unknown",
            "producer": metadata.get("producer") or "Unknown",
            "creationDate": metadata.get("creationDate") or "Unknown",
            "modDate": metadata.get("modDate") or "Unknown",
            "encryption": metadata.get("encryption") or "None"
        }

    def _extract_toc(self, pdf_document: fitz.Document) -> List[Dict[str, Any]]:
        """
        Extract the table of contents (TOC) from the PDF document.
        
        Args:
            pdf_document: PyMuPDF document object
            
        Returns:
            List of TOC entries with title, page number, and level
        """
        toc = []
        raw_toc = []
        
        try:
            # get_toc() returns a list of lists: [lvl, title, page, pos]
            raw_toc = pdf_document.get_toc(simple=True)
        except Exception as e:
            logger.warning(f"Could not extract TOC: {e}")
            return toc  # Return empty toc if extraction fails

        if not raw_toc:
            logger.debug("No TOC found in PDF")
            return toc

        for item in raw_toc:
            try:
                # Safely access item elements with bounds checking
                if not isinstance(item, (list, tuple)) or len(item) < 3:
                    logger.warning(f"Skipping malformed TOC item: {item}")
                    continue
                
                level = int(item[0]) if item[0] is not None else 0
                title = str(item[1]).strip() if item[1] is not None else "Untitled Section"
                page_num = int(item[2]) if item[2] is not None else 0

                toc.append({
                    "level": level,
                    "title": title,
                    "page": page_num
                })
                
            except (IndexError, TypeError, ValueError) as e:
                logger.warning(f"Skipping malformed TOC item: {item}, error: {e}")
                continue
            except Exception as e:
                logger.warning(f"Unexpected error processing TOC item: {item}, error: {e}")
                continue
        
        return toc 