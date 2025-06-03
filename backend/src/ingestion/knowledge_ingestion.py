from typing import List, Dict, Any, Optional
from loguru import logger
import os

from backend.src.ingestion.document_source_manager import DocumentSourceManager
from backend.src.ingestion.pdf_parser import PdfParser
from backend.src.ingestion.text_chunker import TextChunker
from backend.src.ingestion.embedding_generator import EmbeddingGenerator
from backend.src.ingestion.vector_db_manager import VectorDBManager
from backend.src.models.data_models import DocumentType
from backend.src.models.content_types import ContentType

class KnowledgeIngestion:
    """Orchestrates document ingestion pipeline with multi-collection support."""
    
    def __init__(
        self,
        source_paths: List[str],
        openai_api_key: str,
        qdrant_url: str,
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "knowledge_base",
        chunking_strategy: str = "paragraph",
        max_chunk_size_tokens: int = 512,
        chunk_overlap_tokens: int = 100,
        vector_dimensions: int = 1536,
        enable_content_filtering: bool = True,
        enable_deduplication: bool = True,
        content_type: Optional[str] = None,
        enable_auto_detection: bool = True,
        collection_manager: Optional[Any] = None,  # NEW: Collection manager
        vector_db_manager: Optional[Any] = None    # NEW: Vector DB manager
    ):
        self.source_paths = source_paths
        
        # Store collection manager for multi-collection support
        self.collection_manager = collection_manager
        self.vector_db_manager = vector_db_manager
        
        # Convert string to enum if provided
        content_type_enum = None
        if content_type and content_type != "auto":
            content_type_enum = ContentType(content_type)
        
        # Initialize components with optimization settings
        self.document_manager = DocumentSourceManager(source_paths)
        self.pdf_parser = PdfParser(
            enable_content_filtering=enable_content_filtering,
            content_type=content_type_enum,
            enable_auto_detection=enable_auto_detection
        )
        self.text_chunker = TextChunker(
            strategy=chunking_strategy,
            max_chunk_size_tokens=max_chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            enable_deduplication=enable_deduplication,
            enable_content_filtering=enable_content_filtering,
            content_type=content_type_enum,
            enable_auto_detection=enable_auto_detection
        )
        self.embedding_generator = EmbeddingGenerator(
            api_key=openai_api_key,
            dimensions=vector_dimensions
        )
        
        # Use provided vector DB manager or create default one
        if vector_db_manager:
            self.vector_db = vector_db_manager
        else:
            self.vector_db = VectorDBManager(
                url=qdrant_url,
                api_key=qdrant_api_key,
                collection_name=collection_name,
                vector_dimensions=vector_dimensions
            )
    
    def process_documents(self) -> Dict[str, Any]:
        """Process all documents in source paths."""
        stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "errors": []
        }
        
        # Get documents to process
        documents = self.document_manager.get_documents_to_process()
        
        for doc_info in documents:
            try:
                # Parse document
                if doc_info["document_type"] == DocumentType.PDF:
                    parsed = self.pdf_parser.parse_pdf(doc_info["source_path"])
                else:
                    logger.warning(f"Unsupported type: {doc_info['document_type']}")
                    continue
                
                # Chunk document
                chunks = self.text_chunker.chunk_document(parsed)
                stats["chunks_created"] += len(chunks)
                
                # Generate embeddings
                embedded = self.embedding_generator.generate_embeddings(chunks)
                stats["embeddings_generated"] += len(embedded)
                
                # Store in vector database
                self.vector_db.upsert_embeddings(embedded)
                
                # Mark as processed
                self.document_manager.mark_document_processed(
                    doc_info["source_path"],
                    success=True
                )
                stats["documents_processed"] += 1
                
                logger.info(f"Successfully processed: {doc_info['source_path']}")
                
            except Exception as e:
                error_msg = f"Error processing {doc_info['source_path']}: {e}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)
        
        logger.info(f"Ingestion complete: {stats}")
        return stats
    
    def process_single_document(self, file_path: str) -> Dict[str, Any]:
        """Process a single document."""
        stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "errors": []
        }
        
        try:
            # Create document info
            doc_info = {
                "document_id": os.path.basename(file_path),
                "source_path": file_path,
                "document_type": self._infer_document_type(file_path),
                "metadata": {}
            }
            
            # Parse document
            if doc_info["document_type"] == DocumentType.PDF:
                parsed = self.pdf_parser.parse_pdf(doc_info["source_path"])
            else:
                logger.warning(f"Unsupported type: {doc_info['document_type']}")
                return stats
            
            # Chunk document
            chunks = self.text_chunker.chunk_document(parsed)
            stats["chunks_created"] += len(chunks)
            
            # Generate embeddings
            embedded = self.embedding_generator.generate_embeddings(chunks)
            stats["embeddings_generated"] += len(embedded)
            
            # Store in vector database
            self.vector_db.upsert_embeddings(embedded)
            
            # Mark as processed
            self.document_manager.mark_document_processed(doc_info["source_path"])
            stats["documents_processed"] += 1
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            stats["errors"].append(str(e))
        
        return stats
    
    def process_single_document_to_collection(
        self, 
        file_path: str, 
        target_collection: Any
    ) -> Dict[str, Any]:
        """Process a single document and assign to specific collection."""
        stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "errors": []
        }
        
        try:
            # Create document info
            doc_info = {
                "document_id": os.path.basename(file_path),
                "source_path": file_path,
                "document_type": self._infer_document_type(file_path),
                "metadata": {}
            }
            
            # Parse document
            if doc_info["document_type"] == DocumentType.PDF:
                parsed = self.pdf_parser.parse_pdf(doc_info["source_path"])
            else:
                logger.warning(f"Unsupported type: {doc_info['document_type']}")
                return stats
            
            # Chunk document (TextChunk objects - no collection field)
            chunks = self.text_chunker.chunk_document(parsed)
            stats["chunks_created"] += len(chunks)
            
            # Generate embeddings (creates EmbeddedChunk objects with collection field)
            embedded = self.embedding_generator.generate_embeddings(chunks)
            stats["embeddings_generated"] += len(embedded)
            
            # Assign collection to embedded chunks (EmbeddedChunk has collection field)
            for embedded_chunk in embedded:
                embedded_chunk.collection = target_collection
            
            # Store in vector database with explicit collection assignment
            if hasattr(self.vector_db, 'upsert_embeddings'):
                # Use multi-collection manager
                upsert_results = self.vector_db.upsert_embeddings(
                    embedded, 
                    target_collection=target_collection
                )
                logger.info(f"Upserted to collections: {upsert_results}")
            else:
                # Fallback for single collection manager
                self.vector_db.upsert_embeddings(embedded)
            
            # Mark as processed
            self.document_manager.mark_document_processed(doc_info["source_path"])
            stats["documents_processed"] += 1
            
            logger.info(f"Successfully processed {file_path} to collection {target_collection.value}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path} to collection {target_collection}: {e}")
            stats["errors"].append(str(e))
        
        return stats
    
    def _infer_document_type(self, file_path: str) -> DocumentType:
        """Infer document type from file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            return DocumentType.PDF
        elif ext in ['.html', '.htm']:
            return DocumentType.HTML
        else:
            return DocumentType.TEXT
    
    def get_ingestion_info(self) -> Dict[str, Any]:
        """Get information about the ingestion configuration."""
        return {
            "source_paths": self.source_paths,
            "chunking_strategy": self.text_chunker.strategy,
            "max_chunk_size_tokens": self.text_chunker.max_chunk_size_tokens,
            "chunk_overlap_tokens": self.text_chunker.chunk_overlap_tokens,
            "vector_dimensions": self.embedding_generator.dimensions
        } 