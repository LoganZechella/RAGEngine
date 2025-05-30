#!/usr/bin/env python
"""
Interactive RAG Testing Tool for RAGEngine.
Command-line interface for manual RAG testing and debugging with enhanced content filtering.
"""

import os
import sys
import cmd
import json
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from loguru import logger

# Load environment
load_dotenv()

from backend.src.api.knowledge_base_api import KnowledgeBaseAPI
from backend.src.models.content_types import ContentType
from backend.src.ingestion.document_type_detector import DocumentTypeDetector
from backend.src.ingestion.content_filter_factory import ContentFilterFactory

def display_enhanced_synthesis(synthesis):
    """Display synthesized knowledge with enhanced formatting."""
    if not synthesis:
        print("⚠️  Knowledge synthesis not available")
        return
    
    print(f"\n🧠 Enhanced Knowledge Synthesis:")
    print("=" * 80)
    
    # Analysis quality overview
    quality = synthesis.get_analysis_quality_summary()
    print(f"📊 Analysis Quality: {quality['depth']} | Confidence: {quality['confidence']} | "
          f"Sources: {quality['num_sources']} | Concepts: {quality['num_concepts']}")
    
    if synthesis.overall_confidence:
        confidence_bar = "█" * int(synthesis.overall_confidence * 10) + "░" * (10 - int(synthesis.overall_confidence * 10))
        print(f"🎯 Confidence: [{confidence_bar}] {synthesis.overall_confidence:.1%}")
    
    print("-" * 80)
    
    # Summary
    print(f"📋 Executive Summary:")
    print(f"{synthesis.summary}")
    
    if synthesis.key_concepts:
        print(f"\n🔑 Key Concepts ({len(synthesis.key_concepts)}):")
        for i, concept in enumerate(synthesis.key_concepts[:5], 1):  # Show top 5
            evidence_icon = {
                "strong": "🟢",
                "moderate": "🟡", 
                "weak": "🟠",
                "insufficient": "🔴",
                "conflicting": "🟣"
            }.get(concept.evidence_quality.value if concept.evidence_quality else "insufficient", "⚪")
            
            confidence_text = f" ({concept.confidence_score:.1%})" if concept.confidence_score else ""
            
            print(f"\n  {i}. {evidence_icon} {concept.concept}{confidence_text}")
            print(f"     💡 {concept.explanation}")
            print(f"     🎯 {concept.importance}")
            
            if concept.controversies:
                print(f"     ⚠️  Controversies: {concept.controversies}")
            
            if concept.related_concepts:
                print(f"     🔗 Related: {', '.join(concept.related_concepts[:3])}")
    
    if synthesis.synthesis_insights:
        print(f"\n💡 Novel Synthesis Insights ({len(synthesis.synthesis_insights)}):")
        for i, insight in enumerate(synthesis.synthesis_insights[:3], 1):  # Show top 3
            confidence_icon = {"high": "🟢", "moderate": "🟡", "low": "🟠"}.get(
                insight.confidence_level, "⚪"
            )
            
            print(f"\n  {i}. {confidence_icon} {insight.insight}")
            if insight.implications:
                print(f"     📈 Implications: {insight.implications}")
            if insight.supporting_evidence:
                print(f"     📚 Evidence: {len(insight.supporting_evidence)} supporting points")
    
    if synthesis.research_gaps:
        print(f"\n🔍 Research Gaps Identified ({len(synthesis.research_gaps)}):")
        for i, gap in enumerate(synthesis.research_gaps[:3], 1):
            severity_icon = {"critical": "🔴", "moderate": "🟡", "minor": "🟢"}.get(
                gap.severity, "⚪"
            )
            
            print(f"\n  {i}. {severity_icon} {gap.gap_description}")
            if gap.suggested_investigation:
                print(f"     🎯 Suggested: {gap.suggested_investigation}")
    
    if synthesis.topics:
        print(f"\n📚 Key Topics ({len(synthesis.topics)}):")
        for topic in synthesis.topics[:6]:  # Show up to 6 topics
            print(f"  • {topic}")
    
    if synthesis.methodological_observations:
        print(f"\n🔬 Methodological Observations:")
        print(f"{synthesis.methodological_observations}")
    
    if synthesis.theoretical_implications:
        print(f"\n🏛️  Theoretical Implications:")
        print(f"{synthesis.theoretical_implications}")
    
    # Technical details (can be toggled)
    if hasattr(synthesis, 'synthesis_quality_indicators') and synthesis.synthesis_quality_indicators:
        print(f"\n📈 Technical Analysis Metrics:")
        indicators = synthesis.synthesis_quality_indicators
        for key, value in indicators.items():
            print(f"  • {key.replace('_', ' ').title()}: {value}")


def display_contexts_enhanced(contexts):
    """Display retrieved contexts with enhanced formatting."""
    if not contexts:
        print("📊 No contexts retrieved")
        return
    
    print(f"📊 Retrieved {len(contexts)} contexts")
    print("-" * 60)
    
    for i, ctx in enumerate(contexts[:5], 1):  # Show top 5
        initial_score = ctx.get('initial_score', 0)
        rerank_score = ctx.get('rerank_score')
        
        # Score display
        if rerank_score is not None:
            score_bar = "█" * int(rerank_score * 10) + "░" * (10 - int(rerank_score * 10))
            print(f"\n{i}. [{score_bar}] Rerank: {rerank_score:.3f} (Initial: {initial_score:.3f})")
        else:
            score_bar = "█" * int(initial_score * 10) + "░" * (10 - int(initial_score * 10))
            print(f"\n{i}. [{score_bar}] Score: {initial_score:.3f}")
        
        # Document info with content type if available
        metadata = ctx.get('metadata', {})
        doc_id = metadata.get('document_id', 'Unknown')
        source = metadata.get('source', 'Unknown')
        content_type = metadata.get('content_type_detected', 'Unknown')
        
        print(f"   📄 Document: {doc_id}")
        if source != 'Unknown':
            print(f"   📁 Source: {source}")
        if content_type != 'Unknown':
            print(f"   🏷️  Content Type: {content_type}")
        
        # Text preview
        text = ctx['text']
        if len(text) > 200:
            print(f"   📝 {text[:200]}...")
        else:
            print(f"   📝 {text}")

class InteractiveRAGShell(cmd.Cmd):
    """Interactive shell for RAG testing and debugging with enhanced content filtering."""
    
    intro = """
    🔍 RAGEngine Interactive Testing Shell with Content Filtering
    ============================================================
    
    Basic Commands:
    - query <text>        : Test full RAG pipeline with a query
    - search <text>       : Test hybrid search only
    - dense <text>        : Test dense search only
    - sparse <text>       : Test sparse search only
    - rerank <text>       : Test search with reranking
    - analyze <text>      : Test with knowledge synthesis
    
    Content Filtering Commands:
    - detect <text>       : Test document type detection
    - filter_test <type>  : Test specific content filter
    - filter_stats        : Show content filtering statistics
    - content_types       : List available content types
    
    Configuration Commands:
    - filters <filters>   : Set search filters (JSON format)
    - config             : Show current configuration
    - filter_config      : Show content filtering configuration
    - set_content_type <type> : Set default content type for processing
    
    System Commands:
    - stats              : Show database statistics
    - ingest             : Run document ingestion
    - system             : Show system information
    - last               : Show details from last query
    - examples           : Show example queries
    - filter_examples    : Show content filtering examples
    
    Collection Management:
    - clear              : Clear all data from collection
    - delete_collection  : Delete entire collection
    - recreate_collection: Recreate collection with fresh config
    - processed          : Show processed documents
    
    - help               : Show this help
    - quit               : Exit the shell
    
    Type 'help <command>' for detailed help on a command.
    """
    
    prompt = 'RAG> '
    
    def __init__(self):
        super().__init__()
        self.setup_rag_engine()
        self.current_filters = None
        self.last_results = None
        self.document_type_detector = DocumentTypeDetector()
        self.content_filter_factory = ContentFilterFactory()
    
    def setup_rag_engine(self):
        """Initialize RAG engine components with enhanced content filtering."""
        try:
            print("Initializing RAGEngine with Enhanced Content Filtering...")
            
            # Enhanced configuration with content filtering options
            config = {
                "openai_api_key": os.getenv("OPENAI_API_KEY"),
                "google_api_key": os.getenv("GOOGLE_API_KEY"),
                "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
                "qdrant_api_key": os.getenv("QDRANT_API_KEY"),
                "collection_name": os.getenv("QDRANT_COLLECTION_NAME", "knowledge_base"),
                "source_paths": [os.getenv("SOURCE_DOCUMENTS_DIR", "./documents")],
                "chunking_strategy": os.getenv("CHUNKING_STRATEGY", "paragraph"),
                "chunk_size_tokens": int(os.getenv("CHUNK_SIZE_TOKENS", "512")),
                "chunk_overlap_tokens": int(os.getenv("CHUNK_OVERLAP_TOKENS", "100")),
                "vector_dimensions": int(os.getenv("VECTOR_DIMENSIONS", "1536")),
                "top_k_dense": int(os.getenv("TOP_K_DENSE", "10")),
                "top_k_sparse": int(os.getenv("TOP_K_SPARSE", "10")),
                "top_k_rerank": int(os.getenv("TOP_K_RERANK", "5")),
                
                # Content filtering configuration
                "enable_content_filtering": os.getenv("ENABLE_CONTENT_FILTERING", "true").lower() == "true",
                "enable_deduplication": os.getenv("ENABLE_DEDUPLICATION", "true").lower() == "true",
                "enable_document_type_detection": os.getenv("ENABLE_DOCUMENT_TYPE_DETECTION", "true").lower() == "true",
                "default_content_type": os.getenv("DEFAULT_CONTENT_TYPE", "auto"),
                "policy_filter_aggressive": os.getenv("POLICY_FILTER_AGGRESSIVE", "false").lower() == "true",
                "form_filter_preserve_structure": os.getenv("FORM_FILTER_PRESERVE_STRUCTURE", "true").lower() == "true",
                "scientific_filter_legacy_mode": os.getenv("SCIENTIFIC_FILTER_LEGACY_MODE", "false").lower() == "true",
                "min_policy_chunk_length": int(os.getenv("MIN_POLICY_CHUNK_LENGTH", "30")),
                "min_form_chunk_length": int(os.getenv("MIN_FORM_CHUNK_LENGTH", "20")),
                "skip_empty_form_sections": os.getenv("SKIP_EMPTY_FORM_SECTIONS", "true").lower() == "true",
                "preserve_signature_blocks": os.getenv("PRESERVE_SIGNATURE_BLOCKS", "false").lower() == "true"
            }
            
            # Validate required configuration
            if not config["openai_api_key"]:
                print("⚠️  Warning: OPENAI_API_KEY not set. Some features may not work.")
            
            if not config["google_api_key"]:
                print("⚠️  Warning: GOOGLE_API_KEY not set. Knowledge synthesis will not be available.")
            
            self.kb_api = KnowledgeBaseAPI(config)
            self.config = config
            
            print("✅ RAGEngine initialized successfully with enhanced content filtering!")
            
            # Display content filtering status
            if config["enable_content_filtering"]:
                print(f"🎯 Content Filtering: {'✅ Enabled' if config['enable_content_filtering'] else '❌ Disabled'}")
                print(f"🔍 Document Type Detection: {'✅ Enabled' if config['enable_document_type_detection'] else '❌ Disabled'}")
                print(f"🏷️  Default Content Type: {config['default_content_type']}")
            
        except Exception as e:
            print(f"❌ Failed to initialize RAGEngine: {str(e)}")
            print("Make sure Qdrant is running and API keys are set.")
            sys.exit(1)
    
    def do_detect(self, document_text: str):
        """Test document type detection on provided text."""
        if not document_text.strip():
            print("Usage: detect <text>")
            print("Example: detect 'PURPOSE This document establishes procedures...'")
            return
        
        try:
            print(f"\n🔍 Document Type Detection Analysis:")
            print("=" * 60)
            
            detected_type, confidence = self.document_type_detector.detect_content_type(document_text)
            
            # Display results
            confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
            print(f"🏷️  Detected Type: {detected_type.value}")
            print(f"🎯 Confidence: [{confidence_bar}] {confidence:.1%}")
            
            # Show what filter would be used
            content_filter = self.content_filter_factory.get_filter(
                content_type=detected_type,
                document_text=document_text
            )
            
            print(f"🔧 Filter Selected: {content_filter.__class__.__name__}")
            
            # Show pattern analysis
            print(f"\n📊 Pattern Analysis:")
            print(f"Text length: {len(document_text)} characters")
            print(f"First 200 chars: {document_text[:200]}...")
            
        except Exception as e:
            print(f"❌ Document type detection failed: {str(e)}")
    
    def do_filter_test(self, content_type: str):
        """Test a specific content filter type."""
        if not content_type.strip():
            print("Usage: filter_test <content_type>")
            print("Available types: scientific, policy_sop, form, template, general")
            return
        
        try:
            # Convert string to enum
            if content_type.lower() not in [ct.value for ct in ContentType]:
                print(f"❌ Invalid content type: {content_type}")
                print(f"Available types: {', '.join([ct.value for ct in ContentType])}")
                return
            
            content_type_enum = ContentType(content_type.lower())
            content_filter = self.content_filter_factory.get_filter(content_type=content_type_enum)
            
            print(f"\n🔧 Testing {content_filter.__class__.__name__}:")
            print("=" * 60)
            
            # Test with sample text
            test_texts = {
                ContentType.SCIENTIFIC: "Experiment 1 Report - Page 1 of 5\nResults as reported by Mayo Clinic show...",
                ContentType.POLICY_SOP: "PURPOSE\nThis SOP-001 establishes procedures for...\nSCOPE\nThis applies to all personnel...",
                ContentType.FORM: "☐ Check applicable boxes\nSignature: _________________ Date: _______\nEnter information below:",
                ContentType.TEMPLATE: "[Enter type of device here]\nContact QA to obtain the next available document number...",
                ContentType.GENERAL: "This is a general document with standard content and normal text."
            }
            
            sample_text = test_texts.get(content_type_enum, "Sample text for testing.")
            
            print(f"📝 Sample Text:")
            print(f"{sample_text}")
            
            print(f"\n🔍 Filter Analysis:")
            should_skip = content_filter.should_skip_chunk(sample_text)
            cleaned_text = content_filter.clean_text(sample_text)
            
            print(f"Should Skip: {'Yes' if should_skip else 'No'}")
            print(f"Original Length: {len(sample_text)} chars")
            print(f"Cleaned Length: {len(cleaned_text)} chars")
            print(f"Cleaned Text: {cleaned_text[:200]}...")
            
        except Exception as e:
            print(f"❌ Filter test failed: {str(e)}")
    
    def do_filter_stats(self, line):
        """Show content filtering statistics from last ingestion."""
        try:
            processed_docs = self.kb_api.get_processed_documents()
            
            print(f"\n📊 Content Filtering Statistics:")
            print("=" * 60)
            
            if not processed_docs:
                print("No documents processed yet. Run 'ingest' to see filtering stats.")
                return
            
            total_filtering_stats = {
                'total_chunks_processed': 0,
                'chunks_filtered_out': 0,
                'chunks_cleaned': 0,
                'boilerplate_removed': 0,
                'duplicates_removed': 0
            }
            
            docs_with_stats = 0
            content_type_counts = {}
            
            for doc_path, metadata in processed_docs.items():
                doc_name = os.path.basename(doc_path)
                
                # Check if document has filtering stats
                if 'filtering_stats' in metadata:
                    docs_with_stats += 1
                    stats = metadata['filtering_stats']
                    
                    # Aggregate stats
                    for key in total_filtering_stats:
                        if key in stats:
                            total_filtering_stats[key] += stats[key]
                    
                    print(f"📄 {doc_name}:")
                    if 'content_type_detected' in metadata:
                        content_type = metadata['content_type_detected']
                        content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1
                        print(f"   🏷️  Content Type: {content_type}")
                    
                    filter_rate = stats.get('filter_rate_percent', 0)
                    print(f"   🎯 Filter Rate: {filter_rate}%")
                    print(f"   📊 Chunks: {stats.get('total_chunks_processed', 0)} → {stats.get('total_chunks_processed', 0) - stats.get('chunks_filtered_out', 0)}")
            
            if docs_with_stats > 0:
                print(f"\n📈 Overall Statistics:")
                total_processed = total_filtering_stats['total_chunks_processed']
                total_filtered = total_filtering_stats['chunks_filtered_out']
                overall_filter_rate = (total_filtered / total_processed * 100) if total_processed > 0 else 0
                
                print(f"   Documents with filtering: {docs_with_stats}")
                print(f"   Total chunks processed: {total_processed:,}")
                print(f"   Total chunks filtered: {total_filtered:,}")
                print(f"   Overall filter rate: {overall_filter_rate:.1f}%")
                print(f"   Duplicates removed: {total_filtering_stats['duplicates_removed']:,}")
                
                if content_type_counts:
                    print(f"\n🏷️  Content Type Distribution:")
                    for content_type, count in content_type_counts.items():
                        print(f"   • {content_type}: {count} documents")
            
        except Exception as e:
            print(f"❌ Failed to get filtering stats: {str(e)}")
    
    def do_content_types(self, line):
        """List available content types and their descriptions."""
        print(f"\n🏷️  Available Content Types:")
        print("=" * 60)
        
        descriptions = {
            ContentType.SCIENTIFIC: "Scientific reports, experiments, research papers",
            ContentType.POLICY_SOP: "Policies, SOPs, procedures, quality documents",
            ContentType.FORM: "Forms, structured documents with fields and checkboxes",
            ContentType.TEMPLATE: "Templates with placeholders and instructions",
            ContentType.GENERAL: "General documents without specific patterns"
        }
        
        for content_type in ContentType:
            description = descriptions.get(content_type, "No description available")
            print(f"• {content_type.value:15} - {description}")
    
    def do_filter_config(self, line):
        """Show current content filtering configuration."""
        print(f"\n🔧 Content Filtering Configuration:")
        print("=" * 60)
        
        config = self.config
        print(f"Content Filtering: {'✅ Enabled' if config.get('enable_content_filtering') else '❌ Disabled'}")
        print(f"Document Type Detection: {'✅ Enabled' if config.get('enable_document_type_detection') else '❌ Disabled'}")
        print(f"Deduplication: {'✅ Enabled' if config.get('enable_deduplication') else '❌ Disabled'}")
        print(f"Default Content Type: {config.get('default_content_type', 'auto')}")
        
        print(f"\n📋 Filter Settings:")
        print(f"Policy Filter Aggressive: {'✅' if config.get('policy_filter_aggressive') else '❌'}")
        print(f"Form Preserve Structure: {'✅' if config.get('form_filter_preserve_structure') else '❌'}")
        print(f"Scientific Legacy Mode: {'✅' if config.get('scientific_filter_legacy_mode') else '❌'}")
        print(f"Skip Empty Form Sections: {'✅' if config.get('skip_empty_form_sections') else '❌'}")
        print(f"Preserve Signature Blocks: {'✅' if config.get('preserve_signature_blocks') else '❌'}")
        
        print(f"\n📏 Chunk Length Limits:")
        print(f"Min Policy Chunk Length: {config.get('min_policy_chunk_length', 30)} chars")
        print(f"Min Form Chunk Length: {config.get('min_form_chunk_length', 20)} chars")
    
    def do_set_content_type(self, content_type: str):
        """Set default content type for processing."""
        if not content_type.strip():
            print("Usage: set_content_type <type>")
            print("Available types: auto, scientific, policy_sop, form, template, general")
            return
        
        valid_types = ["auto"] + [ct.value for ct in ContentType]
        
        if content_type.lower() not in valid_types:
            print(f"❌ Invalid content type: {content_type}")
            print(f"Available types: {', '.join(valid_types)}")
            return
        
        self.config['default_content_type'] = content_type.lower()
        print(f"✅ Default content type set to: {content_type.lower()}")
        print("Note: This affects new document processing. Restart for full effect.")
    
    def do_filter_examples(self, line):
        """Show content filtering examples for different document types."""
        examples = {
            "Policy/SOP Documents": [
                "What are the quality management procedures?",
                "Explain the document control policy",
                "What is the supplier evaluation process?", 
                "How are training requirements defined?",
                "What are the responsibilities in SOP-001?"
            ],
            "Form Documents": [
                "What information is collected in complaint forms?",
                "What are the required approvals for suppliers?",
                "What fields are in the evaluation form?",
                "How are incidents reported?",
                "What signatures are required?"
            ],
            "Template Documents": [
                "What sections should be included in design documents?",
                "How should concept documents be structured?",
                "What are the template requirements?",
                "How to fill out proposal templates?",
                "What placeholders need to be replaced?"
            ]
        }
        
        print(f"\n💡 Content Filtering Example Queries:")
        print("=" * 60)
        
        for category, queries in examples.items():
            print(f"\n📋 {category}:")
            for i, query in enumerate(queries, 1):
                print(f"   {i}. {query}")
        
        print(f"\n🔍 Try: query <example_text>")
        print(f"🔧 Or: detect '<sample document text>'")
    
    def do_query(self, query_text: str):
        """Enhanced query command with better display."""
        if not query_text.strip():
            print("Usage: query <text>")
            return
        
        try:
            print(f"\n🔍 Enhanced RAG Analysis for: '{query_text}'")
            print("=" * 80)
            
            # Run full RAG pipeline
            results = self.kb_api.search(
                query=query_text,
                filters=self.current_filters,
                synthesize=True
            )
            
            # Display contexts with enhanced formatting
            display_contexts_enhanced(results['contexts'])
            
            # Display enhanced synthesis
            if results.get('synthesis'):
                display_enhanced_synthesis(results['synthesis'])
            else:
                print("\n⚠️  Knowledge synthesis not available (Google API key required)")
            
            # Store results for further analysis
            self.last_results = {
                "query": query_text,
                "results": results
            }
            
            print("\n" + "=" * 80)
            print("✅ Analysis Complete! Use 'last' command to see full details.")
            
        except Exception as e:
            print(f"❌ Enhanced query failed: {str(e)}")
    
    def do_search(self, query_text: str):
        """Test hybrid search only."""
        if not query_text.strip():
            print("Usage: search <text>")
            return
        
        try:
            print(f"\n🔍 Testing hybrid search with: '{query_text}'")
            print("-" * 60)
            
            # Search without synthesis
            results = self.kb_api.search(
                query=query_text,
                filters=self.current_filters,
                synthesize=False,
                search_only=True
            )
            
            print(f"📊 Retrieved {results['num_results']} contexts")
            
            for i, ctx in enumerate(results['contexts'][:5], 1):
                score = ctx.get('initial_score', 0)
                metadata = ctx.get('metadata', {})
                content_type = metadata.get('content_type_detected', 'Unknown')
                
                print(f"\n{i}. Score: {score:.3f}")
                print(f"   Document: {metadata.get('document_id', 'Unknown')}")
                if content_type != 'Unknown':
                    print(f"   Content Type: {content_type}")
                print(f"   Text: {ctx['text'][:150]}...")
            
        except Exception as e:
            print(f"❌ Search failed: {str(e)}")
    
    def do_dense(self, query_text: str):
        """Test dense vector search only."""
        if not query_text.strip():
            print("Usage: dense <text>")
            return
        
        try:
            print(f"\n🔍 Testing dense search with: '{query_text}'")
            print("-" * 60)
            
            results = self.kb_api.dense_search(
                query=query_text,
                top_k=10,
                filters=self.current_filters
            )
            
            print(f"📊 Retrieved {results['num_results']} contexts")
            
            for i, ctx in enumerate(results['contexts'][:5], 1):
                score = ctx.get('score', 0)
                print(f"\n{i}. Score: {score:.3f}")
                print(f"   Document: {ctx.get('metadata', {}).get('document_id', 'Unknown')}")
                print(f"   Text: {ctx['text'][:150]}...")
            
        except Exception as e:
            print(f"❌ Dense search failed: {str(e)}")
    
    def do_sparse(self, query_text: str):
        """Test sparse keyword search only."""
        if not query_text.strip():
            print("Usage: sparse <text>")
            return
        
        try:
            print(f"\n🔍 Testing sparse search with: '{query_text}'")
            print("-" * 60)
            
            results = self.kb_api.sparse_search(
                query=query_text,
                top_k=10,
                filters=self.current_filters
            )
            
            print(f"📊 Retrieved {results['num_results']} contexts")
            
            for i, ctx in enumerate(results['contexts'][:5], 1):
                score = ctx.get('score', 0)
                print(f"\n{i}. Score: {score:.3f}")
                print(f"   Document: {ctx.get('metadata', {}).get('document_id', 'Unknown')}")
                print(f"   Text: {ctx['text'][:150]}...")
            
        except Exception as e:
            print(f"❌ Sparse search failed: {str(e)}")
    
    def do_rerank(self, query_text: str):
        """Test search with reranking."""
        if not query_text.strip():
            print("Usage: rerank <text>")
            return
        
        try:
            print(f"\n🔍 Testing search with reranking: '{query_text}'")
            print("-" * 60)
            
            # Search with reranking but without synthesis
            results = self.kb_api.search(
                query=query_text,
                filters=self.current_filters,
                synthesize=False
            )
            
            print(f"📊 Retrieved and reranked {results['num_results']} contexts")
            
            for i, ctx in enumerate(results['contexts'][:5], 1):
                initial_score = ctx.get('initial_score', 0)
                rerank_score = ctx.get('rerank_score', 0)
                print(f"\n{i}. Initial: {initial_score:.3f}, Rerank: {rerank_score:.3f}")
                print(f"   Document: {ctx.get('metadata', {}).get('document_id', 'Unknown')}")
                print(f"   Text: {ctx['text'][:150]}...")
            
        except Exception as e:
            print(f"❌ Reranking failed: {str(e)}")
    
    def do_analyze(self, query_text: str):
        """Enhanced analyze command focusing on synthesis."""
        if not query_text.strip():
            print("Usage: analyze <text>")
            return
        
        try:
            print(f"\n🧠 Deep Knowledge Analysis for: '{query_text}'")
            print("=" * 80)
            
            # Search with synthesis focus
            results = self.kb_api.search(
                query=query_text,
                filters=self.current_filters,
                synthesize=True
            )
            
            print(f"📊 Analyzed {results['num_results']} contexts")
            
            if results.get('synthesis'):
                display_enhanced_synthesis(results['synthesis'])
                
                # Show additional technical details
                synthesis = results['synthesis']
                if hasattr(synthesis, 'analysis_timestamp'):
                    print(f"\n⏰ Analysis completed at: {synthesis.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                
                if hasattr(synthesis, 'analysis_model'):
                    print(f"🤖 Analysis model: {synthesis.analysis_model}")
            else:
                print("⚠️  Knowledge synthesis not available (Google API key required)")
        
        except Exception as e:
            print(f"❌ Enhanced analysis failed: {str(e)}")
    
    def do_filters(self, filter_json: str):
        """Set search filters (JSON format)."""
        if not filter_json.strip():
            print("Current filters:", self.current_filters)
            print("Usage: filters <JSON>")
            print("Example: filters {\"document_id\": \"specific_doc\"}")
            print("Example: filters {\"content_type_detected\": \"PolicySOPContentFilter\"}")
            print("Clear filters: filters null")
            return
        
        try:
            if filter_json.strip().lower() == "null":
                self.current_filters = None
                print("✅ Filters cleared")
            else:
                self.current_filters = json.loads(filter_json)
                print(f"✅ Filters set: {self.current_filters}")
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {str(e)}")
    
    def do_config(self, line):
        """Show current configuration."""
        print(f"\n⚙️ RAGEngine Configuration:")
        print("=" * 60)
        print(f"Vector DB URL: {self.config['qdrant_url']}")
        print(f"Collection: {self.config['collection_name']}")
        print(f"Vector Dimensions: {self.config['vector_dimensions']}")
        print(f"Chunking Strategy: {self.config['chunking_strategy']}")
        print(f"Chunk Size: {self.config['chunk_size_tokens']} tokens")
        print(f"Chunk Overlap: {self.config['chunk_overlap_tokens']} tokens")
        print(f"Top K Dense: {self.config['top_k_dense']}")
        print(f"Top K Sparse: {self.config['top_k_sparse']}")
        print(f"Top K Rerank: {self.config['top_k_rerank']}")
        print(f"Current Filters: {self.current_filters}")
        
        print(f"\n🔧 Content Filtering:")
        print(f"Enabled: {'✅' if self.config.get('enable_content_filtering') else '❌'}")
        print(f"Auto Detection: {'✅' if self.config.get('enable_document_type_detection') else '❌'}")
        print(f"Default Type: {self.config.get('default_content_type', 'auto')}")
        
        print(f"\n🔑 API Keys:")
        print(f"OpenAI API: {'✅ Available' if self.config['openai_api_key'] else '❌ Not set'}")
        print(f"Google API: {'✅ Available' if self.config['google_api_key'] else '❌ Not set'}")
    
    def do_stats(self, line):
        """Show database statistics."""
        try:
            system_info = self.kb_api.get_system_info()
            vector_db_info = system_info.get('vector_db', {})
            
            print(f"\n📊 Database Statistics:")
            print(f"   Collection Name: {vector_db_info.get('collection_name', 'Unknown')}")
            print(f"   Vector Size: {vector_db_info.get('vector_size', 'Unknown')}")
            print(f"   Distance Metric: {vector_db_info.get('distance', 'Unknown')}")
            print(f"   Total Points: {vector_db_info.get('points_count', 0):,}")
            print(f"   Indexed Vectors: {vector_db_info.get('indexed_vectors_count', 'Unknown')}")
            
        except Exception as e:
            print(f"❌ Failed to get stats: {str(e)}")
    
    def do_ingest(self, line):
        """Run document ingestion with content filtering."""
        try:
            print(f"\n📥 Starting document ingestion with content filtering...")
            print("-" * 60)
            
            stats = self.kb_api.ingest_documents()
            
            print(f"📊 Ingestion Results:")
            print(f"   Documents processed: {stats['documents_processed']}")
            print(f"   Chunks created: {stats['chunks_created']}")
            print(f"   Embeddings generated: {stats['embeddings_generated']}")
            
            if stats['errors']:
                print(f"   Errors: {len(stats['errors'])}")
                for error in stats['errors'][:3]:
                    print(f"     - {error}")
            
            print(f"\n🎯 Use 'filter_stats' to see content filtering details")
            
        except Exception as e:
            print(f"❌ Ingestion failed: {str(e)}")
    
    def do_system(self, line):
        """Show system information."""
        try:
            system_info = self.kb_api.get_system_info()
            
            print(f"\n🔧 System Information:")
            
            # Config info
            config_info = system_info.get('config', {})
            print(f"   Collection: {config_info.get('collection_name')}")
            print(f"   Chunking Strategy: {config_info.get('chunking_strategy')}")
            print(f"   Vector Dimensions: {config_info.get('vector_dimensions')}")
            print(f"   Source Paths: {config_info.get('source_paths')}")
            
            # Content filtering info
            content_filtering_info = system_info.get('content_filtering', {})
            print(f"\n🔧 Content Filtering:")
            print(f"   Enabled: {'✅' if content_filtering_info.get('enabled') else '❌'}")
            print(f"   Auto Detection: {'✅' if content_filtering_info.get('auto_detection') else '❌'}")
            print(f"   Default Type: {content_filtering_info.get('default_type', 'auto')}")
            
            # Ingestion info
            ingestion_info = system_info.get('ingestion', {})
            print(f"\n📥 Ingestion System:")
            print(f"   Max Chunk Size: {ingestion_info.get('max_chunk_size_tokens')} tokens")
            print(f"   Chunk Overlap: {ingestion_info.get('chunk_overlap_tokens')} tokens")
            
            # RAG engine info
            rag_info = system_info.get('rag_engine', {})
            print(f"\n🔍 RAG Engine:")
            hybrid_info = rag_info.get('hybrid_searcher', {})
            print(f"   Dense Top-K: {hybrid_info.get('top_k_dense')}")
            print(f"   Sparse Top-K: {hybrid_info.get('top_k_sparse')}")
            print(f"   RRF Constant: {hybrid_info.get('rrf_k')}")
            
            reranker_info = rag_info.get('reranker')
            if reranker_info:
                print(f"   Reranker: {reranker_info.get('model')} ({'✅ Available' if reranker_info.get('api_available') else '❌ Not available'})")
            
            analyzer_info = rag_info.get('deep_analyzer')
            if analyzer_info:
                print(f"   Analyzer: {analyzer_info.get('model')} ({'✅ Available' if analyzer_info.get('api_available') else '❌ Not available'})")
            
        except Exception as e:
            print(f"❌ Failed to get system info: {str(e)}")
    
    def do_last(self, line):
        """Show details from the last query results."""
        if not self.last_results:
            print("No previous query results available")
            return
        
        print(f"\n📋 Last Query Details:")
        print(f"Query: {self.last_results['query']}")
        
        results = self.last_results['results']
        print(f"Retrieved Contexts: {results['num_results']}")
        
        contexts = results['contexts']
        if contexts:
            print(f"\nAll Retrieved Contexts:")
            for i, ctx in enumerate(contexts, 1):
                score = ctx.get('rerank_score', ctx.get('initial_score', 0))
                metadata = ctx.get('metadata', {})
                content_type = metadata.get('content_type_detected', 'Unknown')
                
                print(f"\n{i}. Score: {score:.3f}")
                print(f"   Document: {metadata.get('document_id', 'Unknown')}")
                if content_type != 'Unknown':
                    print(f"   Content Type: {content_type}")
                print(f"   Text: {ctx['text'][:100]}...")
        
        synthesis = results.get('synthesis')
        if synthesis:
            display_enhanced_synthesis(synthesis)
    
    def do_processed(self, line):
        """Show processed documents with content filtering info."""
        try:
            processed_docs = self.kb_api.get_processed_documents()
            
            print(f"\n📚 Processed Documents ({len(processed_docs)}):")
            
            if not processed_docs:
                print("   No documents processed yet. Run 'ingest' to process documents.")
                return
            
            for doc_path, metadata in list(processed_docs.items())[:10]:
                doc_name = os.path.basename(doc_path)
                success = metadata.get('processing_success', False)
                last_processed = metadata.get('last_processed', 'Unknown')
                content_type = metadata.get('content_type_detected', 'Unknown')
                
                status = "✅" if success else "❌"
                type_display = f" ({content_type})" if content_type != 'Unknown' else ""
                
                print(f"   {status} {doc_name}{type_display}")
                print(f"      Processed: {last_processed[:19]}")
                
                if 'filtering_stats' in metadata:
                    stats = metadata['filtering_stats']
                    filter_rate = stats.get('filter_rate_percent', 0)
                    print(f"      Filter Rate: {filter_rate}%")
            
            if len(processed_docs) > 10:
                print(f"   ... and {len(processed_docs) - 10} more")
                
        except Exception as e:
            print(f"❌ Failed to get processed documents: {str(e)}")
    
    def do_examples(self, line):
        """Show example queries for testing."""
        examples = [
            "What are the quality management procedures?",
            "Explain the supplier evaluation process",
            "What is the complaint investigation procedure?",
            "How are personnel qualifications determined?",
            "What are the document control requirements?",
            "What is the purpose of SOP-001?",
            "How are training requirements established?",
            "What approvals are needed for suppliers?",
            "What information is collected in forms?",
            "How are identification and traceability managed?"
        ]
        
        print(f"\n💡 Example Test Queries (Policy/SOP focused):")
        for i, query in enumerate(examples, 1):
            print(f"   {i}. {query}")
        
        print(f"\nTry: query <example_text>")
        print(f"Also try: filter_examples")
    
    def do_clear(self, line):
        """Clear all data from the collection without deleting the collection itself."""
        print("\n⚠️  This will remove ALL data from the collection but preserve the collection configuration.")
        confirm = input("Are you sure you want to clear all data? (yes/no): ").strip().lower()
        
        if confirm in ['yes', 'y']:
            try:
                print("\n🗑️  Clearing collection data...")
                success = self.kb_api.clear_collection()
                
                if success:
                    print("✅ Collection data cleared successfully!")
                    print("   The collection structure is preserved.")
                    print("   You can now run 'ingest' to reload your documents.")
                else:
                    print("❌ Failed to clear collection data.")
                    
            except Exception as e:
                print(f"❌ Error clearing collection: {str(e)}")
        else:
            print("Operation cancelled.")
    
    def do_delete_collection(self, line):
        """Delete the entire collection."""
        print("\n⚠️  This will COMPLETELY DELETE the collection and all its data.")
        print("   The collection will be recreated on next use.")
        confirm = input("Are you sure you want to delete the collection? (yes/no): ").strip().lower()
        
        if confirm in ['yes', 'y']:
            try:
                print("\n🗑️  Deleting collection...")
                success = self.kb_api.delete_collection()
                
                if success:
                    print("✅ Collection deleted successfully!")
                    print("   The collection will be recreated when you run 'ingest'.")
                else:
                    print("❌ Failed to delete collection.")
                    
            except Exception as e:
                print(f"❌ Error deleting collection: {str(e)}")
        else:
            print("Operation cancelled.")
    
    def do_recreate_collection(self, line):
        """Delete and recreate the collection with fresh configuration."""
        print("\n⚠️  This will DELETE and RECREATE the collection with fresh indexes.")
        print("   All data will be lost, but the collection will have optimal configuration.")
        confirm = input("Are you sure you want to recreate the collection? (yes/no): ").strip().lower()
        
        if confirm in ['yes', 'y']:
            try:
                print("\n🔄 Recreating collection...")
                success = self.kb_api.recreate_collection()
                
                if success:
                    print("✅ Collection recreated successfully!")
                    print("   The collection now has fresh indexes and optimal configuration.")
                    print("   You can now run 'ingest' to reload your documents.")
                else:
                    print("❌ Failed to recreate collection.")
                    
            except Exception as e:
                print(f"❌ Error recreating collection: {str(e)}")
        else:
            print("Operation cancelled.")
    
    def help_detect(self):
        """Help for detect command."""
        print("""
detect <text> - Test document type detection
    
    Analyzes the provided text to determine document type and confidence.
    Shows which content filter would be selected for processing.
    
    Example: detect "PURPOSE This SOP establishes procedures..."
        """)
    
    def help_filter_test(self):
        """Help for filter_test command."""
        print("""
filter_test <content_type> - Test a specific content filter
    
    Tests a specific content filter with sample text to see how it behaves.
    Available types: scientific, policy_sop, form, template, general
    
    Example: filter_test policy_sop
        """)
    
    def help_query(self):
        """Help for query command."""
        print("""
query <text> - Test the full RAG pipeline with a query
    
    This command runs the complete RAG pipeline:
    1. Hybrid search (dense + sparse)
    2. LLM-based reranking (if OpenAI key available)
    3. Knowledge synthesis (if Google key available)
    
    Now includes content type information in results.
    
    Example: query What are the quality procedures?
        """)
    
    def help_search(self):
        """Help for search command."""
        print("""
search <text> - Test hybrid search only
    
    This command tests only the hybrid search component,
    combining dense vector search and sparse keyword search.
    Shows content type information for each result.
    
    Example: search quality management procedures
        """)
    
    def help_filters(self):
        """Help for filters command."""
        print("""
filters <JSON> - Set search filters
    
    Set filters to constrain search results to specific criteria.
    Use JSON format for filter specification.
    
    Examples:
    filters {"document_id": "specific_document"}
    filters {"content_type_detected": "PolicySOPContentFilter"}
    filters {"category": "technical"}
    filters null  (to clear filters)
        """)
    
    def do_quit(self, line):
        """Exit the RAG shell."""
        print("👋 Goodbye!")
        return True
    
    def do_exit(self, line):
        """Exit the RAG shell."""
        return self.do_quit(line)
    
    def do_EOF(self, line):
        """Handle Ctrl+D."""
        print("\n👋 Goodbye!")
        return True

def main():
    """Main entry point for interactive RAG shell."""
    # Configure logging
    logger.add("interactive_rag.log", rotation="10 MB", level="INFO")
    
    try:
        shell = InteractiveRAGShell()
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 