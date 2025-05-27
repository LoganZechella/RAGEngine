#!/usr/bin/env python
"""
Interactive RAG Testing Tool for RAGEngine.
Command-line interface for manual RAG testing and debugging.
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
        
        # Document info
        doc_id = ctx.get('metadata', {}).get('document_id', 'Unknown')
        source = ctx.get('metadata', {}).get('source', 'Unknown')
        print(f"   📄 Document: {doc_id}")
        if source != 'Unknown':
            print(f"   📁 Source: {source}")
        
        # Text preview
        text = ctx['text']
        if len(text) > 200:
            print(f"   📝 {text[:200]}...")
        else:
            print(f"   📝 {text}")

class InteractiveRAGShell(cmd.Cmd):
    """Interactive shell for RAG testing and debugging."""
    
    intro = """
    🔍 RAGEngine Interactive Testing Shell
    ======================================
    
    Available commands:
    - query <text>        : Test full RAG pipeline with a query
    - search <text>       : Test hybrid search only
    - dense <text>        : Test dense search only
    - sparse <text>       : Test sparse search only
    - rerank <text>       : Test search with reranking
    - analyze <text>      : Test with knowledge synthesis
    - filters <filters>   : Set search filters (JSON format)
    - config             : Show current configuration
    - stats              : Show database statistics
    - ingest             : Run document ingestion
    - system             : Show system information
    - last               : Show details from last query
    - examples           : Show example queries
    - clear              : Clear all data from collection
    - delete_collection  : Delete entire collection
    - recreate_collection: Recreate collection with fresh config
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
    
    def setup_rag_engine(self):
        """Initialize RAG engine components."""
        try:
            print("Initializing RAGEngine...")
            
            # Configuration
            config = {
                "openai_api_key": os.getenv("OPENAI_API_KEY"),
                "google_api_key": os.getenv("GOOGLE_API_KEY"),
                "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
                "qdrant_api_key": os.getenv("QDRANT_API_KEY"),
                "collection_name": os.getenv("COLLECTION_NAME", "knowledge_base"),
                "source_paths": [os.getenv("SOURCE_DOCUMENTS_DIR", "./documents")],
                "chunking_strategy": os.getenv("CHUNKING_STRATEGY", "paragraph"),
                "chunk_size_tokens": int(os.getenv("CHUNK_SIZE_TOKENS", "512")),
                "chunk_overlap_tokens": int(os.getenv("CHUNK_OVERLAP_TOKENS", "100")),
                "vector_dimensions": int(os.getenv("VECTOR_DIMENSIONS", "1536")),
                "top_k_dense": int(os.getenv("TOP_K_DENSE", "10")),
                "top_k_sparse": int(os.getenv("TOP_K_SPARSE", "10")),
                "top_k_rerank": int(os.getenv("TOP_K_RERANK", "5"))
            }
            
            # Validate required configuration
            if not config["openai_api_key"]:
                print("⚠️  Warning: OPENAI_API_KEY not set. Some features may not work.")
            
            if not config["google_api_key"]:
                print("⚠️  Warning: GOOGLE_API_KEY not set. Knowledge synthesis will not be available.")
            
            self.kb_api = KnowledgeBaseAPI(config)
            self.config = config
            
            print("✅ RAGEngine initialized successfully!")
            
        except Exception as e:
            print(f"❌ Failed to initialize RAGEngine: {str(e)}")
            print("Make sure Qdrant is running and API keys are set.")
            sys.exit(1)
    
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
                print(f"\n{i}. Score: {score:.3f}")
                print(f"   Document: {ctx.get('metadata', {}).get('document_id', 'Unknown')}")
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
            print("Example: filters {\"category\": \"technical\"}")
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
        print(f"   Vector DB URL: {self.config['qdrant_url']}")
        print(f"   Collection: {self.config['collection_name']}")
        print(f"   Vector Dimensions: {self.config['vector_dimensions']}")
        print(f"   Chunking Strategy: {self.config['chunking_strategy']}")
        print(f"   Chunk Size: {self.config['chunk_size_tokens']} tokens")
        print(f"   Chunk Overlap: {self.config['chunk_overlap_tokens']} tokens")
        print(f"   Top K Dense: {self.config['top_k_dense']}")
        print(f"   Top K Sparse: {self.config['top_k_sparse']}")
        print(f"   Top K Rerank: {self.config['top_k_rerank']}")
        print(f"   Current Filters: {self.current_filters}")
        print(f"   OpenAI API: {'✅ Available' if self.config['openai_api_key'] else '❌ Not set'}")
        print(f"   Google API: {'✅ Available' if self.config['google_api_key'] else '❌ Not set'}")
    
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
        """Run document ingestion."""
        try:
            print(f"\n📥 Starting document ingestion...")
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
                print(f"\n{i}. Score: {score:.3f}")
                print(f"   Document: {ctx.get('metadata', {}).get('document_id', 'Unknown')}")
                print(f"   Text: {ctx['text'][:100]}...")
        
        synthesis = results.get('synthesis')
        if synthesis:
            display_enhanced_synthesis(synthesis)
    
    def do_processed(self, line):
        """Show processed documents."""
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
                status = "✅" if success else "❌"
                print(f"   {status} {doc_name} (processed: {last_processed[:19]})")
            
            if len(processed_docs) > 10:
                print(f"   ... and {len(processed_docs) - 10} more")
                
        except Exception as e:
            print(f"❌ Failed to get processed documents: {str(e)}")
    
    def do_examples(self, line):
        """Show example queries for testing."""
        examples = [
            "What is machine learning?",
            "Explain neural networks and deep learning",
            "How does natural language processing work?",
            "What are the applications of artificial intelligence?",
            "Compare supervised and unsupervised learning",
            "What is the difference between classification and regression?",
            "How do convolutional neural networks work?",
            "What are the challenges in computer vision?",
            "Explain reinforcement learning concepts",
            "What are transformers in deep learning?"
        ]
        
        print(f"\n💡 Example Test Queries:")
        for i, query in enumerate(examples, 1):
            print(f"   {i}. {query}")
        
        print(f"\nTry: query <example_text>")
    
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
    
    def help_query(self):
        """Help for query command."""
        print("""
query <text> - Test the full RAG pipeline with a query
    
    This command runs the complete RAG pipeline:
    1. Hybrid search (dense + sparse)
    2. LLM-based reranking (if OpenAI key available)
    3. Knowledge synthesis (if Google key available)
    
    Example: query What is machine learning?
        """)
    
    def help_search(self):
        """Help for search command."""
        print("""
search <text> - Test hybrid search only
    
    This command tests only the hybrid search component,
    combining dense vector search and sparse keyword search.
    
    Example: search machine learning algorithms
        """)
    
    def help_filters(self):
        """Help for filters command."""
        print("""
filters <JSON> - Set search filters
    
    Set filters to constrain search results to specific criteria.
    Use JSON format for filter specification.
    
    Examples:
    filters {"document_id": "specific_document"}
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