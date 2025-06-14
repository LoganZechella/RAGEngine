# RAGEngine: Breath Diagnostics Knowledge Base

A powerful, production-ready multi-collection Retrieval-Augmented Generation (RAG) engine specifically designed for medical and regulatory environments. RAGEngine combines sophisticated document processing, intelligent content filtering, hybrid search capabilities, and AI-powered knowledge synthesis to create comprehensive information retrieval systems for breath diagnostics and medical device organizations.

## Key Features

### Multi-Collection Architecture
- **Specialized Collections**: Seven purpose-built collections for medical/regulatory content
  - **Current Documents**: Active documents and latest implementations
  - **Legacy/Historical**: Archived and historical documents
  - **SOPs & Policies**: Standard Operating Procedures and company policies
  - **Research Data**: Research findings, studies, and analytical data
  - **Clinical Studies**: Clinical trial data and diagnostic studies
  - **Regulatory Documents**: FDA submissions, compliance documents, and approvals
  - **Training Materials**: User guides, manuals, and training resources
- **Auto-Classification**: Intelligent document routing to appropriate collections
- **Cross-Collection Search**: Search across selected collections or all collections simultaneously

### Advanced Document Processing
- **Multi-format Support**: PDF, HTML, and text document parsing with OCR capabilities
- **Intelligent Content Filtering**: Specialized filters for different content types:
  - Scientific reports and research papers
  - Policy and SOP documents
  - Forms and structured documents
  - Templates with placeholders
  - General business documents
- **Document Type Detection**: Automatic identification and classification of document types
- **Intelligent Chunking**: Multiple strategies including hierarchical, semantic, and hybrid approaches
- **Change Detection**: Automatic detection of document updates using MD5 hashing
- **Metadata Extraction**: Rich metadata extraction including tables, structure, and document properties

### Enhanced Search & Retrieval
- **Hybrid Search**: Combines dense vector search and sparse keyword matching
- **AI-Powered Reranking**: LLM-based relevance scoring using OpenAI models
- **Collection-Specific Filtering**: Filter results by document collections and metadata
- **Multiple Search Modes**: Dense-only, sparse-only, or hybrid search options
- **Content-Aware Results**: Search results include document type and collection information

### Knowledge Synthesis & Analysis
- **Deep Analysis**: AI-powered knowledge synthesis using Google Gemini
- **Enhanced Insights**: Novel synthesis insights from cross-referencing information
- **Research Gap Identification**: Automatic identification of knowledge gaps and limitations
- **Evidence Quality Assessment**: Quality scoring of supporting evidence
- **Methodological Analysis**: Technical and theoretical analysis of content
- **Concept Extraction**: Automatic identification of key concepts and relationships
- **Topic Modeling**: Intelligent topic extraction and categorization

### Web Interface & User Experience
- **Modern Web Application**: HTMX-powered responsive web interface
- **Collection Management**: Visual management of document collections
- **File Upload & Processing**: Drag-and-drop file upload with real-time progress
- **Interactive Search**: Advanced search interface with collection filtering
- **Progress Tracking**: Real-time progress updates for long-running operations
- **System Monitoring**: Collection statistics and system health monitoring

### Developer Experience
- **Interactive Testing Tool**: Comprehensive command-line tool for testing and debugging
- **REST API**: Production-ready API with FastAPI integration
- **Docker Deployment**: Complete containerized setup with docker-compose
- **Comprehensive Logging**: Detailed logging with loguru
- **Type Safety**: Full type hints and Pydantic models throughout
- **Content Filtering APIs**: Programmatic access to document classification and filtering

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Document Sources                              │
│         (PDFs, SOPs, Clinical Studies, Regulatory Docs)         │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│              Enhanced Document Processing Pipeline               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ DocumentTypeDetector → ContentFilterFactory            │   │
│  │         ↓                      ↓                        │   │
│  │ Auto-classify content    Apply specialized filters      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                               ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ CollectionManager → MultiCollectionVectorDB            │   │
│  │  Route to collections    Store in Qdrant               │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                Multi-Collection Vector Database                  │
│        Current | Legacy | SOPs | Research | Clinical            │
│               Regulatory | Training Collections                  │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                Enhanced RAG Engine                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Multi-Collection Hybrid Searcher                        │   │
│  │  ├─ Dense search across selected collections             │   │
│  │  ├─ Sparse search with collection filtering             │   │
│  │  └─ Reciprocal Rank Fusion with collection weights     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                               ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Enhanced ReRanker (OpenAI o4-mini)                      │   │
│  │  └─ Context-aware relevance scoring                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                               ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Advanced DeepAnalyzer (Google Gemini 2.5 Pro)          │   │
│  │  ├─ Knowledge synthesis and insight generation          │   │
│  │  ├─ Research gap identification                         │   │
│  │  ├─ Evidence quality assessment                         │   │
│  │  └─ Methodological analysis                             │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    User Interfaces                              │
│     Web Application  |  Python API  |  Interactive CLI         │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- **Python 3.8+**
- **Docker and Docker Compose** (for Qdrant vector database)
- **OpenAI API key** (required for embeddings and reranking)
- **Google API key** (optional, for knowledge synthesis)

### Quick Start with Docker (Recommended)

The easiest way to get started is using the provided Docker setup:

```bash
# 1. Clone the repository
git clone <repository-url>
cd RAGEngine

# 2. Set up API keys
echo "your-openai-api-key" > secrets/openai_api_key.txt
echo "your-google-api-key" > secrets/google_api_key.txt  # Optional

# 3. Start the application
docker-compose up -d

# 4. Access the web interface
# Open your browser to: http://localhost:8080
```

### Manual Installation

For development or custom deployments:

```bash
# 1. Clone the repository
git clone <repository-url>
cd RAGEngine

# 2. Install Python dependencies
pip install -r backend/requirements.txt

# 3. Start Qdrant vector database
docker run -p 6333:6333 qdrant/qdrant

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your API keys and configuration

# 5. Test the installation
python -c "from backend.src.api.multi_collection_knowledge_base_api import MultiCollectionKnowledgeBaseAPI; print('Installation successful!')"
```

## Configuration

### Environment Variables

Create a `.env` file with the following configuration:

```env
# API Keys (Required)
OPENAI_API_KEY=your-openai-api-key-here
GOOGLE_API_KEY=your-google-api-key-here  # Optional for synthesis

# Vector Database
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=  # For cloud deployments
DEFAULT_COLLECTION=current_documents

# Document Processing
SOURCE_DOCUMENTS_DIR=./documents
CHUNKING_STRATEGY=hybrid_hierarchical_semantic
CHUNK_SIZE_TOKENS=512                    # Optimized for cost efficiency
CHUNK_OVERLAP_TOKENS=100

# Content Filtering (Advanced Features)
ENABLE_CONTENT_FILTERING=true            # Enable intelligent content filtering
ENABLE_DEDUPLICATION=true               # Remove duplicate chunks
ENABLE_DOCUMENT_TYPE_DETECTION=true     # Auto-detect document types
DEFAULT_CONTENT_TYPE=auto               # Auto-classify or set specific type
MIN_CHUNK_LENGTH=50                     # Skip very short chunks
SKIP_BOILERPLATE=true                   # Remove boilerplate text

# Collection Management
AUTO_COLLECTION_ASSIGNMENT=true         # Auto-assign documents to collections
AVAILABLE_COLLECTIONS=current_documents,legacy_documents,sop_policy,research_data,clinical_studies,regulatory_documents,training_materials

# Search Configuration
VECTOR_DIMENSIONS=1536                   # Using text-embedding-3-small for cost efficiency
TOP_K_DENSE=10
TOP_K_SPARSE=10
TOP_K_RERANK=5

# Web Application
WEB_PORT=8080
ENABLE_UPLOAD=true                      # Enable file upload via web interface
MAX_FILE_SIZE=50MB                      # Maximum file size for uploads
```

### Cost Optimization Features

RAGEngine includes advanced optimization features that can reduce OpenAI embedding costs by **70-85%** while maintaining or improving retrieval quality:

#### Immediate Cost Reduction (85% savings)
- **Optimized Embedding Model**: Uses `text-embedding-3-small` instead of `text-embedding-3-large`
- **Reduced Vector Dimensions**: 1536 dimensions instead of 3072 (85% cost reduction)
- **Smaller Chunk Sizes**: 512 tokens instead of 1024 (30-40% token reduction)

#### Content Filtering Pipeline (20-30% additional savings)
- **Medical Content Filters**: Specialized filters for scientific, regulatory, and clinical content
- **Document Type Detection**: Automatic classification of SOPs, policies, forms, and templates
- **Intelligent Deduplication**: Hash-based removal of duplicate chunks
- **Boilerplate Removal**: Automated removal of headers, footers, and repetitive content
- **Quality Thresholds**: Skip chunks that don't meet quality criteria

#### Performance Benefits
- **Faster Processing**: Fewer chunks to process and embed
- **Better Retrieval**: Smaller, more focused chunks improve precision
- **Reduced Storage**: 60% reduction in vector database storage requirements
- **Lower Memory Usage**: Significantly reduced memory footprint

### Document Collections

RAGEngine organizes documents into specialized collections:

| Collection | Purpose | Auto-Classification Rules |
|------------|---------|---------------------------|
| **Current Documents** | Active documents and latest implementations | Default for unclassified documents |
| **Legacy Documents** | Archived and historical documents | Files with dates, "legacy", "archive" keywords |
| **SOPs & Policies** | Standard Operating Procedures and policies | "SOP", "policy", "procedure" in filename/content |
| **Research Data** | Research findings and analytical data | "research", "study", "analysis", "experiment" keywords |
| **Clinical Studies** | Clinical trial data and diagnostic studies | "clinical", "trial", "patient", "diagnostic" keywords |
| **Regulatory Documents** | FDA submissions and compliance documents | "FDA", "regulatory", "compliance", "510(k)" keywords |
| **Training Materials** | User guides and training resources | "training", "manual", "guide", "tutorial" keywords |

## Usage

### Web Interface (Recommended)

The web interface provides the most user-friendly way to interact with RAGEngine:

```bash
# Start the web application
docker-compose up -d

# Access the interface
open http://localhost:8080
```

**Web Interface Features:**
- **Collection Dashboard**: View statistics and manage document collections
- **Smart Search**: Search across selected collections with real-time results
- **File Upload**: Drag-and-drop file upload with automatic collection assignment
- **Progress Tracking**: Real-time progress updates for document processing
- **System Monitoring**: View system health and collection statistics

### Python API

For programmatic access and integration:

```python
import os
from dotenv import load_dotenv
from backend.src.api.multi_collection_knowledge_base_api import MultiCollectionKnowledgeBaseAPI

# Load environment variables
load_dotenv()

# Initialize the multi-collection knowledge base
config = {
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "google_api_key": os.getenv("GOOGLE_API_KEY"),
    "qdrant_url": "http://localhost:6333",
    "default_collection": "current_documents",
    "source_paths": ["./documents"],
    "chunking_strategy": "hybrid_hierarchical_semantic",
    "chunk_size_tokens": 512,
    "top_k_rerank": 5,
    "auto_collection_assignment": True,
    "enable_content_filtering": True
}

kb = MultiCollectionKnowledgeBaseAPI(config)

# Ingest documents with automatic collection assignment
print("Ingesting documents...")
stats = kb.ingest_documents()
print(f"Processed {stats['documents_processed']} documents")
print(f"Created {stats['chunks_created']} chunks")

# Search across collections
print("\nSearching knowledge base...")
results = kb.search_collections(
    query="What are the quality management procedures?",
    collections=["sop_policy", "current_documents"],  # Search specific collections
    synthesize=True
)

print(f"Found {results['total_results']} relevant contexts")
print(f"Collections searched: {results['collections_searched']}")

# Display results with collection information
for i, ctx in enumerate(results['contexts'][:3], 1):
    score = ctx.get('rerank_score', ctx.get('initial_score', 0))
    collection = ctx.get('collection', 'Unknown')
    print(f"\n{i}. Score: {score:.3f} | Collection: {collection}")
    print(f"   Document: {ctx.get('metadata', {}).get('document_id', 'Unknown')}")
    print(f"   Text: {ctx['text'][:200]}...")

# Display AI synthesis if available
if results.get('synthesized_knowledge'):
    synthesis = results['synthesized_knowledge']
    print(f"\nAI Synthesis:")
    print(f"Summary: {synthesis.summary[:300]}...")
    
    if synthesis.key_concepts:
        print(f"Key Concepts: {[c.concept for c in synthesis.key_concepts[:3]]}")
    
    if synthesis.research_gaps:
        print(f"Research Gaps: {len(synthesis.research_gaps)} identified")
```

### Collection-Specific Operations

```python
# Upload document to specific collection
result = kb.ingest_document_to_collection(
    file_path="path/to/sop.pdf",
    target_collection="sop_policy",
    auto_classify=False  # Force assignment to specified collection
)

# Search specific collections
clinical_results = kb.search_collections(
    query="patient diagnostic accuracy",
    collections=["clinical_studies", "research_data"],
    synthesize=True
)

# Get collection statistics
stats = kb.get_collection_statistics()
for collection, info in stats['collections'].items():
    print(f"{collection}: {info['document_count']} docs, {info['chunk_count']} chunks")

# Clear specific collection
kb.clear_collection("legacy_documents")
```

### Interactive Testing Tool

RAGEngine includes a powerful interactive CLI for testing and debugging:

```bash
python interactive_rag.py
```

**Available Commands:**
```
# Search and Analysis
query <text>                    # Test full RAG pipeline
search <text>                   # Test hybrid search only
analyze <text>                  # Focus on knowledge synthesis

# Content Filtering
detect <text>                   # Test document type detection
filter_test <type>              # Test specific content filter
filter_stats                    # Show filtering statistics
content_types                   # List available content types

# Collection Management
stats                           # Show database statistics
ingest                          # Run document ingestion
processed                       # Show processed documents
clear                           # Clear collection data

# Configuration
config                          # Show current configuration
filter_config                   # Show content filtering config
filters <JSON>                  # Set search filters
examples                        # Show example queries
```

### Advanced Usage Examples

#### Filtered Search by Document Type

```python
# Search only regulatory documents
regulatory_results = kb.search_collections(
    query="FDA approval requirements",
    collections=["regulatory_documents"],
    synthesize=True
)

# Search with metadata filters
filtered_results = kb.search_collections(
    query="training procedures",
    collections=["sop_policy", "training_materials"],
    synthesize=False
)
```

#### Content Filtering Analysis

```python
from backend.src.ingestion.document_type_detector import DocumentTypeDetector
from backend.src.ingestion.content_filter_factory import ContentFilterFactory

# Analyze document type
detector = DocumentTypeDetector()
content_type, confidence = detector.detect_content_type(document_text)
print(f"Detected: {content_type.value} (confidence: {confidence:.2%})")

# Apply appropriate filter
filter_factory = ContentFilterFactory()
content_filter = filter_factory.get_filter(content_type=content_type)
cleaned_text = content_filter.clean_text(document_text)
```

#### Batch Processing with Collection Assignment

```python
# Process multiple documents with automatic collection assignment
documents = [
    "path/to/sop_001.pdf",
    "path/to/clinical_study.pdf",
    "path/to/fda_submission.pdf"
]

for doc_path in documents:
    result = kb.ingest_document_to_collection(
        file_path=doc_path,
        auto_classify=True  # Let the system decide the collection
    )
    print(f"Assigned {doc_path} to {result['collection']}")
```

## Deployment

### Docker Deployment (Production)

The provided Docker setup includes all necessary components:

```bash
# Production deployment
docker-compose up -d

# Check service health
docker-compose ps

# View logs
docker-compose logs -f web-app

# Update to latest version
docker-compose pull
docker-compose up -d
```

### Environment-Specific Configurations

#### Development
```bash
# Start with development settings
cp .env.example .env.dev
# Edit .env.dev with development configurations
docker-compose --env-file .env.dev up
```

#### Production
```bash
# Production environment file
cp .env.example .env.prod
# Configure production settings
docker-compose --env-file .env.prod up -d
```

### Scaling and Performance

#### Resource Requirements
- **Minimum**: 4GB RAM, 2 CPU cores, 20GB storage
- **Recommended**: 8GB RAM, 4 CPU cores, 50GB storage
- **High Volume**: 16GB+ RAM, 8+ CPU cores, 100GB+ storage

#### Performance Optimization
```bash
# Configure Qdrant for high performance
# Edit docker/qdrant/config.yaml:
service:
  max_workers: 8
  max_request_size_mb: 64

storage:
  performance:
    max_search_threads: 8
```

### Cloud Deployment

#### AWS ECS/EKS
```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: breath-diagnostics-ragengine
spec:
  replicas: 3
  selector:
    matchLabels:
      app: breath-diagnostics
  template:
    metadata:
      labels:
        app: breath-diagnostics
    spec:
      containers:
      - name: ragengine
        image: breath-diagnostics/ragengine:latest
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai
        - name: QDRANT_URL
          value: "http://qdrant-service:6333"
```

## Development

### Project Structure

```
RAGEngine/
├── backend/                     # Backend API and processing
│   ├── src/
│   │   ├── api/                # API interfaces
│   │   │   └── multi_collection_knowledge_base_api.py
│   │   ├── ingestion/          # Document processing
│   │   │   ├── collection_manager.py
│   │   │   ├── document_type_detector.py
│   │   │   ├── content_filter_factory.py
│   │   │   ├── multi_collection_vector_db_manager.py
│   │   │   └── knowledge_ingestion.py
│   │   ├── rag/                # RAG engine components
│   │   │   ├── hybrid_searcher.py
│   │   │   ├── reranker.py
│   │   │   ├── deep_analyzer.py
│   │   │   └── rag_engine.py
│   │   ├── models/             # Data models
│   │   │   └── data_models.py
│   │   └── utils/              # Utilities
│   ├── requirements.txt        # Python dependencies
│   └── web_app.py             # FastAPI web application
├── templates/                   # HTML templates for web interface
│   ├── dashboard.html          # Main dashboard
│   ├── fragments/              # HTMX fragments
│   └── components/             # Reusable components
├── static/                     # Static assets
├── docker/                     # Docker configurations
│   ├── web-app/               # Web application container
│   └── qdrant/                # Qdrant configuration
├── secrets/                    # API keys (not in git)
├── documents/                  # Document storage
├── tests/                      # Test suite
├── docker-compose.yml          # Docker setup
├── interactive_rag.py          # Interactive testing tool
├── main.py                     # Example usage
└── README.md                   # This file
```

### Adding New Features

#### Custom Content Filters
```python
# backend/src/ingestion/filters/custom_filter.py
from backend.src.models.content_types import ContentType
from backend.src.ingestion.content_filter_base import ContentFilterBase

class CustomContentFilter(ContentFilterBase):
    def __init__(self, content_type: ContentType = ContentType.GENERAL):
        super().__init__(content_type)
    
    def should_skip_chunk(self, text: str) -> bool:
        # Custom logic to determine if chunk should be skipped
        return len(text) < 10
    
    def clean_text(self, text: str) -> str:
        # Custom text cleaning logic
        return text.strip()
```

#### New Document Collections
```python
# Update backend/src/models/data_models.py
class DocumentCollection(str, Enum):
    # ... existing collections ...
    CUSTOM_COLLECTION = "custom_collection"

# Update backend/src/ingestion/collection_manager.py
# Add classification rules for the new collection
```

### Code Style and Standards

- **Type Hints**: All code uses comprehensive type hints
- **Pydantic Models**: Data validation with Pydantic throughout
- **Logging**: Structured logging with loguru for all components
- **Error Handling**: Comprehensive exception handling and user feedback
- **Testing**: Unit tests and integration tests for all components

### Development Setup

```bash
# Clone and setup development environment
git clone <repository-url>
cd RAGEngine
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r backend/requirements.txt

# Install development dependencies
pip install pytest pytest-asyncio black flake8 mypy

# Set up pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# Run tests
pytest tests/

# Run with development settings
python web_app.py  # Starts development server
```

## Testing

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
pytest

# Run with coverage
pytest --cov=backend/src tests/

# Run specific test categories
pytest tests/test_collections.py      # Collection management tests
pytest tests/test_content_filtering.py # Content filtering tests
pytest tests/test_search.py           # Search functionality tests
```

### Interactive Testing

The interactive CLI provides comprehensive testing capabilities:

```bash
python interactive_rag.py

# Example testing session:
RAG> detect "PURPOSE This SOP establishes procedures for quality management..."
RAG> filter_test policy_sop
RAG> query "What are the quality management procedures?"
RAG> filter_stats
RAG> examples
```

### Content Filter Testing

```bash
# Test document type detection
python -c "
from backend.src.ingestion.document_type_detector import DocumentTypeDetector
detector = DocumentTypeDetector()
content_type, confidence = detector.detect_content_type('PURPOSE This SOP...')
print(f'Type: {content_type.value}, Confidence: {confidence:.2%}')
"

# Test specific filters
python interactive_rag.py
RAG> filter_test scientific
RAG> filter_test policy_sop
RAG> filter_test form
```

## Performance & Monitoring

### Performance Metrics

- **Document Processing**: ~100 PDF pages per minute
- **Search Latency**: <200ms for hybrid search across collections
- **Knowledge Synthesis**: 2-5 seconds depending on context size and complexity
- **Memory Usage**: ~500MB baseline + 1GB per 10k documents
- **Cost Reduction**: 70-85% reduction in embedding costs with content filtering

### Monitoring and Alerts

#### System Health Endpoints
```bash
# Check application health
curl http://localhost:8080/

# Get system information
curl http://localhost:8080/system-info

# Collection statistics
curl http://localhost:8080/collection-stats

# Debug active tasks
curl http://localhost:8080/debug/search-tasks
```

#### Performance Optimization

```bash
# Monitor resource usage
docker stats

# Optimize Qdrant performance
# Edit docker/qdrant/config.yaml for your workload

# Enable query caching for frequent searches
# Set ENABLE_QUERY_CACHE=true in environment
```

### Content Filtering Analytics

The system provides detailed analytics on content filtering effectiveness:

```python
# Get filtering statistics
stats = kb.get_collection_statistics()
for collection, info in stats['collections'].items():
    filter_rate = info.get('filter_rate_percent', 0)
    print(f"{collection}: {filter_rate}% of content filtered")

# Via interactive CLI
python interactive_rag.py
RAG> filter_stats
```

## Contributing

We welcome contributions to improve the breath diagnostics capabilities and medical content processing features:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/medical-enhancement`
3. **Make changes with comprehensive tests**
4. **Run the test suite**: `pytest`
5. **Test with interactive CLI**: `python interactive_rag.py`
6. **Submit a pull request**

### Development Priorities

- **Medical Content Filters**: Enhance filters for clinical and diagnostic content
- **Regulatory Compliance**: Improve handling of FDA and regulatory documents
- **Multi-language Support**: Support for international medical documents
- **Advanced Analytics**: Enhanced metrics for medical document processing
- **Integration APIs**: APIs for medical device software integration

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **OpenAI** for embeddings and reranking models optimized for medical content
- **Google** for Gemini synthesis capabilities for complex medical analysis
- **Qdrant** for high-performance vector search supporting multiple collections
- **PyMuPDF** for excellent PDF processing of medical documents
- **FastAPI** for modern API development with medical data security considerations

## Support

- **Documentation**: Comprehensive guides in the `docs/` directory
- **Issues**: Report bugs and feature requests on GitHub Issues
- **Interactive Testing**: Use `python interactive_rag.py` for hands-on testing and debugging
- **Collection Management**: Access the web interface at `http://localhost:8080` for visual management

## Roadmap

### Short Term
- [ ] **Enhanced Medical Filters**: Specialized filters for different medical document types
- [ ] **Regulatory Templates**: Pre-built templates for common regulatory submissions
- [ ] **Advanced Analytics Dashboard**: Real-time analytics for document processing and search

### Medium Term
- [ ] **Multi-language Support**: Support for international medical documents
- [ ] **DICOM Integration**: Support for medical imaging metadata
- [ ] **Audit Trail**: Comprehensive audit logging for regulatory compliance
- [ ] **Real-time Collaboration**: Multi-user document review and annotation

### Long Term
- [ ] **AI-Powered Regulatory Assistant**: Automated regulatory compliance checking
- [ ] **Clinical Decision Support**: Integration with clinical decision support systems
- [ ] **Advanced Medical NLP**: Specialized medical entity recognition and extraction
- [ ] **Integration Ecosystem**: Pre-built integrations with popular medical device software

---

**RAGEngine: Empowering breath diagnostics and medical device organizations with intelligent knowledge management.**

For detailed deployment instructions, see [DOCKER.md](DOCKER.md) (legacy - now integrated above).
For distribution-specific setup, see [README-DISTRIBUTION.md](README-DISTRIBUTION.md) (legacy - now integrated above).

---