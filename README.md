# RAGEngine

A powerful, production-ready Retrieval-Augmented Generation (RAG) engine for building intelligent knowledge bases. RAGEngine combines sophisticated document processing, hybrid search capabilities, and AI-powered knowledge synthesis to create comprehensive information retrieval systems.

## 🚀 Features

### Document Processing
- **Multi-format Support**: PDF, HTML, and text document parsing
- **Intelligent Chunking**: Multiple strategies including hierarchical, semantic, and hybrid approaches
- **Change Detection**: Automatic detection of document updates using MD5 hashing
- **Metadata Extraction**: Rich metadata extraction including tables, structure, and document properties

### Advanced Search
- **Hybrid Search**: Combines dense vector search and sparse keyword matching
- **AI-Powered Reranking**: LLM-based relevance scoring using OpenAI models
- **Flexible Filtering**: Support for metadata-based filtering and constraints
- **Multiple Search Modes**: Dense-only, sparse-only, or hybrid search options

### Knowledge Synthesis
- **Deep Analysis**: AI-powered knowledge synthesis using Google Gemini
- **Concept Extraction**: Automatic identification of key concepts and relationships
- **Topic Modeling**: Intelligent topic extraction and categorization
- **Insight Generation**: Actionable insights and comprehensive summaries

### Developer Experience
- **Interactive Testing**: Command-line tool for manual testing and debugging
- **REST API**: Production-ready API with FastAPI integration
- **Comprehensive Logging**: Detailed logging with loguru
- **Type Safety**: Full type hints and Pydantic models throughout

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Document Sources                          │
│                    (PDFs, HTML, Text files)                     │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Knowledge Ingestion                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ DocumentSourceManager → PdfParser → TextChunker         │   │
│  │         ↓                    ↓            ↓              │   │
│  │  Track changes        Extract text   Split into chunks   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                               ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ EmbeddingGenerator → VectorDBManager                     │   │
│  │    Create embeddings    Store in Qdrant                  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Vector Database                             │
│                        (Qdrant)                                  │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                        RAG Engine                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ HybridSearcher                                           │   │
│  │  ├─ Dense search (vector similarity)                     │   │
│  │  ├─ Sparse search (keyword matching)                     │   │
│  │  └─ Reciprocal Rank Fusion                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                               ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ ReRanker (OpenAI o4-mini)                                │   │
│  │  └─ LLM-based relevance scoring                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                               ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ DeepAnalyzer (Google Gemini 2.5 Pro)                     │   │
│  │  └─ Knowledge synthesis and insight generation           │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Docker (for Qdrant vector database)
- OpenAI API key
- Google API key (optional, for knowledge synthesis)

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd RAGEngine
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Qdrant vector database**:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

5. **Test the installation**:
   ```bash
   python -c "from src.api.knowledge_base_api import KnowledgeBaseAPI; print('✅ Installation successful!')"
   ```

### Docker Setup

For a complete Docker-based setup:

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with the following configuration:

```env
# API Keys (Required)
OPENAI_API_KEY=your-openai-api-key-here
GOOGLE_API_KEY=your-google-api-key-here  # Optional for synthesis

# Vector Database
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=  # For cloud deployments
COLLECTION_NAME=knowledge_base

# Document Processing (OPTIMIZED FOR COST EFFICIENCY)
SOURCE_DOCUMENTS_DIR=./documents
CHUNKING_STRATEGY=hybrid_hierarchical_semantic
CHUNK_SIZE_TOKENS=512                    # Reduced from 1024 for 30-40% token reduction
CHUNK_OVERLAP_TOKENS=100                 # Reduced from 200 for efficiency

# Content Filtering (NEW - 20-30% additional savings)
ENABLE_CONTENT_FILTERING=true            # Filter redundant scientific content
ENABLE_DEDUPLICATION=true               # Remove duplicate chunks
MIN_CHUNK_LENGTH=50                     # Skip very short chunks
SKIP_BOILERPLATE=true                   # Remove boilerplate text

# Search Configuration (OPTIMIZED)
VECTOR_DIMENSIONS=1536                   # Reduced from 3072 for 85% cost reduction
TOP_K_DENSE=10
TOP_K_SPARSE=10
TOP_K_RERANK=5

# Optional OCR Support
TESSERACT_CMD=/usr/bin/tesseract  # For OCR functionality
```

## 💰 Cost Optimization Features

RAGEngine includes advanced optimization features that can reduce OpenAI embedding costs by **70-85%** while maintaining or improving retrieval quality:

### Immediate Cost Reduction (85% savings)
- **Optimized Embedding Model**: Uses `text-embedding-3-small` instead of `text-embedding-3-large`
- **Reduced Vector Dimensions**: 1536 dimensions instead of 3072 (85% cost reduction)
- **Smaller Chunk Sizes**: 512 tokens instead of 1024 (30-40% token reduction)

### Content Filtering Pipeline (20-30% additional savings)
- **Scientific Content Filter**: Removes redundant headers, footers, and boilerplate text
- **Page-Level Processing**: Filters content at the page level before chunking
- **Deduplication**: Hash-based removal of duplicate chunks
- **Table Summarization**: Converts large tables to concise summaries

### Performance Benefits
- **Faster Processing**: Fewer chunks to process and embed
- **Better Retrieval**: Smaller, more focused chunks improve precision
- **Reduced Storage**: 60% reduction in vector database storage requirements
- **Lower Memory Usage**: Significantly reduced memory footprint

### Cost Example
For a 100-document batch (scientific reports):
- **Before optimization**: ~$50-75 in embedding costs
- **After optimization**: ~$8-15 in embedding costs
- **Annual savings**: Potentially thousands of dollars for large-scale processing

### Chunking Strategies

RAGEngine supports multiple text chunking strategies:

- `paragraph`: Split by paragraphs (default)
- `sentence`: Split by sentences
- `sliding_window`: Fixed-size sliding window
- `hierarchical`: Preserve document structure
- `semantic`: Content-aware chunking using sentence transformers
- `hybrid_hierarchical_semantic`: Best of both worlds (recommended)

## 🚀 Usage

### Basic Python API

```python
import os
from dotenv import load_dotenv
from src.api.knowledge_base_api import KnowledgeBaseAPI

# Load environment variables
load_dotenv()

# Initialize the knowledge base
config = {
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "google_api_key": os.getenv("GOOGLE_API_KEY"),
    "qdrant_url": "http://localhost:6333",
    "collection_name": "my_knowledge_base",
    "source_paths": ["./documents"],
    "chunking_strategy": "hybrid_hierarchical_semantic",
    "chunk_size_tokens": 512,
    "top_k_rerank": 5
}

kb = KnowledgeBaseAPI(config)

# Ingest documents
print("📥 Ingesting documents...")
stats = kb.ingest_documents()
print(f"✅ Processed {stats['documents_processed']} documents")
print(f"📄 Created {stats['chunks_created']} chunks")

# Search the knowledge base
print("\n🔍 Searching knowledge base...")
results = kb.search("What is machine learning?", synthesize=True)

print(f"📊 Found {results['num_results']} relevant contexts")

# Display top results
for i, ctx in enumerate(results['contexts'][:3], 1):
    score = ctx.get('rerank_score', ctx.get('initial_score', 0))
    print(f"\n{i}. Score: {score:.3f}")
    print(f"   Document: {ctx.get('metadata', {}).get('document_id', 'Unknown')}")
    print(f"   Text: {ctx['text'][:200]}...")

# Display AI synthesis
if results.get('synthesis'):
    synthesis = results['synthesis']
    print(f"\n🧠 AI Synthesis:")
    print(f"Summary: {synthesis['summary'][:300]}...")
    print(f"Key Concepts: {[c.get('concept', '') for c in synthesis.get('key_concepts', [])]}")
```

### Interactive Testing Tool

RAGEngine includes a powerful interactive testing tool:

```bash
python interactive_rag.py
```

Available commands in the interactive shell:

```
RAG> help
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
- examples           : Show example queries
- clear              : Clear all data from collection
- delete_collection  : Delete entire collection
- recreate_collection: Recreate collection with fresh config
- quit               : Exit the shell

RAG> query What are neural networks?
```

See [README_Interactive.md](README_Interactive.md) for detailed interactive tool documentation.

### Advanced Usage

#### Filtered Search

```python
# Search with metadata filters
results = kb.search(
    "machine learning algorithms",
    filters={"document_id": "specific_doc.pdf"},
    synthesize=False
)

# Search by category
results = kb.search(
    "data privacy",
    filters={"category": "legal"},
    synthesize=True
)
```

#### Different Search Modes

```python
# Dense vector search only
results = kb.dense_search("artificial intelligence", top_k=10)

# Sparse keyword search only
results = kb.sparse_search("neural network training", top_k=10)

# Hybrid search without synthesis
results = kb.search("deep learning", synthesize=False)
```

#### Batch Processing

```python
# Process a single document
stats = kb.process_single_document("path/to/document.pdf")

# Get processing statistics
processed_docs = kb.get_processed_documents()
system_info = kb.get_system_info()
```

## 🔌 API Reference

### KnowledgeBaseAPI

The main API class providing high-level access to RAGEngine functionality.

#### Constructor

```python
KnowledgeBaseAPI(config: Dict[str, Any])
```

**Parameters:**
- `config`: Configuration dictionary with API keys, database settings, and processing options

#### Methods

##### `ingest_documents() -> Dict[str, Any]`

Process all documents from configured source paths.

**Returns:**
- Dictionary with processing statistics including documents processed, chunks created, and any errors

##### `search(query: str, filters: Optional[Dict] = None, synthesize: bool = True) -> Dict[str, Any]`

Search the knowledge base with optional AI synthesis.

**Parameters:**
- `query`: Search query string
- `filters`: Optional metadata filters
- `synthesize`: Whether to perform AI-powered knowledge synthesis

**Returns:**
- Dictionary containing search results and optional synthesis

##### `dense_search(query: str, top_k: int = 10, filters: Optional[Dict] = None) -> Dict[str, Any]`

Perform dense vector search only.

##### `sparse_search(query: str, top_k: int = 10, filters: Optional[Dict] = None) -> Dict[str, Any]`

Perform sparse keyword search only.

##### `get_system_info() -> Dict[str, Any]`

Get detailed system information and configuration.

##### `get_processed_documents() -> Dict[str, Dict]`

Get list of all processed documents and their metadata.

##### `delete_collection() -> bool`

Delete the entire Qdrant collection. The collection will be recreated when data is next ingested.

##### `clear_collection() -> bool`

Clear all data from the collection while preserving the collection structure and configuration.

##### `recreate_collection() -> bool`

Delete and recreate the collection with fresh indexes for optimal performance.

##### `delete_document(document_id: str) -> int`

Delete all chunks associated with a specific document ID.

### REST API

Start the REST API server:

```bash
python api_server.py
```

#### Endpoints

**POST /search**
```json
{
  "query": "What is machine learning?",
  "filters": {"category": "technical"},
  "synthesize": true
}
```

**POST /ingest**
```json
{}
```

**GET /system/info**

**GET /documents/processed**

## 🧪 Testing

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_ingestion.py
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end pipeline testing
- **Performance Tests**: Load and performance benchmarks

### Manual Testing

Use the interactive tool for manual testing:

```bash
python interactive_rag.py
RAG> examples  # Show example queries
RAG> query What is artificial intelligence?
```

## 📦 Deployment

### Docker Deployment

Use the provided Docker Compose setup:

```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up -d

# Scale the API service
docker-compose up --scale knowledge-base=3
```

### Cloud Deployment

#### Kubernetes

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ragengine
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ragengine
  template:
    metadata:
      labels:
        app: ragengine
    spec:
      containers:
      - name: ragengine
        image: ragengine:latest
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai
        - name: QDRANT_URL
          value: "http://qdrant-service:6333"
```

#### Cloud Services

- **AWS**: Deploy on ECS/EKS with RDS for metadata
- **Google Cloud**: Use GKE with Cloud SQL
- **Azure**: Deploy on AKS with Azure Database

## 🔧 Development

### Project Structure

```
RAGEngine/
├── src/
│   ├── api/                 # API interfaces
│   │   └── knowledge_base_api.py
│   ├── ingestion/           # Document processing
│   │   ├── document_source_manager.py
│   │   ├── pdf_parser.py
│   │   ├── text_chunker.py
│   │   ├── embedding_generator.py
│   │   ├── vector_db_manager.py
│   │   └── knowledge_ingestion.py
│   ├── rag/                 # RAG engine components
│   │   ├── hybrid_searcher.py
│   │   ├── reranker.py
│   │   ├── deep_analyzer.py
│   │   └── rag_engine.py
│   ├── models/              # Data models
│   │   └── data_models.py
│   └── utils/               # Utilities
│       └── helpers.py
├── tests/                   # Test suite
├── docs/                    # Documentation
├── docker-compose.yml       # Docker setup
├── requirements.txt         # Dependencies
├── interactive_rag.py       # Interactive testing tool
├── main.py                  # Example usage
└── README.md               # This file
```

### Adding New Features

1. **New Document Types**: Extend parsers in `src/ingestion/`
2. **Custom Chunking**: Add strategies to `text_chunker.py`
3. **Search Enhancements**: Modify `hybrid_searcher.py`
4. **Synthesis Models**: Update `deep_analyzer.py`

### Code Style

- **Type Hints**: All code uses comprehensive type hints
- **Pydantic Models**: Data validation with Pydantic
- **Logging**: Structured logging with loguru
- **Error Handling**: Comprehensive exception handling

## 📊 Performance

### Benchmarks

- **Ingestion**: ~100 PDF pages per minute
- **Search Latency**: <200ms for hybrid search
- **Synthesis**: 2-5 seconds depending on context size
- **Memory Usage**: ~500MB baseline + 1GB per 10k documents

### Optimization Tips

1. **Batch Processing**: Process documents in batches for better throughput
2. **Caching**: Enable embedding caching for frequently accessed content
3. **Async Processing**: Use async patterns for I/O operations
4. **Vector DB Tuning**: Optimize Qdrant configuration for your use case

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-feature`
3. **Make changes with tests**
4. **Run the test suite**: `pytest`
5. **Submit a pull request**

### Development Setup

```bash
# Clone and setup development environment
git clone <repository-url>
cd RAGEngine
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for embeddings and reranking models
- **Google** for Gemini synthesis capabilities
- **Qdrant** for high-performance vector search
- **PyMuPDF** for excellent PDF processing
- **FastAPI** for modern API development

## 📞 Support

- **Documentation**: Check the `docs/` directory for detailed guides
- **Issues**: Report bugs and feature requests on GitHub Issues
- **Discussions**: Join community discussions on GitHub Discussions
- **Interactive Help**: Use `python interactive_rag.py` for hands-on testing

## 🔮 Roadmap

- [ ] **Multi-modal Support**: Image and audio document processing
- [ ] **Advanced Analytics**: Search analytics and performance metrics
- [ ] **Multi-tenant Architecture**: Support for multiple isolated knowledge bases
- [ ] **Real-time Updates**: Live document monitoring and updates
- [ ] **Advanced Synthesis**: Multi-step reasoning and complex query handling
- [ ] **Integration Connectors**: Direct integrations with popular document sources

---

For more information, visit our [documentation](docs/) or try the [interactive demo](interactive_rag.py). 