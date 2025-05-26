#!/bin/sh
# =============================================================================
# RAGEngine Backend Entrypoint Script
# =============================================================================

set -e

echo "🚀 Starting RAGEngine Backend..."

# Wait for Qdrant to be ready
echo "⏳ Waiting for Qdrant to be ready..."
until curl -f "${QDRANT_URL}/healthz" >/dev/null 2>&1; do
    echo "   Qdrant not ready yet, waiting 5 seconds..."
    sleep 5
done
echo "✅ Qdrant is ready!"

# Load API keys from Docker secrets if available
if [ -f "/run/secrets/openai_api_key" ]; then
    export OPENAI_API_KEY=$(cat /run/secrets/openai_api_key)
    echo "✅ Loaded OpenAI API key from Docker secret"
fi

if [ -f "/run/secrets/google_api_key" ]; then
    export GOOGLE_API_KEY=$(cat /run/secrets/google_api_key)
    echo "✅ Loaded Google API key from Docker secret"
fi

# Validate required environment variables
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY is required but not set"
    exit 1
fi

# Create documents directory if it doesn't exist
mkdir -p "${SOURCE_DOCUMENTS_DIR:-/app/documents}"

echo "🔧 Configuration:"
echo "   • Collection: ${COLLECTION_NAME:-knowledge_base}"
echo "   • Qdrant URL: ${QDRANT_URL:-http://qdrant:6333}"
echo "   • Documents: ${SOURCE_DOCUMENTS_DIR:-/app/documents}"
echo "   • Chunking: ${CHUNKING_STRATEGY:-hybrid_hierarchical_semantic}"

# Start the application
echo "🌟 Starting FastAPI server on 0.0.0.0:8080..."
exec "$@"