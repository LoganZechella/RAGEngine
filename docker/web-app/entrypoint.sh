#!/bin/bash
set -e

echo "🚀 Starting Breath Diagnostics RAGEngine Web Application"

# Wait for Qdrant to be ready
echo "⏳ Waiting for Qdrant to be ready..."
until curl -s "http://qdrant:6333/collections" > /dev/null 2>&1; do
    echo "Waiting for Qdrant..."
    sleep 2
done
echo "✅ Qdrant is ready"

# Read API keys from Docker secrets and export as environment variables
echo "🔐 Loading API keys from secrets..."

if [ -f "/run/secrets/openai_api_key" ]; then
    export OPENAI_API_KEY=$(cat /run/secrets/openai_api_key)
    echo "✅ OpenAI API key loaded from secrets"
else
    echo "❌ Error: OpenAI API key secret file not found at /run/secrets/openai_api_key"
    exit 1
fi

if [ -f "/run/secrets/google_api_key" ]; then
    export GOOGLE_API_KEY=$(cat /run/secrets/google_api_key)
    echo "✅ Google API key loaded from secrets"
else
    echo "⚠️  Warning: Google API key secret file not found at /run/secrets/google_api_key"
    echo "   Google API features will be disabled"
fi

# Set up environment
export PYTHONPATH="/app/backend:$PYTHONPATH"

# Verify required environment variables are now set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY not set after reading secrets"
    exit 1
fi

echo "🚀 Starting RAGEngine Web App initialization..."

# Create necessary directories with proper permissions
echo "📁 Setting up directory structure..."
mkdir -p /app/documents/uploads
mkdir -p /app/config
mkdir -p /app/logs

# Set proper permissions for upload directory
chmod 755 /app/documents
chmod 755 /app/documents/uploads
chmod 755 /app/config
chmod 755 /app/logs

echo "✅ Directory structure created successfully"

# Log environment info
echo "🔧 Environment Configuration:"
echo "  - Docker Environment: ${DOCKER_ENV:-false}"
echo "  - Qdrant URL: ${QDRANT_URL:-http://qdrant:6333}"
echo "  - Default Collection: ${DEFAULT_COLLECTION:-current_documents}"
echo "  - Available Collections: ${AVAILABLE_COLLECTIONS}"
echo "  - Documents Directory: ${SOURCE_DOCUMENTS_DIR:-/app/documents}"
echo "  - Upload Directory: /app/documents/uploads"
echo "  - OpenAI API Key: [LOADED]"
echo "  - Google API Key: $([ -n "$GOOGLE_API_KEY" ] && echo "[LOADED]" || echo "[NOT SET]")"

# Verify directories exist
echo "📋 Verifying directory structure:"
ls -la /app/documents/
ls -la /app/

echo "🎯 Starting application..."

# Execute the command passed to the container
exec "$@"
