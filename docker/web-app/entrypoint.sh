#!/bin/bash

# RAGEngine Web App Entrypoint Script
# Sets up proper directory structure and permissions

set -e

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
echo "  - Documents Directory: ${SOURCE_DOCUMENTS_DIR:-/app/documents}"
echo "  - Upload Directory: /app/documents/uploads"

# Verify directories exist
echo "📋 Verifying directory structure:"
ls -la /app/documents/
ls -la /app/

# Check if API keys are provided
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  WARNING: OPENAI_API_KEY not set - document processing will fail"
fi

echo "🎯 Starting application..."

# Execute the command passed to the container
exec "$@" 