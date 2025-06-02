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

echo "🔧 Configuration:"
echo "  • Qdrant URL: ${QDRANT_URL}"
echo "  • Default Collection: ${DEFAULT_COLLECTION}"
echo "  • Available Collections: ${AVAILABLE_COLLECTIONS}"
echo "  • Document Directory: ${SOURCE_DOCUMENTS_DIR}"
echo "  • OpenAI API Key: [LOADED]"
echo "  • Google API Key: $([ -n "$GOOGLE_API_KEY" ] && echo "[LOADED]" || echo "[NOT SET]")"

# Execute the command
echo "🚀 Starting web application..."
exec "$@" 