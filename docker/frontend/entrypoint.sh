#!/bin/sh
# =============================================================================
# RAGEngine Frontend Entrypoint Script
# =============================================================================

set -e

echo "🚀 Starting RAGEngine Frontend..."

# Wait for backend to be ready
echo "⏳ Waiting for backend to be ready..."
BACKEND_URL=${VITE_API_BASE_URL:-http://backend:8080}
until curl -f "${BACKEND_URL}/system-info" >/dev/null 2>&1; do
    echo "   Backend not ready yet, waiting 5 seconds..."
    sleep 5
done
echo "✅ Backend is ready!"

echo "🔧 Configuration:"
echo "   • Backend URL: ${VITE_API_BASE_URL:-http://backend:8080}"
echo "   • Environment: ${NODE_ENV:-production}"
echo "   • Port: 3000"

# Start the application
echo "🌟 Starting SvelteKit server on 0.0.0.0:3000..."
exec "$@"