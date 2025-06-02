#!/bin/bash

# RAGEngine Docker Image Loader
# Run this script on the target machine to load the exported images

echo "🚀 Loading RAGEngine Docker Images..."

# Check if files exist
if [ ! -f "ragengine-web-app.tar" ]; then
    echo "❌ Error: ragengine-web-app.tar not found"
    exit 1
fi

if [ ! -f "qdrant.tar" ]; then
    echo "⚠️  Warning: qdrant.tar not found - will pull from Docker Hub instead"
    PULL_QDRANT=true
else
    PULL_QDRANT=false
fi

# Load main application image
echo "📦 Loading RAGEngine Web App image..."
docker load -i ragengine-web-app.tar

if [ $? -eq 0 ]; then
    echo "✅ RAGEngine Web App image loaded successfully"
else
    echo "❌ Failed to load RAGEngine Web App image"
    exit 1
fi

# Load or pull Qdrant image
if [ "$PULL_QDRANT" = true ]; then
    echo "📦 Pulling Qdrant image from Docker Hub..."
    docker pull qdrant/qdrant:latest
else
    echo "📦 Loading Qdrant image..."
    docker load -i qdrant.tar
fi

if [ $? -eq 0 ]; then
    echo "✅ Qdrant image ready"
else
    echo "❌ Failed to load/pull Qdrant image"
    exit 1
fi

# Verify images are available
echo ""
echo "🔍 Verifying loaded images:"
docker images | grep -E "(ragengine|qdrant)"

echo ""
echo "✅ All images loaded successfully!"
echo "📋 Next steps:"
echo "   1. Copy env.distribution to .env"
echo "   2. Edit .env with your API keys"
echo "   3. Run: docker-compose -f docker-compose.distribution.yml up -d"
echo "   4. Access: http://localhost:8080" 