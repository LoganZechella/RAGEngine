#!/bin/bash
# =============================================================================
# RAGEngine Docker Cleanup Script
# Clean up Docker resources for RAGEngine
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}🧹 RAGEngine Docker Cleanup${NC}"
echo "=================================================="

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Ask for confirmation
confirm_cleanup() {
    echo -e "${YELLOW}This will remove all RAGEngine Docker containers, images, and volumes.${NC}"
    echo -e "${YELLOW}Your data in the data/ directory will be preserved.${NC}"
    echo
    read -p "Are you sure you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cleanup cancelled."
        exit 0
    fi
}

# Stop and remove containers
stop_containers() {
    echo "🛑 Stopping RAGEngine containers..."
    
    cd "$PROJECT_DIR"
    
    if docker compose ps -q | grep -q .; then
        docker compose down
        print_status "Containers stopped and removed"
    else
        print_status "No running containers found"
    fi
}

# Remove Docker images
remove_images() {
    echo "🗑️  Removing RAGEngine Docker images..."
    
    # Remove custom built images
    if docker images | grep -q "ragengine"; then
        docker images | grep "ragengine" | awk '{print $3}' | xargs docker rmi -f 2>/dev/null || true
        print_status "RAGEngine images removed"
    else
        print_status "No RAGEngine images found"
    fi
    
    # Optionally remove Qdrant image
    read -p "Remove Qdrant image as well? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker rmi qdrant/qdrant:latest 2>/dev/null || true
        print_status "Qdrant image removed"
    fi
}

# Remove Docker volumes
remove_volumes() {
    echo "🗂️  Removing Docker volumes..."
    
    cd "$PROJECT_DIR"
    
    # Remove named volumes
    docker volume ls | grep "ragengine" | awk '{print $2}' | xargs docker volume rm 2>/dev/null || true
    
    print_status "Docker volumes removed"
}

# Clean up Docker system
cleanup_system() {
    echo "🧽 Cleaning up Docker system..."
    
    # Remove unused networks
    docker network prune -f
    
    # Remove dangling images
    docker image prune -f
    
    # Remove unused volumes
    docker volume prune -f
    
    print_status "Docker system cleaned"
}

# Optional: Remove data directories
remove_data() {
    echo
    read -p "Remove local data directories? This will delete all your documents and vector data! (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$PROJECT_DIR/data"
        rm -rf "$PROJECT_DIR/logs"
        print_warning "Local data directories removed"
    else
        print_status "Local data directories preserved"
    fi
}

# Show cleanup summary
show_summary() {
    echo
    echo "=================================================="
    echo -e "${GREEN}🎉 Cleanup completed!${NC}"
    echo
    echo "What was cleaned:"
    echo "  ✅ RAGEngine containers stopped and removed"
    echo "  ✅ RAGEngine Docker images removed"
    echo "  ✅ Docker volumes removed"
    echo "  ✅ Unused Docker resources cleaned"
    echo
    echo "To start fresh:"
    echo "  1. Run: ./scripts/docker-setup.sh"
    echo "  2. Configure your API keys"
    echo "  3. Run: docker compose up"
}

# Main cleanup function
main() {
    confirm_cleanup
    stop_containers
    remove_images
    remove_volumes
    cleanup_system
    remove_data
    show_summary
}

# Run main function
main "$@"