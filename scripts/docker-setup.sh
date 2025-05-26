#!/bin/bash
# =============================================================================
# RAGEngine Docker Setup Script
# Automated setup for Docker deployment
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

echo -e "${BLUE}🚀 RAGEngine Docker Setup${NC}"
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

# Check if Docker is installed and running
check_docker() {
    echo "🔍 Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker Desktop first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker Desktop."
        exit 1
    fi
    
    print_status "Docker is installed and running"
}

# Check if Docker Compose is available
check_docker_compose() {
    echo "🔍 Checking Docker Compose..."
    
    if ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not available. Please update Docker Desktop."
        exit 1
    fi
    
    print_status "Docker Compose is available"
}

# Create necessary directories
create_directories() {
    echo "📁 Creating data directories..."
    
    mkdir -p "$PROJECT_DIR/data/qdrant"
    mkdir -p "$PROJECT_DIR/data/documents"
    mkdir -p "$PROJECT_DIR/data/manifests"
    mkdir -p "$PROJECT_DIR/data/logs"
    mkdir -p "$PROJECT_DIR/documents"
    mkdir -p "$PROJECT_DIR/config"
    mkdir -p "$PROJECT_DIR/logs"
    
    print_status "Data directories created"
}

# Setup environment file
setup_environment() {
    echo "⚙️  Setting up environment configuration..."
    
    if [ ! -f "$PROJECT_DIR/.env" ]; then
        if [ -f "$PROJECT_DIR/.env.docker.example" ]; then
            cp "$PROJECT_DIR/.env.docker.example" "$PROJECT_DIR/.env"
            print_warning "Created .env file from template. Please edit it with your API keys."
        else
            print_error ".env.docker.example not found. Cannot create .env file."
            exit 1
        fi
    else
        print_status "Environment file already exists"
    fi
}

# Check API keys
check_api_keys() {
    echo "🔑 Checking API keys..."
    
    if [ -f "$PROJECT_DIR/secrets/openai_api_key.txt" ]; then
        if grep -q "your_openai_api_key_here" "$PROJECT_DIR/secrets/openai_api_key.txt"; then
            print_warning "Please update your OpenAI API key in secrets/openai_api_key.txt"
        else
            print_status "OpenAI API key configured"
        fi
    else
        print_warning "OpenAI API key file not found. Please create secrets/openai_api_key.txt"
    fi
    
    if [ -f "$PROJECT_DIR/secrets/google_api_key.txt" ]; then
        if grep -q "your_google_api_key_here" "$PROJECT_DIR/secrets/google_api_key.txt"; then
            print_warning "Google API key not configured (optional for synthesis)"
        else
            print_status "Google API key configured"
        fi
    else
        print_warning "Google API key file not found (optional for synthesis)"
    fi
}

# Build Docker images
build_images() {
    echo "🏗️  Building Docker images..."
    
    cd "$PROJECT_DIR"
    
    echo "Building backend image..."
    docker compose build backend
    
    echo "Building frontend image..."
    docker compose build frontend
    
    print_status "Docker images built successfully"
}

# Pull external images
pull_images() {
    echo "📥 Pulling external Docker images..."
    
    cd "$PROJECT_DIR"
    docker compose pull qdrant
    
    print_status "External images pulled"
}

# Validate configuration
validate_config() {
    echo "✅ Validating Docker Compose configuration..."
    
    cd "$PROJECT_DIR"
    if docker compose config &> /dev/null; then
        print_status "Docker Compose configuration is valid"
    else
        print_error "Docker Compose configuration is invalid"
        docker compose config
        exit 1
    fi
}

# Main setup function
main() {
    echo "Starting RAGEngine Docker setup..."
    echo
    
    check_docker
    check_docker_compose
    create_directories
    setup_environment
    check_api_keys
    validate_config
    pull_images
    build_images
    
    echo
    echo "=================================================="
    echo -e "${GREEN}🎉 Setup completed successfully!${NC}"
    echo
    echo "Next steps:"
    echo "1. Edit your API keys in the secrets/ directory"
    echo "2. Review and customize .env file if needed"
    echo "3. Start the application with: docker compose up"
    echo
    echo "Useful commands:"
    echo "  • Start:           docker compose up -d"
    echo "  • Stop:            docker compose down"
    echo "  • View logs:       docker compose logs -f"
    echo "  • Development:     docker compose -f docker-compose.yml -f docker-compose.dev.yml up"
    echo "  • Production:      docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d"
    echo
    echo "Access URLs:"
    echo "  • Frontend:        http://localhost:3000"
    echo "  • Backend API:     http://localhost:8080"
    echo "  • Qdrant:          http://localhost:6333"
}

# Run main function
main "$@"