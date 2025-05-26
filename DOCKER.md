# 🐳 RAGEngine Docker Deployment Guide

This guide provides comprehensive instructions for deploying RAGEngine using Docker containers. The setup includes a FastAPI backend, SvelteKit frontend, and Qdrant vector database, all orchestrated with Docker Compose.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Development](#development)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd RAGEngine

# 2. Run the setup script
./scripts/docker-setup.sh

# 3. Configure your API keys
echo "your-openai-api-key" > secrets/openai_api_key.txt
echo "your-google-api-key" > secrets/google_api_key.txt  # Optional

# 4. Start the application
docker compose up -d

# 5. Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8080
# Qdrant: http://localhost:6333
```

## 🏗️ Architecture Overview

RAGEngine uses a 3-container architecture:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │    Qdrant       │
│   (SvelteKit)   │◄──►│   (FastAPI)     │◄──►│  (Vector DB)    │
│   Port: 3000    │    │   Port: 8080    │    │   Port: 6333    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Container Details

| Container | Technology | Purpose | Port |
|-----------|------------|---------|------|
| **Frontend** | SvelteKit + Node.js | Web interface | 3000 |
| **Backend** | FastAPI + Python | API server & RAG engine | 8080 |
| **Qdrant** | Qdrant Vector DB | Vector storage & search | 6333 |

## 📋 Prerequisites

- **Docker Desktop** 4.0+ with Docker Compose V2
- **4GB+ RAM** available for containers
- **10GB+ disk space** for images and data
- **API Keys**:
  - OpenAI API key (required)
  - Google API key (optional, for synthesis)

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4GB | 8GB+ |
| Storage | 10GB | 20GB+ |
| Network | Broadband | Broadband |

## 🛠️ Installation

### Automated Setup

The easiest way to get started:

```bash
./scripts/docker-setup.sh
```

This script will:
- ✅ Check Docker installation
- ✅ Create necessary directories
- ✅ Set up environment files
- ✅ Build Docker images
- ✅ Validate configuration

### Manual Setup

If you prefer manual setup:

```bash
# 1. Create directories
mkdir -p data/{qdrant,documents,manifests,logs}
mkdir -p documents config logs

# 2. Setup environment
cp .env.docker.example .env

# 3. Configure API keys
echo "your-openai-api-key" > secrets/openai_api_key.txt
echo "your-google-api-key" > secrets/google_api_key.txt

# 4. Build images
docker compose build

# 5. Start services
docker compose up -d
```

## ⚙️ Configuration

### Environment Variables

Edit the `.env` file to customize your deployment:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Database
QDRANT_URL=http://qdrant:6333
COLLECTION_NAME=knowledge_base

# Document Processing
CHUNKING_STRATEGY=hybrid_hierarchical_semantic
CHUNK_SIZE_TOKENS=512
CHUNK_OVERLAP_TOKENS=100

# Search Configuration
TOP_K_DENSE=10
TOP_K_SPARSE=10
TOP_K_RERANK=5
```

### API Keys Setup

RAGEngine uses Docker secrets for secure API key management:

```bash
# OpenAI API Key (Required)
echo "sk-your-openai-key" > secrets/openai_api_key.txt

# Google API Key (Optional - for knowledge synthesis)
echo "your-google-api-key" > secrets/google_api_key.txt
```

### Volume Configuration

Data is persisted using Docker volumes:

| Volume | Purpose | Host Path |
|--------|---------|-----------|
| `qdrant_data` | Vector database storage | `./data/qdrant` |
| `documents_data` | Uploaded documents | `./data/documents` |
| `manifests_data` | Processing manifests | `./data/manifests` |
| `logs_data` | Application logs | `./data/logs` |

## 🎯 Usage

### Starting the Application

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f

# Check status
docker compose ps
```

### Stopping the Application

```bash
# Stop all services
docker compose down

# Stop and remove volumes (⚠️ deletes data)
docker compose down -v
```

### Accessing Services

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend** | http://localhost:3000 | Main web interface |
| **Backend API** | http://localhost:8080 | REST API endpoints |
| **API Docs** | http://localhost:8080/docs | Interactive API documentation |
| **Qdrant** | http://localhost:6333 | Vector database interface |

### Common Operations

```bash
# View real-time logs
docker compose logs -f backend

# Restart a specific service
docker compose restart backend

# Execute commands in containers
docker compose exec backend python -c "print('Hello from backend')"

# Update images
docker compose pull
docker compose up -d
```

## 🔧 Development

For development with hot reloading:

```bash
# Start in development mode
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# This enables:
# - Hot reloading for backend and frontend
# - Source code mounting
# - Development-optimized settings
```

### Development Features

- **Hot Reloading**: Changes to source code automatically restart services
- **Debug Logging**: Enhanced logging for troubleshooting
- **Source Mounting**: Local code changes reflected immediately
- **Development Ports**: Frontend runs on port 5173 (Vite dev server)

### Making Changes

1. **Backend Changes**: Edit files in `backend/` - uvicorn will auto-reload
2. **Frontend Changes**: Edit files in `RAGEngine-Frontend/` - Vite will hot-reload
3. **Configuration Changes**: Restart containers with `docker compose restart`

## 🚀 Production Deployment

For production environments:

```bash
# Start with production optimizations
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Production Features

- **Resource Limits**: CPU and memory constraints
- **Security Hardening**: Read-only containers, no-new-privileges
- **Logging**: Structured JSON logging with rotation
- **Health Checks**: Enhanced monitoring and auto-recovery
- **Scaling**: Multiple replicas for high availability

### Production Checklist

- [ ] Configure proper API keys
- [ ] Set up SSL/TLS termination (reverse proxy)
- [ ] Configure backup strategy for volumes
- [ ] Set up monitoring and alerting
- [ ] Review security settings
- [ ] Configure log aggregation

## 🔍 Troubleshooting

### Common Issues

#### 1. Container Won't Start

```bash
# Check logs
docker compose logs backend

# Common causes:
# - Missing API keys
# - Port conflicts
# - Insufficient resources
```

#### 2. API Connection Issues

```bash
# Check network connectivity
docker compose exec frontend curl http://backend:8080/system-info

# Verify CORS settings in backend
```

#### 3. Qdrant Connection Issues

```bash
# Check Qdrant health
docker compose exec backend curl http://qdrant:6333/healthz

# Restart Qdrant
docker compose restart qdrant
```

#### 4. Frontend Build Issues

```bash
# Rebuild frontend
docker compose build frontend --no-cache

# Check Node.js version compatibility
```

### Debug Commands

```bash
# View container resource usage
docker stats

# Inspect container configuration
docker compose config

# Check container health
docker compose ps

# Access container shell
docker compose exec backend sh
```

### Performance Issues

```bash
# Monitor resource usage
docker stats

# Check disk space
df -h

# View memory usage
docker system df
```

## 🔧 Advanced Configuration

### Custom Qdrant Configuration

Edit `docker/qdrant/config.yaml`:

```yaml
service:
  max_request_size_mb: 64
  max_workers: 8

storage:
  performance:
    max_search_threads: 8
```

### Backend Scaling

```yaml
# In docker-compose.prod.yml
backend:
  deploy:
    replicas: 3
  environment:
    - WORKERS=4
```

### Custom Networks

```yaml
networks:
  ragengine-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### SSL/TLS Setup

For production, use a reverse proxy like Nginx:

```yaml
# docker-compose.prod.yml
nginx:
  image: nginx:alpine
  ports:
    - "443:443"
    - "80:80"
  volumes:
    - ./nginx.conf:/etc/nginx/nginx.conf
    - ./ssl:/etc/ssl
```

## 🧹 Cleanup

To completely remove RAGEngine:

```bash
# Automated cleanup
./scripts/docker-cleanup.sh

# Manual cleanup
docker compose down -v
docker rmi $(docker images | grep ragengine | awk '{print $3}')
docker volume prune -f
```

## 📚 Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SvelteKit Documentation](https://kit.svelte.dev/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)

## 🆘 Support

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review container logs: `docker compose logs`
3. Verify your configuration matches the examples
4. Open an issue with detailed error messages and logs

---

**Happy containerizing! 🐳**