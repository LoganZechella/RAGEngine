# RAGEngine Distribution Guide 🚀

Quick setup guide for running RAGEngine Breath Diagnostics on any machine.

## Prerequisites

- Docker and Docker Compose installed
- OpenAI API key (required)
- Google API key (optional)

## Quick Start

### 1. Download Files

Download these 3 files to your target machine:
- `docker-compose.distribution.yml`
- `env.distribution`
- `README-DISTRIBUTION.md` (this file)

### 2. Setup Environment

```bash
# Copy environment template
cp env.distribution .env

# Edit .env and add your API keys
nano .env  # or use any text editor
```

**Required**: Set your `OPENAI_API_KEY` in the `.env` file.

### 3. Start the Application

```bash
# Pull latest images and start
docker-compose -f docker-compose.distribution.yml up -d

# Check status
docker-compose -f docker-compose.distribution.yml ps

# View logs
docker-compose -f docker-compose.distribution.yml logs -f web-app
```

### 4. Access the Application

Open your browser and go to: **http://localhost:8080**

## What You Get

- 🔍 **Multi-Collection Search**: Search across different document types
- 📄 **Document Upload**: Upload PDFs, Word docs, text files
- 🤖 **Auto-Classification**: Documents automatically sorted into collections
- 📊 **Collection Management**: View stats and manage document collections
- 🎯 **Breath Diagnostics Focus**: Specialized for medical/diagnostic content

## Collections Available

- **Current Documents**: Recent uploads and active documents
- **Legacy Documents**: Historical documents and archives
- **SOP & Policy**: Standard operating procedures and policies
- **Research Data**: Research papers and studies
- **Clinical Studies**: Clinical trial data and medical studies

## Troubleshooting

### Application won't start
```bash
# Check if containers are running
docker ps

# Check logs for errors
docker-compose -f docker-compose.distribution.yml logs
```

### Can't access at localhost:8080
```bash
# Check if port 8080 is available
docker-compose -f docker-compose.distribution.yml ps

# Try different port (edit docker-compose.distribution.yml)
# Change "8080:8080" to "8081:8080" and restart
```

### File upload issues (Fixed in latest version)
```bash
# Check upload directory exists and has permissions
docker exec breath-diagnostics-web ls -la /app/documents/uploads

# Check logs for upload errors
docker-compose -f docker-compose.distribution.yml logs web-app | grep -i upload

# Restart if upload directory missing
docker-compose -f docker-compose.distribution.yml restart web-app
```

### Document processing fails
```bash
# Verify API keys are set
docker exec breath-diagnostics-web printenv | grep API_KEY

# Check document processing logs
docker-compose -f docker-compose.distribution.yml logs web-app | grep -i "processing\|ingest"

# Test with a small PDF file first
```

### Vector database issues
```bash
# Restart with fresh database
docker-compose -f docker-compose.distribution.yml down -v
docker-compose -f docker-compose.distribution.yml up -d
```

## Commands Reference

```bash
# Start services
docker-compose -f docker-compose.distribution.yml up -d

# Stop services
docker-compose -f docker-compose.distribution.yml down

# Update to latest version
docker-compose -f docker-compose.distribution.yml pull
docker-compose -f docker-compose.distribution.yml up -d

# View logs
docker-compose -f docker-compose.distribution.yml logs -f

# Reset everything (WARNING: deletes all data)
docker-compose -f docker-compose.distribution.yml down -v
```

## Image Details

- **Main Application**: `loganzechella/ragengine-web-app:latest` (~1.9GB)
- **Vector Database**: `qdrant/qdrant:latest` (~293MB)

Total download: ~2.2GB on first run

## Performance Notes

- Initial startup takes 1-2 minutes
- Document processing is faster after first few uploads
- Search performance improves as the knowledge base grows
- Requires ~4GB RAM for optimal performance

## Support

For issues or questions, check the logs first:
```bash
docker-compose -f docker-compose.distribution.yml logs web-app
```

Common log locations inside containers:
- Application logs: `/app/logs/`
- System logs: Use `docker logs` command 