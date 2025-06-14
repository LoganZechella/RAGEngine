# Multi-stage build for optimized production image
FROM python:3.13-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    tesseract-ocr \
    tesseract-ocr-eng \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY templates/ ./templates/
COPY static/ ./static/
COPY *.py ./
COPY .env.docker.example ./.env

# Copy and set up entrypoint script
COPY docker/web-app/entrypoint.sh /docker/web-app/entrypoint.sh
RUN chmod +x /docker/web-app/entrypoint.sh

# Create necessary directories
RUN mkdir -p documents config logs /docker/web-app

# Create upload directory within documents volume
RUN mkdir -p documents/uploads && \
    chmod 755 documents && \
    chmod 755 documents/uploads

# Set Python path
ENV PYTHONPATH=/app/backend

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# Default command
CMD ["python", "run_web.py"] 