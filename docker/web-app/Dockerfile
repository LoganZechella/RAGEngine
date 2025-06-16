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

# Pre-download ML models to avoid runtime download timeouts
# Set environment variables for model caching to match where sentence-transformers looks
ENV SENTENCE_TRANSFORMERS_HOME=/root/.cache/sentence_transformers
ENV TRANSFORMERS_CACHE=/root/.cache/transformers
ENV HF_HOME=/root/.cache/huggingface

# Create cache directories
RUN mkdir -p /root/.cache/sentence_transformers /root/.cache/transformers /root/.cache/huggingface

# Copy model preloading script and run it to cache models during build
COPY docker/web-app/simple_preload.py /tmp/simple_preload.py
RUN python /tmp/simple_preload.py && rm /tmp/simple_preload.py

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