# =============================================================================
# RAGEngine Backend Dockerfile
# Multi-stage build for optimized production image
# =============================================================================

# =============================================================================
# Stage 1: Build Dependencies
# =============================================================================
FROM python:3.11-alpine AS builder

# Install build dependencies
RUN apk add --no-cache \
    gcc \
    g++ \
    musl-dev \
    linux-headers \
    libffi-dev \
    openssl-dev \
    cargo \
    rust \
    freetype-dev \
    jpeg-dev \
    openjpeg-dev \
    zlib-dev \
    libjpeg-turbo-dev \
    tesseract-ocr-dev \
    mupdf-dev

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# =============================================================================
# Stage 2: Runtime Image
# =============================================================================
FROM python:3.11-alpine AS runtime

# Install runtime dependencies only
RUN apk add --no-cache \
    tesseract-ocr \
    mupdf \
    freetype \
    jpeg \
    openjpeg \
    zlib \
    libjpeg-turbo \
    curl \
    && rm -rf /var/cache/apk/*

# Create non-root user
RUN addgroup -g 1000 ragengine && \
    adduser -D -s /bin/sh -u 1000 -G ragengine ragengine

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy NLTK data from builder
COPY --from=builder /root/nltk_data /home/ragengine/nltk_data
ENV NLTK_DATA=/home/ragengine/nltk_data

# Set working directory
WORKDIR /app

# Copy application code
COPY backend/ /app/
COPY run_web.py /app/

# Copy entrypoint script
COPY docker/backend/entrypoint.sh /docker/backend/entrypoint.sh
RUN chmod +x /docker/backend/entrypoint.sh

# Create necessary directories
RUN mkdir -p /app/documents /app/config /app/logs && \
    chown -R ragengine:ragengine /app /docker

# Switch to non-root user
USER ragengine

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/system-info || exit 1

# Default command
CMD ["python", "run_web.py"]