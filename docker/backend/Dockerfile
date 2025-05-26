# =============================================================================
# RAGEngine Backend Dockerfile
# Multi-stage build for optimized production image
# =============================================================================

# =============================================================================
# Stage 1: Build Dependencies
# =============================================================================
FROM python:3.12 AS builder

# Install build dependencies for Debian-based image
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc6-dev \
    linux-libc-dev \
    libffi-dev \
    libssl-dev \
    cargo \
    rustc \
    libfreetype6-dev \
    libjpeg-dev \
    libopenjp2-7-dev \
    zlib1g-dev \
    libjpeg62-turbo-dev \
    tesseract-ocr \
    libtesseract-dev \
    libmupdf-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"

# =============================================================================
# Stage 2: Runtime Image
# =============================================================================
FROM python:3.12 AS runtime

# Install runtime dependencies only for Debian-based image
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    mupdf \
    libfreetype6 \
    libjpeg62-turbo \
    libopenjp2-7 \
    zlib1g \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -g 1000 ragengine && \
    useradd -r -u 1000 -g ragengine -s /bin/bash ragengine

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy NLTK data from builder and set proper ownership
COPY --from=builder /root/nltk_data /home/ragengine/nltk_data
RUN chown -R ragengine:ragengine /home/ragengine/nltk_data
ENV NLTK_DATA=/home/ragengine/nltk_data

# Set working directory
WORKDIR /app

# Copy application code
COPY backend/ /app/
COPY run_web.py /app/

# Copy static files and templates
COPY static/ /app/static/
COPY templates/ /app/templates/

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