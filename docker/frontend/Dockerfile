# =============================================================================
# RAGEngine Frontend Dockerfile
# Multi-stage build for optimized SvelteKit production image
# =============================================================================

# =============================================================================
# Stage 1: Build Stage
# =============================================================================
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files
COPY RAGEngine-Frontend/package*.json ./

# Install dependencies
RUN npm ci --only=production --silent

# Copy source code
COPY RAGEngine-Frontend/ ./

# Set build-time environment variables for Docker deployment
ENV VITE_API_BASE_URL=http://localhost:8080

# Build the application
RUN npm run build

# =============================================================================
# Stage 2: Runtime Stage
# =============================================================================
FROM node:18-alpine AS runtime

# Install curl for health checks
RUN apk add --no-cache curl

# Create non-root user with available IDs
RUN addgroup -g 1001 ragengine && \
    adduser -D -s /bin/sh -u 1001 -G ragengine ragengine

# Set working directory
WORKDIR /app

# Copy built application from builder stage
COPY --from=builder /app/build ./build
COPY --from=builder /app/package*.json ./

# Copy entrypoint script
COPY docker/frontend/entrypoint.sh /docker/frontend/entrypoint.sh
RUN chmod +x /docker/frontend/entrypoint.sh

# Install only production dependencies
RUN npm ci --only=production --silent && \
    npm cache clean --force

# Create docker directory and change ownership to non-root user
RUN mkdir -p /docker && \
    chown -R ragengine:ragengine /app /docker

# Switch to non-root user
USER ragengine

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:3000 || exit 1

# Start the application
CMD ["node", "build"]