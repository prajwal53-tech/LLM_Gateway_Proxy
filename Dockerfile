# ==============================================================================
# Stage 1: Builder
# ==============================================================================
FROM python:3.12-slim AS builder

WORKDIR /build

# Install system dependencies needed for compiling native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy dependency specifications first for optimal Docker layer caching
COPY pyproject.toml /build/

# Install python package dependencies into virtualenv
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir .

# ==============================================================================
# Stage 2: Production Runtime
# ==============================================================================
FROM python:3.12-slim AS runtime

# Runtime labels
LABEL org.opencontainers.image.title="llm-eval-proxy" \
      org.opencontainers.image.description="High-Throughput LLM Gateway & Evaluation Proxy" \
      org.opencontainers.image.version="0.1.0"

# Python environment hardening
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000 \
    APP_NAME="llm-eval-proxy"

WORKDIR /app

# Install minimal runtime libraries (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged system user for secure container execution
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /sbin/nologin -M -d /app appuser

# Copy virtual environment containing all dependencies from builder stage
COPY --from=builder --chown=appuser:appgroup /opt/venv /opt/venv

# Copy application source code
COPY --chown=appuser:appgroup pyproject.toml /app/
COPY --chown=appuser:appgroup app/ /app/app/
COPY --chown=appuser:appgroup scripts/ /app/scripts/

# Install package in editable/local mode inside virtualenv
RUN pip install --no-cache-dir --no-deps -e /app

# Switch to non-root user
USER appuser

# Expose HTTP gateway port
EXPOSE 8000

# Health check configuration
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI application with Uvicorn ASGI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
