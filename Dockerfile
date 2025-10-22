# Use Python 3.12 slim image as base
FROM python:3.12-slim as base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
# Set working directory
WORKDIR /app
# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && rm -rf /var/lib/apt/lists/*

RUN pip install uv    

COPY pyproject.toml uv.lock ./

FROM base AS api

# Install dependencies using uv
RUN uv sync --group api --group dev --frozen

# Copy application code
COPY api/ ./api/
COPY data/ ./data/

# Create a non-root user
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# Run the application
CMD ["uv", "run", "python", "-m", "uvicorn", "api.app:api", "--host", "0.0.0.0", "--port", "8000"]

FROM api as test

RUN uv sync --group api --group dev --frozen

COPY api/ ./api/
COPY tests/ ./tests/

CMD ["bash"]

