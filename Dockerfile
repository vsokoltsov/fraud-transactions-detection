# Use Python 3.12 slim image as base
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
# Set working directory
WORKDIR /app
# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

RUN pip install --no-cache-dir uv
ENV PATH="/home/app/.local/bin:${PATH}" 

COPY pyproject.toml uv.lock ./

FROM base AS api

# Install dependencies using uv
RUN uv sync --group api --group dev --frozen

# Copy application code
COPY api/ ./api/
COPY data/ ./data/

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

FROM api AS test

RUN uv sync --group api --group dev --frozen

COPY api/ ./api/
COPY tests/ ./tests/

CMD ["bash"]

FROM base AS ml

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz build-essential && rm -rf /var/lib/apt/lists/*
COPY src/fraudlib ./src/fraudlib
RUN chown -R app:app ./src/fraudlib

USER app
ARG NB_UID=1000
RUN uv sync --group ml --no-dev --no-group dev --no-group ploomber
RUN uv pip install --no-cache-dir -e src/fraudlib


# Install Jupyter Lab extensions to fix compatibility issues
RUN uv run pip install --no-cache-dir \
    jupyterlab-server>=2.25.0 \
    jupyter-server>=2.16.0 \
    jupyterlab-lsp \
    'python-lsp-server[all]' \
    'jedi>=0.18.1' \
    notebook>=7.0.0

# Install kernel with user permissions
RUN uv run python -m ipykernel install --user --name ml-tech-assignment \
    --display-name "Machine Learning Tech Assignment"

# Set proper Jupyter data directory
ENV JUPYTER_DATA_DIR=/home/app/.local/share/jupyter
ENV JUPYTER_ENABLE_LAB=yes 
ENV PYTHONUNBUFFERED=1
ENV PLOMBER_DISABLE_JUPYTER=1

EXPOSE 8888