# GridPulse 2.0 Production Container
FROM python:3.13-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Create non-root user
RUN useradd --create-home --shell /bin/bash gridpulse

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging metadata and package source
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install package with serving and dashboard dependencies
RUN pip install --no-cache-dir .[serving,dashboard]

# Create runtime directories
RUN mkdir -p /app/data/bronze /app/data/silver /app/data/gold /app/data/models /app/data/reports /app/config

# Set ownership to non-root user
RUN chown -R gridpulse:gridpulse /app

USER gridpulse

EXPOSE 8000 8501

# Default command runs FastAPI serving layer
CMD ["uvicorn", "gridpulse.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
