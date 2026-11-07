# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Prevent Python from writing .pyc and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (curl for healthchecks, gcc/libpq-dev for postgres support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Sovra application code
COPY core/ ./core/
COPY scripts/ ./scripts/
COPY pyproject.toml .

# Expose Gateway (8000) and Control Plane (8001) ports
EXPOSE 8000 8001

# Default command (overridden in docker-compose.yml)
CMD ["uvicorn", "core.control_plane.app:app", "--host", "0.0.0.0", "--port", "8001"]

