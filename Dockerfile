FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application
COPY rag/ ./rag/
COPY .env.example ./.env

# Data directory (mounted at runtime)
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["python", "-m", "rag.server.main"]
