# Dockerfile for VNF Ensemble Forecasting
# Usage: docker run -v ./data:/app/data your-image

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/
COPY models/ ./models/  # Optional: pre-trained models

# Create directories for outputs
RUN mkdir -p plots results

# Set environment variables
ENV PYTHONPATH=/app
ENV TF_CPP_MIN_LOG_LEVEL=2

# Default command: run evaluation
CMD ["python", "src/evaluate.py"]