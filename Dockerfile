FROM python:3.10-slim

WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/

# Copy trained models
COPY models/ ./models/

# Set environment
ENV PYTHONPATH=/app
ENV TF_CPP_MIN_LOG_LEVEL=2

# Default command: run evaluation
CMD ["python", "src/evaluate.py"]