# Multi-stage build for smaller image size
FROM python:3.9-slim AS builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU version first (with retries and timeout)
RUN pip install --default-timeout=1000 --no-cache-dir --user \
    --retries 5 \
    torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Copy and install other requirements
COPY requirements-docker.txt .
RUN pip install --default-timeout=1000 --no-cache-dir --user \
    --retries 5 \
    -r requirements-docker.txt

# Final stage
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY setup.py .

# Install the package (lightweight, no heavy dependencies)
RUN pip install --no-cache-dir -e . --no-deps

# Make sure scripts are in PATH
ENV PATH=/root/.local/bin:$PATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables
ENV MODEL_PATH=models/checkpoints/best_model.pth \
    MODEL_NAME=improved \
    PORT=8000 \
    PYTHONUNBUFFERED=1

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
