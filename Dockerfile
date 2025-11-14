# Dockerfile for Heart Segmentation project
# Based on install.md instructions

FROM python:3.9-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements-segmentation.txt /workspace/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-segmentation.txt

# Install additional testing dependencies
RUN pip install --no-cache-dir pytest

# Copy source code
COPY src/ /workspace/src/
COPY tests/ /workspace/tests/

# Create necessary directories
RUN mkdir -p /workspace/data /workspace/annotations /workspace/models /workspace/logs

# Set Python path
ENV PYTHONPATH=/workspace

# Default command
CMD ["python", "src/convert.py", "--help"]