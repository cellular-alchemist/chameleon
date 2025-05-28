# Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    libhdf5-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    ffmpeg \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python packages
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create working directory
WORKDIR /app

# Copy source code
COPY src/ /app/src/

# Add src to Python path
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Create data and results directories
RUN mkdir -p /app/data /app/results

# Default command
CMD ["python", "/app/src/planar_waves_and_ripples.py"]