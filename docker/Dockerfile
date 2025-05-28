# docker/Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install system dependencies (REMOVED ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    libhdf5-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    git \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Create working directory
WORKDIR /workspace
RUN mkdir -p /workspace/src /workspace/data /workspace/output

# Copy requirements file
COPY docker/requirements.txt /workspace/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Install braingeneers and setup HDF5 plugin
RUN pip install --no-cache-dir braingeneers && \
    mkdir -p /usr/local/hdf5/lib/plugin && \
    python -c "import braingeneers; import os; plugin_path = os.path.join(os.path.dirname(braingeneers.__file__), 'data', 'mxw_h5_plugin', 'Linux', 'libcompression.so'); os.system(f'cp {plugin_path} /usr/local/hdf5/lib/plugin/')" || echo "HDF5 plugin setup skipped"

# Set HDF5 plugin path
ENV HDF5_PLUGIN_PATH=/usr/local/hdf5/lib/plugin

# Copy source code
COPY src/ /workspace/src/

# Set Python path
ENV PYTHONPATH=/workspace/src:$PYTHONPATH

# Default command
CMD ["python", "/workspace/src/runner.py"]