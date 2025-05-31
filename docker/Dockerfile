# docker/Dockerfile
# Stage 1: Build dependencies (for non-CUDA packages)
FROM python:3.10-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libhdf5-dev \
    libopenblas-dev \
    liblapack-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python packages (except CuPy)
COPY docker/requirements-no-cuda.txt /tmp/requirements.txt
RUN pip install --user --no-cache-dir -r /tmp/requirements.txt

# Stage 2: Runtime image with CUDA
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install runtime dependencies including pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-distutils \
    libhdf5-103 \
    libopenblas0 \
    libgfortran5 \
    libgomp1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links
RUN ln -sf /usr/bin/python3.10 /usr/bin/python
RUN ln -sf /usr/bin/pip3 /usr/bin/pip

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Update PATH and PYTHONPATH
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/root/.local/lib/python3.10/site-packages:$PYTHONPATH

# Install CuPy in the runtime stage where CUDA is available
RUN pip install --no-cache-dir cupy-cuda11x>=11.0.0

# Create working directory
WORKDIR /workspace
RUN mkdir -p /workspace/src /workspace/data /workspace/output

# Copy source code
COPY src/ /workspace/src/
ENV PYTHONPATH=/workspace/src:$PYTHONPATH

# Setup HDF5 plugin if braingeneers is installed
RUN mkdir -p /usr/local/hdf5/lib/plugin && \
    python3 -c "import sys; exec('''try:\n    import braingeneers, os, shutil\n    plugin_path = os.path.join(os.path.dirname(braingeneers.__file__), \"data\", \"mxw_h5_plugin\", \"Linux\", \"libcompression.so\")\n    if os.path.exists(plugin_path):\n        shutil.copy(plugin_path, \"/usr/local/hdf5/lib/plugin/\")\n        print(\"HDF5 plugin setup completed\")\n    else:\n        print(\"HDF5 plugin file not found\")\nexcept Exception as e:\n    print(f\"HDF5 plugin setup skipped: {e}\")\n    sys.exit(0)\n''')"

ENV HDF5_PLUGIN_PATH=/usr/local/hdf5/lib/plugin

CMD ["python", "/workspace/src/runner.py"]