# syntax=docker/dockerfile:1

# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Create OCI directory and copy credentials from the runner's home directory
# This requires the build to be run with Docker BuildKit (buildx).
# It mounts the OCI config and key from the host runner, then copies them
# into the image's filesystem.
RUN --mount=type=bind,source=/home/runner/.oci/config,target=/tmp/config \
    --mount=type=bind,source=/home/runner/.oci/key.pem,target=/tmp/key.pem \
    mkdir -p /root/.oci && \
    cp /tmp/config /root/.oci/config && \
    cp /tmp/key.pem /root/.oci/key.pem && \
    chmod 600 /root/.oci/config /root/.oci/key.pem

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"] 