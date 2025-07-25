FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set timezone to avoid interactive prompt
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libx264-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY src/ ./src/
COPY job_config.yaml deploy_vertex_ai.py ./

# Set environment variables for GCS
ENV PYTHONPATH=/app

# Run training
CMD ["python", "-m", "src.v2.train"]