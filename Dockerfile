FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set timezone to avoid interactive prompt
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies with uv
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Copy only necessary files
COPY src/ ./src/
COPY job_config.yaml deploy_vertex_ai.py ./

# Set environment variables for GCS
ENV PYTHONPATH=/app

# Run training
CMD ["python", "-m", "src.v2.train"]