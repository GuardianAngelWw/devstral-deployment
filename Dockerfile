FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY server.py .
COPY inference.py .

# Create directory for model files
RUN mkdir -p /root/mistral_models/Devstral

# Set environment variables
ENV MODEL_ID=mistralai/Devstral-Small-2505
ENV MODEL_DIR=/root/mistral_models/Devstral

# Set up a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app /root/mistral_models

# Expose the server port
EXPOSE 8000

# Switch to non-root user
USER appuser

# Download model at build time (optional, comment out to download at runtime)
# RUN python -c "from huggingface_hub import snapshot_download; from pathlib import Path; snapshot_download(repo_id='mistralai/Devstral-Small-2505', allow_patterns=['params.json', 'consolidated.safetensors', 'tekken.json'], local_dir=Path('/root/mistral_models/Devstral'))"

# Start the server
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8000"]