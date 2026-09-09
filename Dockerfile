FROM python:3.12-slim

# Install system deps for spaCy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source code and config
COPY pyproject.toml README.md ./
COPY src/ src/

# Install CPU-only PyTorch first, then everything else via pip
# This avoids the 3.5GB NVIDIA CUDA libraries entirely
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    "transformers>=4.36.0" \
    "spacy>=3.8.0" \
    "scikit-learn>=1.3.0" \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.27.0" \
    "numpy" && \
    pip install --no-cache-dir en-core-web-sm \
    -f https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0.tar.gz && \
    pip install --no-cache-dir -e .

EXPOSE 8000

ENV PYTHONPATH=src
CMD ["sh", "-c", "MES_PORT=${PORT:-8000} python -m mental_entropy.api"]
