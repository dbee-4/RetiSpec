# ==============================================================================
# RetiSpec - Docker Container Configuration (CPU-Optimized)
# ==============================================================================
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

WORKDIR /app

# Install system dependencies required for OpenCV and image operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Pre-install lightweight CPU PyTorch wheels (avoids downloading 2GB+ CUDA wheels)
RUN pip install --no-cache-dir torch==2.2.2+cpu torchvision==0.17.2+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# Install project requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY config.py model.py tta.py preprocessing.py ./
COPY xai_gradcam.py xai_gradientxinput.py export_onnx.py app.py ./
COPY templates/ ./templates/
COPY static/ ./static/
COPY models/ ./models/

# Healthcheck to verify FastAPI server responsiveness
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-7860}/health || curl -f http://localhost:8000/health || exit 1

ENV PORT=7860
EXPOSE 7860 8000

# Run FastAPI via Uvicorn (reads $PORT dynamically)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]

