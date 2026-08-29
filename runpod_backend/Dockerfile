# SupoClip v2.0 — RunPod Backend Container
# Base: RunPod PyTorch 2.1.0 + Python 3.11 + CUDA 12.1

FROM runpod/pytorch:2.1.0-py3.11-cuda12.1.0-devel-ubuntu22.04

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    fonts-dejavu \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Set working directory ────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (install before copying code for layer caching) ──────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── App code ─────────────────────────────────────────────────────────────────
COPY . /app

# ── Create asset directories ─────────────────────────────────────────────────
RUN mkdir -p /workspace/assets/sfx \
             /workspace/assets/music \
             /workspace/assets/fonts \
             /workspace/models \
             /workspace/temp

# ── Copy bundled assets ───────────────────────────────────────────────────────
COPY assets/sfx/ /workspace/assets/sfx/
COPY assets/music/ /workspace/assets/music/
COPY assets/fonts/ /workspace/assets/fonts/

# ── Download Montserrat font if not bundled ───────────────────────────────────
RUN wget -q -O /workspace/assets/fonts/Montserrat-Bold.ttf \
    "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Bold.ttf" \
    || echo "Font download failed — will use fallback"

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TORCH_HOME=/workspace/models
ENV HF_HOME=/workspace/models

# Expose for local testing only (RunPod serverless doesn't need this)
EXPOSE 8000

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import runpod; import moviepy; import faster_whisper; print('OK')" || exit 1

# ── Start RunPod serverless worker ───────────────────────────────────────────
CMD ["python", "-u", "handler.py"]
