# SupoClip v2.0 — RunPod Serverless Backend
#
# Base: runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
# This image is PRE-CACHED on RunPod GPU nodes — no multi-GB cold pull.
# PyTorch + CUDA + Python are already included.

FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    fonts-dejavu \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ── Set working directory ────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── App code ─────────────────────────────────────────────────────────────────
COPY . /app

# ── Asset directories ─────────────────────────────────────────────────────────
RUN mkdir -p /workspace/assets/sfx \
             /workspace/assets/music \
             /workspace/assets/fonts \
             /workspace/models \
             /workspace/temp

# ── Copy bundled assets ───────────────────────────────────────────────────────
COPY assets/sfx/   /workspace/assets/sfx/
COPY assets/music/ /workspace/assets/music/
COPY assets/fonts/ /workspace/assets/fonts/

# ── Download Montserrat font ──────────────────────────────────────────────────
RUN wget -q -O /workspace/assets/fonts/Montserrat-Bold.ttf \
    "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Bold.ttf" \
    || echo "Font download failed — will use system fallback"

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH="/app"
ENV TORCH_HOME=/workspace/models
ENV HF_HOME=/workspace/models

# ── Serverless Entrypoint ────────────────────────────────────────────────────
# Use direct python3 entrypoint — no shell wrapper, no healthcheck conflicts
ENTRYPOINT ["python3", "-u", "/app/handler.py"]
