# SupoClip v2.0 — RunPod Backend Serverless Container
# Official RunPod Serverless Base with CUDA 12.2

FROM runpod/base:0.6.2-cuda12.2.0

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
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

# ── Python dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir -r requirements.txt

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
ENV PYTHONPATH="/app"
ENV TORCH_HOME=/workspace/models
ENV HF_HOME=/workspace/models

# ── Serverless Entrypoint (Reset base entrypoint) ────────────────────────────
ENTRYPOINT []
CMD ["python3", "-u", "/app/handler.py"]
