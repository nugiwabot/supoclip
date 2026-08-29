# SupoClip v2.0 🎬

> **Open-source alternative to CapCut Pro / Opus Clip**  
> Mass AI Video Clipper & Auto-Editor — 100% Python 3.11+

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)](https://streamlit.io)
[![RunPod](https://img.shields.io/badge/Compute-RunPod-purple)](https://runpod.io)

---

## Architecture

```
User Browser ──► Streamlit UI (VPS) ──► S3 Storage ◄──► RunPod Worker
                      │                                      │
                      └──► LLM API (OpenAI/Gemini/DeepSeek) │
                                                              │
                                         Whisper + MoviePy + FFmpeg
```

## Features

| Feature | Details |
|---|---|
| 🎯 **AI Viral Detection** | OpenAI-compatible — works with GPT-4o, Gemini Flash, DeepSeek-R1, Claude |
| 📐 **9:16 Reframing** | MediaPipe face-tracked intelligent crop from 16:9 widescreen |
| 🎵 **Audio Ducking** | Librosa vocal energy detection, auto-ducks music -25% during speech |
| 💥 **SFX Injection** | Whoosh/ding/sub-bass precisely at AI-flagged emotional timestamps |
| 📝 **Kinetic Captions** | Word-by-word Whisper-synced, neon impact words 1.5x scaled |
| 🔄 **Fault Tolerance** | Per-video try/except isolation — one failure never stops the queue |
| 🗄️ **Task Queue** | SQLite-backed real-time status monitoring with 7 distinct states |

## Quick Start

### 1. Frontend (VPS or Local PC)

```bash
cd vps_frontend
pip install -r requirements.txt
streamlit run app.py
```

### 2. Backend (RunPod Serverless) -- LIVE & CONFIGURED ✅

- **GitHub Repo:** [https://github.com/nugiwabot/supoclip](https://github.com/nugiwabot/supoclip)
- **Container Image:** `ghcr.io/nugiwabot/supoclip-backend:latest`
- **RunPod Endpoint ID:** `zdxos18f8mlof6`
- **GPU Pools:** `AMPERE_24`, `ADA_24` (RTX 3090, RTX 4090, A5000)
- **Scaling:** Auto-scale 0 to 2 workers (Zero idle cost)

### 3. Optional S3 Environment Variables (Only if you want custom S3/R2)

| Variable | Description |
|---|---|
| `SUPOCLIP_S3_ENDPOINT` | *(Optional)* Cloudflare R2 / AWS S3 endpoint URL |
| `SUPOCLIP_S3_ACCESS_KEY` | *(Optional)* S3 access key ID |
| `SUPOCLIP_S3_SECRET_KEY` | *(Optional)* S3 secret access key |
| `SUPOCLIP_S3_BUCKET` | *(Optional)* Bucket name for video storage |

## Asset Setup

Add your assets to `runpod_backend/assets/`:

```
assets/
├── sfx/
│   ├── whoosh.mp3      # Transition swoosh
│   ├── ding.mp3        # Achievement bell
│   └── sub_bass.mp3    # Deep bass boom for revelations
├── music/
│   └── *.mp3           # Chill lo-fi instrumentals (any number)
└── fonts/
    └── Montserrat-Bold.ttf  # Auto-downloaded on Docker build
```

> **Free SFX Sources**: [Freesound.org](https://freesound.org) (CC0 license)  
> **Free Music**: [Free Music Archive](https://freemusicarchive.org) (CC0/CC-BY)

## LLM Compatibility

Any OpenAI-compatible endpoint works. Change the Base URL in the sidebar:

| Provider | Base URL |
|---|---|
| OpenAI GPT-4o | `https://api.openai.com/v1` |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` |
| DeepSeek | `https://api.deepseek.com/v1` |
| Groq | `https://api.groq.com/openai/v1` |
| Local (Ollama) | `http://localhost:11434/v1` |

## Project Structure

```
supoclip/
├── vps_frontend/
│   ├── app.py                  # Streamlit UI
│   ├── core/
│   │   ├── db.py               # SQLite task queue
│   │   ├── ai_brain.py         # Viral retention LLM engine
│   │   ├── runpod_client.py    # RunPod API client
│   │   └── storage.py          # S3-compatible storage
│   ├── requirements.txt
│   └── .streamlit/config.toml
│
└── runpod_backend/
    ├── handler.py              # RunPod serverless entry
    ├── core/
    │   ├── transcriber.py      # faster-whisper GPU transcription
    │   ├── face_tracker.py     # MediaPipe smart crop
    │   ├── video_editor.py     # 9:16 reframe + cuts
    │   ├── audio_mixer.py      # Ducking + SFX + crossfade
    │   └── caption_renderer.py # Kinetic CapCut-style captions
    ├── assets/                 # SFX, music, fonts
    ├── requirements.txt
    └── Dockerfile
```

## License

MIT License — Free for commercial and personal use.
