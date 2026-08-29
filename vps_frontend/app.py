"""
SupoClip v2.0 — Mass AI Video Clipper & Auto-Editor
Frontend: Streamlit Web UI

Features:
- Drag-and-drop mass video upload
- Real-time task queue monitoring
- Model-agnostic AI clip detection
- RunPod serverless job submission & polling
- Full fault tolerance (per-video try/except isolation)
"""

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path

import streamlit as st

# --- Core modules ---
from core.db import (
    TaskStatus,
    STATUS_EMOJI,
    STATUS_COLOR,
    add_task,
    clear_completed,
    delete_task,
    get_all_tasks,
    get_task,
    init_db,
    log_error,
    update_status,
)
from core.ai_brain import AIBrain
from core.runpod_client import RunPodClient
from core.storage import StorageClient

# ---------------------------------------------------------------------------
# Config & Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("supoclip")

APP_DIR = Path(__file__).parent
CONFIG_FILE = APP_DIR / ".supoclip_config.json"
TEMP_DIR = APP_DIR / "temp_uploads"
TEMP_DIR.mkdir(exist_ok=True)

# Initialize database
init_db()


# ---------------------------------------------------------------------------
# Config Persistence
# ---------------------------------------------------------------------------
def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            pass
    return {}


def save_config(cfg: dict) -> None:
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


# ---------------------------------------------------------------------------
# Processing Pipeline (runs in background thread per video)
# ---------------------------------------------------------------------------
def process_video_task(
    task_id: str,
    local_path: Path,
    filename: str,
    config: dict,
) -> None:
    """
    Full processing pipeline for a single video.
    Fully isolated: errors are caught and logged without affecting other tasks.
    """
    try:
        logger.info(f"[{task_id}] Starting pipeline for: {filename}")

        # ── Step 1: Upload Video (Zero-Config / S3) ──────────────────────
        update_status(task_id, TaskStatus.UPLOADING)
        storage = StorageClient(
            endpoint_url=config.get("s3_endpoint", ""),
            access_key=config.get("s3_access_key", ""),
            secret_key=config.get("s3_secret_key", ""),
            bucket_name=config.get("s3_bucket", ""),
        )
        remote_key = f"inputs/{task_id}_{filename}"
        video_url = storage.upload_video(local_path, remote_key=remote_key)
        update_status(task_id, TaskStatus.UPLOADING, video_url=video_url)
        logger.info(f"[{task_id}] Uploaded to storage: {video_url}")

        # ── Step 2: Transcription (via RunPod or local Whisper API) ──────
        update_status(task_id, TaskStatus.TRANSCRIPTION)
        # Note: Transcription is handled inside the RunPod backend.
        # We pass video_url and RunPod will transcribe + extract clips.
        # For the AI Brain, we request a "transcribe_only" first pass.
        # This simplified flow sends video_url to RunPod in one shot:
        logger.info(f"[{task_id}] Submitting transcription job to RunPod")

        rp_client = RunPodClient(
            api_key=config["runpod_api_key"],
            endpoint_id=config["runpod_endpoint_id"],
        )

        # First pass: transcription only
        transcribe_job_id = rp_client.submit_job({
            "action": "transcribe",
            "video_url": video_url,
            "task_id": task_id,
        })
        transcribe_result = rp_client.poll_until_complete(
            transcribe_job_id,
            poll_interval=8,
            timeout=1800,
        )
        transcript_data = transcribe_result.get("output", {})
        transcript_text = transcript_data.get("transcript", "")
        video_duration = float(transcript_data.get("duration", 300.0))

        # ── Step 3: AI Brain — Viral Clip Analysis ───────────────────────
        update_status(task_id, TaskStatus.AI_CURATING)
        brain = AIBrain(
            base_url=config["llm_base_url"],
            api_key=config["llm_api_key"],
            model=config["llm_model"],
        )
        clips_json = brain.analyze_transcript(transcript_text, video_duration)
        update_status(task_id, TaskStatus.AI_CURATING, ai_json=clips_json)
        logger.info(f"[{task_id}] AI returned {len(clips_json)} clips")

        # ── Step 4: RunPod Rendering ─────────────────────────────────────
        update_status(task_id, TaskStatus.RUNPOD_RENDERING)
        render_job_id = rp_client.submit_job({
            "action": "render",
            "video_url": video_url,
            "clips_json": clips_json,
            "word_timestamps": transcript_data.get("words", []),
            "task_id": task_id,
            "output_resolution": config.get("output_resolution", "1080x1920"),
            "whisper_model": config.get("whisper_model", "small"),
            "enable_face_tracking": config.get("enable_face_tracking", True),
        })

        render_result = rp_client.poll_until_complete(
            render_job_id,
            poll_interval=15,
            timeout=7200,
        )
        output_urls = render_result.get("output", {}).get("output_urls", [])
        update_status(
            task_id,
            TaskStatus.COMPLETED,
            runpod_job_id=render_job_id,
            output_urls=output_urls,
        )
        logger.info(f"[{task_id}] ✅ COMPLETED | {len(output_urls)} clips produced")

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error(f"[{task_id}] FAILED -- {error_msg}", exc_info=True)
        log_error(task_id, error_msg)

    finally:
        # Clean up temp file
        try:
            if local_path and local_path.exists():
                local_path.unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Processing Pipeline -- URL Mode (YouTube / TikTok / IG etc.)
# ---------------------------------------------------------------------------
def process_url_task(
    task_id: str,
    source_url: str,
    config: dict,
) -> None:
    """
    Pipeline for URL-sourced videos.
    RunPod handles the download via yt-dlp (no local download needed).
    Frontend only handles: transcription job -> AI Brain -> render+download job.
    """
    try:
        logger.info(f"[{task_id}] Starting URL pipeline: {source_url[:80]}")
        cookies_content = config.get("yt_cookies", "")

        # -- Step 1: Transcription via RunPod (RunPod downloads + transcribes) ----
        update_status(task_id, TaskStatus.TRANSCRIPTION)
        rp_client = RunPodClient(
            api_key=config["runpod_api_key"],
            endpoint_id=config["runpod_endpoint_id"],
        )

        logger.info(f"[{task_id}] Submitting transcription job (RunPod will download the video)")
        transcribe_job_id = rp_client.submit_job({
            "action": "transcribe_from_url",
            "source_url": source_url,
            "cookies_content": cookies_content,
            "task_id": task_id,
            "whisper_model": config.get("whisper_model", "small"),
        })
        transcribe_result = rp_client.poll_until_complete(
            transcribe_job_id,
            poll_interval=8,
            timeout=1800,
        )
        transcript_data = transcribe_result.get("output", {})
        transcript_text = transcript_data.get("transcript", "")
        video_duration = float(transcript_data.get("duration", 300.0))
        video_url = transcript_data.get("source_storage_url", "")

        # -- Step 2: AI Brain Viral Clip Analysis ---------------------------------
        update_status(task_id, TaskStatus.AI_CURATING)
        brain = AIBrain(
            base_url=config["llm_base_url"],
            api_key=config["llm_api_key"],
            model=config["llm_model"],
        )
        clips_json = brain.analyze_transcript(transcript_text, video_duration)
        update_status(task_id, TaskStatus.AI_CURATING, ai_json=clips_json)
        logger.info(f"[{task_id}] AI returned {len(clips_json)} clips")

        # -- Step 3: RunPod Render (download_and_process action) ------------------
        update_status(task_id, TaskStatus.RUNPOD_RENDERING)
        render_job_id = rp_client.submit_job({
            "action": "download_and_process",
            "source_url": source_url,
            "clips_json": clips_json,
            "word_timestamps": transcript_data.get("words", []),
            "cookies_content": cookies_content,
            "task_id": task_id,
            "output_resolution": config.get("output_resolution", "1080x1920"),
            "whisper_model": config.get("whisper_model", "small"),
            "enable_face_tracking": config.get("enable_face_tracking", True),
        })
        render_result = rp_client.poll_until_complete(
            render_job_id,
            poll_interval=15,
            timeout=7200,
        )
        output_urls = render_result.get("output", {}).get("output_urls", [])
        update_status(
            task_id,
            TaskStatus.COMPLETED,
            runpod_job_id=render_job_id,
            output_urls=output_urls,
        )
        logger.info(f"[{task_id}] COMPLETED | {len(output_urls)} clips produced")

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error(f"[{task_id}] FAILED -- {error_msg}", exc_info=True)
        log_error(task_id, error_msg)


# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SupoClip v2.0 — AI Video Clipper",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Premium Dark Theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --primary: #7C3AED;
        --primary-light: #8B5CF6;
        --primary-glow: rgba(124, 58, 237, 0.4);
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
        --bg-dark: #0A0A0F;
        --bg-card: #111117;
        --bg-card2: #16161F;
        --border: rgba(255,255,255,0.08);
        --text-primary: #F1F0FF;
        --text-muted: #6B7280;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        background-color: var(--bg-dark) !important;
        color: var(--text-primary) !important;
    }

    .stApp {
        background: linear-gradient(135deg, #0A0A0F 0%, #0D0A1A 50%, #0A0A0F 100%) !important;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #7C3AED, #EC4899, #F59E0B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: var(--text-muted);
        font-size: 1.05rem;
        font-weight: 400;
    }

    /* Cards */
    .stat-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .stat-card:hover {
        border-color: var(--primary-light);
        box-shadow: 0 0 20px var(--primary-glow);
        transform: translateY(-2px);
    }
    .stat-card .value {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--primary-light);
    }
    .stat-card .label {
        font-size: 0.8rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.2rem;
    }

    /* Status badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-queued    { background: rgba(107,114,128,0.2); color: #9CA3AF; border: 1px solid rgba(107,114,128,0.3); }
    .badge-uploading { background: rgba(59,130,246,0.2); color: #60A5FA; border: 1px solid rgba(59,130,246,0.3); }
    .badge-transcription { background: rgba(245,158,11,0.2); color: #FCD34D; border: 1px solid rgba(245,158,11,0.3); }
    .badge-ai_curating { background: rgba(124,58,237,0.2); color: #A78BFA; border: 1px solid rgba(124,58,237,0.3); }
    .badge-runpod_rendering { background: rgba(59,130,246,0.2); color: #93C5FD; border: 1px solid rgba(59,130,246,0.3); }
    .badge-completed { background: rgba(16,185,129,0.2); color: #6EE7B7; border: 1px solid rgba(16,185,129,0.3); }
    .badge-failed    { background: rgba(239,68,68,0.2); color: #FCA5A5; border: 1px solid rgba(239,68,68,0.3); }

    /* Task row */
    .task-row {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        transition: border-color 0.2s;
    }
    .task-row:hover { border-color: rgba(124,58,237,0.4); }
    .task-filename {
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.25rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 350px;
    }
    .task-meta { font-size: 0.78rem; color: var(--text-muted); }

    /* Viral score bar */
    .viral-bar {
        height: 6px;
        border-radius: 3px;
        background: linear-gradient(90deg, #7C3AED, #EC4899);
        transition: width 0.5s ease;
    }

    /* Clip card */
    .clip-card {
        background: var(--bg-card2);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    .clip-hook { font-weight: 700; font-size: 1rem; margin-bottom: 0.5rem; color: #E2E8F0; }
    .impact-word {
        display: inline-block;
        background: rgba(239,68,68,0.2);
        color: #FCA5A5;
        border: 1px solid rgba(239,68,68,0.4);
        border-radius: 4px;
        padding: 0.1rem 0.4rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.1rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--bg-card) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] .stTextInput > div > div > input,
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: var(--bg-dark) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7C3AED, #6D28D9) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        box-shadow: 0 4px 20px var(--primary-glow) !important;
        transform: translateY(-1px) !important;
    }

    /* File uploader */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(124,58,237,0.05) !important;
        border: 2px dashed rgba(124,58,237,0.4) !important;
        border-radius: 16px !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text-muted) !important;
        font-weight: 500 !important;
        padding: 0.5rem 1.25rem !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7C3AED, #6D28D9) !important;
        color: white !important;
        border-color: transparent !important;
    }

    div[data-testid="stExpander"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
    
    .stMarkdown hr { border-color: var(--border) !important; }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-dark); }
    ::-webkit-scrollbar-thumb { background: #374151; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--primary); }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — Credentials & Settings
# ---------------------------------------------------------------------------
def render_sidebar():
    cfg = load_config()

    with st.sidebar:
        st.markdown("### ✂️ SupoClip v2.0")
        st.markdown("---")

        st.markdown("#### 🔑 RunPod Credentials")
        runpod_key = st.text_input(
            "RunPod API Key",
            value=cfg.get("runpod_api_key", ""),
            type="password",
            key="runpod_api_key_input",
            placeholder="rpa_xxxxxxxxxxxx",
        )
        runpod_endpoint = st.text_input(
            "RunPod Endpoint ID",
            value=cfg.get("runpod_endpoint_id", ""),
            key="runpod_endpoint_input",
            placeholder="xxxxxxxxxxxxxxxx",
        )

        st.markdown("#### 🧠 LLM Configuration")
        llm_base_url = st.text_input(
            "LLM Base URL",
            value=cfg.get("llm_base_url", "https://api.openai.com/v1"),
            key="llm_base_url_input",
            help="Supports any OpenAI-compatible endpoint (Gemini, DeepSeek, Claude via proxy)",
        )
        llm_api_key = st.text_input(
            "LLM API Key",
            value=cfg.get("llm_api_key", ""),
            type="password",
            key="llm_api_key_input",
        )
        llm_model = st.text_input(
            "LLM Model Name",
            value=cfg.get("llm_model", "gpt-4o"),
            key="llm_model_input",
            placeholder="gpt-4o / gemini-2.5-flash / deepseek-r1",
        )

        st.markdown("#### Cookies YouTube *(Opsional)*")
        st.caption(
            "Diperlukan untuk video age-restricted atau private. "
            "Export dari browser: ekstensi **'Get cookies.txt LOCALLY'** "
            "lalu upload file cookies.txt di sini."
        )
        cookies_file = st.file_uploader(
            "Upload cookies.txt (Netscape format)",
            type=["txt"],
            key="cookies_upload",
            label_visibility="collapsed",
        )
        if cookies_file:
            cookies_content = cookies_file.read().decode("utf-8", errors="ignore")
            cfg["yt_cookies"] = cookies_content
            save_config(cfg)
            st.success(f"Cookies saved ({len(cookies_content)} chars)")
        elif cfg.get("yt_cookies"):
            ck_len = len(cfg["yt_cookies"])
            col_ck1, col_ck2 = st.columns([3, 1])
            with col_ck1:
                st.caption(f"Cookies active: {ck_len:,} chars")
            with col_ck2:
                if st.button("Clear", key="clear_cookies"):
                    cfg["yt_cookies"] = ""
                    save_config(cfg)
                    st.rerun()
        else:
            st.caption("No cookies — public videos only")

        st.markdown("#### ☁️ Storage (Zero-Config Active)")
        st.caption("✨ Zero-Registration storage is active by default. S3 fields below are 100% optional.")
        with st.expander("⚙️ Optional Custom S3 / R2 Settings", expanded=False):
            s3_endpoint = st.text_input(
                "S3 Endpoint URL (Optional)",
                value=cfg.get("s3_endpoint", ""),
                key="s3_endpoint_input",
                placeholder="https://<account>.r2.cloudflarestorage.com",
            )
            s3_access_key = st.text_input(
                "S3 Access Key (Optional)",
                value=cfg.get("s3_access_key", ""),
                type="password",
                key="s3_access_key_input",
            )
            s3_secret_key = st.text_input(
                "S3 Secret Key (Optional)",
                value=cfg.get("s3_secret_key", ""),
                type="password",
                key="s3_secret_key_input",
            )
            s3_bucket = st.text_input(
                "S3 Bucket Name (Optional)",
                value=cfg.get("s3_bucket", ""),
                key="s3_bucket_input",
                placeholder="supoclip",
            )

        st.markdown("#### ⚙️ Processing Settings")
        output_res = st.selectbox(
            "Output Resolution",
            options=["1080x1920", "720x1280"],
            index=0 if cfg.get("output_resolution", "1080x1920") == "1080x1920" else 1,
            key="output_res_input",
        )
        whisper_model = st.selectbox(
            "Whisper Model",
            options=["tiny", "base", "small", "medium", "large"],
            index=["tiny", "base", "small", "medium", "large"].index(
                cfg.get("whisper_model", "small")
            ),
            key="whisper_model_input",
            help="Larger = more accurate but slower on RunPod GPU",
        )
        face_tracking = st.toggle(
            "🎯 Face Tracking (MediaPipe)",
            value=cfg.get("enable_face_tracking", True),
            key="face_tracking_input",
            help="Intelligently centers crop on speaker's face",
        )

        if st.button("💾 Save Configuration", use_container_width=True):
            new_cfg = {
                "runpod_api_key": runpod_key,
                "runpod_endpoint_id": runpod_endpoint,
                "llm_base_url": llm_base_url,
                "llm_api_key": llm_api_key,
                "llm_model": llm_model,
                "s3_endpoint": s3_endpoint,
                "s3_access_key": s3_access_key,
                "s3_secret_key": s3_secret_key,
                "s3_bucket": s3_bucket,
                "output_resolution": output_res,
                "whisper_model": whisper_model,
                "enable_face_tracking": face_tracking,
            }
            save_config(new_cfg)
            st.success("✅ Configuration saved!")

        st.markdown("---")

        # Endpoint health check
        if st.button("🩺 Check RunPod Health", use_container_width=True):
            if runpod_key and runpod_endpoint:
                try:
                    rp = RunPodClient(runpod_key, runpod_endpoint)
                    health = rp.check_health()
                    if "error" in health:
                        st.error(f"❌ Connection failed: {health['error']}")
                    else:
                        workers = health.get("workers", {})
                        idle = workers.get("idle", 0)
                        running = workers.get("running", 0)
                        in_queue = health.get("jobs", {}).get("inQueue", 0)
                        st.success(f"✅ Terhubung ke RunPod! | Status: Siap (Standby Auto-Scale 0/2 GPU) | Jobs: {in_queue}")
                except Exception as e:
                    st.error(f"❌ {e}")
            else:
                st.warning("Masukkan RunPod API Key & Endpoint ID terlebih dahulu")

        return load_config()


# ---------------------------------------------------------------------------
# Stat Cards
# ---------------------------------------------------------------------------
def render_stats(tasks: list[dict]):
    total = len(tasks)
    completed = sum(1 for t in tasks if t["status"] == TaskStatus.COMPLETED)
    processing = sum(
        1 for t in tasks
        if t["status"] not in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.QUEUED)
    )
    failed = sum(1 for t in tasks if t["status"] == TaskStatus.FAILED)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="stat-card">
            <div class="value">{total}</div>
            <div class="label">Total Videos</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="stat-card">
            <div class="value" style="color:#6EE7B7">{completed}</div>
            <div class="label">Completed</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="stat-card">
            <div class="value" style="color:#93C5FD">{processing}</div>
            <div class="label">Processing</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="stat-card">
            <div class="value" style="color:#FCA5A5">{failed}</div>
            <div class="label">Failed</div>
        </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Task Queue Renderer
# ---------------------------------------------------------------------------
def render_task_row(task: dict):
    status = task["status"]
    status_label = STATUS_EMOJI.get(status, status)
    badge_class = f"badge-{status}"
    filename = task.get("filename", "unknown")
    created = task.get("created_at", "")[:19].replace("T", " ")
    error = task.get("error_log", "")

    with st.container():
        col1, col2, col3, col4 = st.columns([4, 2, 2, 1])

        with col1:
            st.markdown(
                f'<div class="task-filename" title="{filename}">{filename}</div>'
                f'<div class="task-meta">Added: {created} UTC</div>',
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f'<div style="padding-top:0.6rem">'
                f'<span class="status-badge {badge_class}">{status_label}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with col3:
            ai_json = task.get("ai_json")
            if ai_json and isinstance(ai_json, list) and len(ai_json) > 0:
                avg_score = sum(c.get("viral_score", 0) for c in ai_json) / len(ai_json)
                st.markdown(
                    f'<div style="padding-top:0.5rem">'
                    f'<div style="font-size:0.78rem;color:#6B7280;margin-bottom:0.2rem">'
                    f'Viral Avg: {avg_score:.0f}/100 ({len(ai_json)} clips)</div>'
                    f'<div style="background:#1F2937;border-radius:3px;height:6px;overflow:hidden">'
                    f'<div class="viral-bar" style="width:{avg_score}%"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            elif error:
                st.markdown(
                    f'<div style="padding-top:0.5rem;font-size:0.75rem;color:#FCA5A5;'
                    f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:200px"'
                    f'title="{error}">⚠ {error[:60]}...</div>',
                    unsafe_allow_html=True,
                )

        with col4:
            if st.button("🗑", key=f"del_{task['id']}", help="Remove from queue"):
                delete_task(task["id"])
                st.rerun()

        st.markdown('<hr style="border-color:rgba(255,255,255,0.05);margin:0.5rem 0">', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# AI Preview Panel
# ---------------------------------------------------------------------------
def render_ai_preview(tasks: list[dict]):
    completed_tasks = [t for t in tasks if t.get("ai_json") and t["status"] in (
        TaskStatus.COMPLETED, TaskStatus.RUNPOD_RENDERING, TaskStatus.AI_CURATING
    )]

    if not completed_tasks:
        st.markdown("""
        <div style="text-align:center;padding:4rem;color:#4B5563">
            <div style="font-size:3rem;margin-bottom:1rem">🧠</div>
            <div style="font-size:1.1rem;font-weight:500">No AI analysis yet</div>
            <div style="font-size:0.85rem;margin-top:0.5rem">Upload and process videos to see viral clip analysis here</div>
        </div>
        """, unsafe_allow_html=True)
        return

    for task in completed_tasks:
        ai_json = task["ai_json"]
        with st.expander(
            f"📽 {task['filename']} — {len(ai_json)} clips detected",
            expanded=True,
        ):
            for clip in ai_json:
                score = clip.get("viral_score", 0)
                score_color = "#6EE7B7" if score >= 80 else "#FCD34D" if score >= 60 else "#FCA5A5"
                impact_words_html = " ".join(
                    f'<span class="impact-word">{w}</span>'
                    for w in clip.get("high_impact_words", [])
                )
                st.markdown(f"""
                <div class="clip-card">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.5rem">
                        <div class="clip-hook">#{clip.get('clip_id','')} {clip.get('hook_title','')}</div>
                        <div style="font-size:1.4rem;font-weight:800;color:{score_color};white-space:nowrap;margin-left:1rem">
                            {score}<span style="font-size:0.75rem;color:#6B7280">/100</span>
                        </div>
                    </div>
                    <div style="font-size:0.82rem;color:#6B7280;margin-bottom:0.4rem">
                        ⏱ {clip.get('start_time',0):.1f}s → {clip.get('end_time',0):.1f}s
                        &nbsp;|&nbsp; ⏳ {clip.get('duration',0):.1f}s
                        &nbsp;|&nbsp; 💥 SFX @{clip.get('sfx_timestamp',0):.1f}s ({clip.get('sfx_type','')})
                    </div>
                    <div style="font-size:0.82rem;color:#9CA3AF;margin-bottom:0.6rem;font-style:italic">
                        "{clip.get('emotional_peak','')}"
                    </div>
                    <div>{impact_words_html}</div>
                </div>
                """, unsafe_allow_html=True)

            # Raw JSON viewer
            with st.expander("🔍 Raw JSON Output"):
                st.json(ai_json)

            # Output clips download section
            output_urls = task.get("output_urls", [])
            if output_urls:
                st.markdown("#### 📥 Download Rendered Clips")
                for i, url in enumerate(output_urls, 1):
                    st.markdown(f"[📥 Download Clip #{i}]({url})")


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
def main():
    # Render sidebar and get current config
    config = render_sidebar()

    # Page header
    st.markdown("""
    <div class="main-header">
        <h1>✂️ SupoClip v2.0</h1>
        <p>Mass AI Video Clipper & Auto-Editor — Open-source CapCut Pro Alternative</p>
    </div>
    """, unsafe_allow_html=True)

    # Live task data
    tasks = get_all_tasks()

    # Stats
    render_stats(tasks)
    st.markdown("<br>", unsafe_allow_html=True)

    # Main tabs
    tab_url, tab_upload, tab_queue, tab_preview = st.tabs([
        "🔗  URL Input",
        "📥  Upload Videos",
        "📊  Task Queue",
        "🔍  AI Preview",
    ])

    # ── URL Input Tab ─────────────────────────────────────────────────────
    with tab_url:
        st.markdown("#### Paste Video URL — YouTube / TikTok / Instagram / Facebook")
        st.markdown(
            '<div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:1rem">'
            '<span style="background:rgba(255,0,0,0.15);color:#FF6B6B;border:1px solid rgba(255,0,0,0.3);'
            'padding:0.2rem 0.75rem;border-radius:999px;font-size:0.8rem;font-weight:600">▶ YouTube</span>'
            '<span style="background:rgba(0,0,0,0.3);color:#E0E0E0;border:1px solid rgba(255,255,255,0.15);'
            'padding:0.2rem 0.75rem;border-radius:999px;font-size:0.8rem;font-weight:600">♪ TikTok</span>'
            '<span style="background:rgba(193,53,132,0.15);color:#E1306C;border:1px solid rgba(193,53,132,0.3);'
            'padding:0.2rem 0.75rem;border-radius:999px;font-size:0.8rem;font-weight:600">◉ Instagram</span>'
            '<span style="background:rgba(24,119,242,0.15);color:#4B9CF5;border:1px solid rgba(24,119,242,0.3);'
            'padding:0.2rem 0.75rem;border-radius:999px;font-size:0.8rem;font-weight:600">f Facebook</span>'
            '<span style="background:rgba(100,100,100,0.15);color:#9CA3AF;border:1px solid rgba(100,100,100,0.3);'
            'padding:0.2rem 0.75rem;border-radius:999px;font-size:0.8rem;font-weight:600">+ 1000 sites</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        col_url_main, col_url_side = st.columns([3, 1])

        with col_url_main:
            url_input_raw = st.text_area(
                "Video URLs (satu URL per baris)",
                height=160,
                key="url_input",
                placeholder=(
                    "https://www.youtube.com/watch?v=xxxx\n"
                    "https://www.tiktok.com/@user/video/xxxx\n"
                    "https://www.instagram.com/reel/xxxx\n"
                    "https://www.facebook.com/watch?v=xxxx"
                ),
                label_visibility="collapsed",
            )

        with col_url_side:
            st.markdown("#### Status")
            has_cookies = bool(config.get("yt_cookies", "").strip())
            if has_cookies:
                st.success("🍪 Cookies aktif")
            else:
                st.info("🔓 Public videos only\n\nUpload cookies di sidebar untuk YouTube age-restricted/private")

            st.markdown("---")
            st.caption(
                f"🧠 **{config.get('llm_model', 'Not set')}**  \n"
                f"📐 **{config.get('output_resolution', '1080x1920')}**  \n"
                f"🎙️ Whisper: **{config.get('whisper_model', 'small')}**"
            )

        # Parse URLs
        urls = [u.strip() for u in (url_input_raw or "").splitlines() if u.strip().startswith("http")]

        if urls:
            st.markdown(f"**{len(urls)} URL siap diproses**")

            # Validate credentials
            missing_url = []
            for key in ["runpod_api_key", "runpod_endpoint_id", "llm_api_key"]:
                if not config.get(key):
                    missing_url.append(key)

            if missing_url:
                st.error(
                    f"⚠️ Lengkapi konfigurasi berikut di sidebar: "
                    f"{', '.join(missing_url)}"
                )
            else:
                process_url_btn = st.button(
                    f"🚀 Clip {len(urls)} Video dari URL",
                    use_container_width=True,
                    type="primary",
                    key="process_url_btn",
                )
                if process_url_btn:
                    for raw_url in urls:
                        # Derive filename from URL
                        url_hash = hashlib.md5(raw_url.encode()).hexdigest()[:8]
                        task_id = f"url_{url_hash}_{uuid.uuid4().hex[:8]}"
                        # Use domain as display filename
                        from urllib.parse import urlparse
                        parsed = urlparse(raw_url)
                        domain = parsed.netloc.replace("www.", "")
                        path_slug = parsed.path.strip("/").replace("/", "_")[:30]
                        display_name = f"[{domain}] {path_slug or url_hash}"

                        add_task(task_id, display_name, 0)

                        thread = threading.Thread(
                            target=process_url_task,
                            args=(task_id, raw_url, config),
                            daemon=True,
                            name=f"supoclip-url-{task_id[:8]}",
                        )
                        thread.start()
                        logger.info(f"Launched URL thread for task {task_id} | url={raw_url[:60]}")

                    st.success(f"✅ {len(urls)} URL ditambahkan ke antrian!")
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
        else:
            st.markdown(
                '<div style="text-align:center;padding:3rem;color:#4B5563">'
                '<div style="font-size:3rem;margin-bottom:1rem">🔗</div>'
                '<div style="font-size:1.05rem;font-weight:500">Paste link video di atas</div>'
                '<div style="font-size:0.85rem;margin-top:0.4rem">Bisa paste banyak URL sekaligus — 1 URL per baris</div>'
                '</div>',
                unsafe_allow_html=True,
            )



    # ── Upload Tab ────────────────────────────────────────────────────────
    with tab_upload:
        col_main, col_side = st.columns([3, 1])

        with col_main:
            st.markdown("#### Drag & Drop Your Videos")
            uploaded_files = st.file_uploader(
                "Drop .mp4 files here — multiple files supported",
                type=["mp4", "mov", "avi", "mkv"],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )

        with col_side:
            st.markdown("#### Quick Settings")
            auto_process = st.toggle(
                "Auto-process on upload",
                value=True,
                key="auto_process",
            )
            st.info(
                f"🧠 Model: **{config.get('llm_model', 'Not set')}**\n\n"
                f"🎬 GPU: **RunPod**\n\n"
                f"📐 Output: **{config.get('output_resolution', '1080x1920')}**"
            )

        if uploaded_files:
            st.markdown(f"**{len(uploaded_files)} file(s) ready to process**")

            # Validate config before processing (S3 is optional)
            missing = []
            for key in ["runpod_api_key", "runpod_endpoint_id", "llm_api_key"]:
                if not config.get(key):
                    missing.append(key)

            if missing:
                st.error(
                    f"⚠️ Please configure the following in the sidebar before processing: "
                    f"{', '.join(missing)}"
                )
            else:
                process_btn = st.button(
                    f"🚀 Process {len(uploaded_files)} Video(s)",
                    use_container_width=True,
                    type="primary",
                )

                if process_btn or auto_process:
                    for uploaded_file in uploaded_files:
                        filename = uploaded_file.name

                        # Generate unique task ID based on filename + content hash
                        content_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
                        task_id = f"{content_hash}_{uuid.uuid4().hex[:8]}"

                        # Save to temp file
                        temp_path = TEMP_DIR / f"{task_id}_{filename}"
                        temp_path.write_bytes(uploaded_file.read())
                        file_size = temp_path.stat().st_size

                        # Add to queue
                        add_task(task_id, filename, file_size)

                        # Launch background thread (fully isolated per video)
                        thread = threading.Thread(
                            target=process_video_task,
                            args=(task_id, temp_path, filename, config),
                            daemon=True,
                            name=f"supoclip-{task_id[:8]}",
                        )
                        thread.start()
                        logger.info(f"Launched thread for task {task_id} | file={filename}")

                    st.success(
                        f"✅ {len(uploaded_files)} video(s) added to queue and processing started!"
                    )
                    st.balloons()
                    time.sleep(1)
                    st.rerun()

    # ── Queue Tab ─────────────────────────────────────────────────────────
    with tab_queue:
        col_h, col_actions = st.columns([4, 2])
        with col_h:
            st.markdown(f"#### Task Queue — {len(tasks)} Total")
        with col_actions:
            col_r, col_c = st.columns(2)
            with col_r:
                if st.button("🔄 Refresh", use_container_width=True):
                    st.rerun()
            with col_c:
                if st.button("🧹 Clear Done", use_container_width=True):
                    n = clear_completed()
                    st.success(f"Cleared {n} task(s)")
                    st.rerun()

        if not tasks:
            st.markdown("""
            <div style="text-align:center;padding:4rem;color:#4B5563">
                <div style="font-size:3rem;margin-bottom:1rem">📭</div>
                <div style="font-size:1rem;font-weight:500">Queue is empty</div>
                <div style="font-size:0.85rem;margin-top:0.5rem">Upload videos in the Upload tab to get started</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Status filter
            status_filter = st.multiselect(
                "Filter by status",
                options=[s.value for s in TaskStatus],
                default=[],
                key="status_filter",
                label_visibility="collapsed",
                placeholder="Filter by status...",
            )

            filtered = (
                [t for t in tasks if t["status"] in status_filter]
                if status_filter else tasks
            )

            for task in filtered:
                render_task_row(task)

        # Auto-refresh if there are active tasks
        active = [
            t for t in tasks
            if t["status"] not in (TaskStatus.COMPLETED, TaskStatus.FAILED)
        ]
        if active:
            st.markdown(
                f'<div style="text-align:center;font-size:0.8rem;color:#6B7280;margin-top:1rem">'
                f'⟳ Auto-refreshing every 10s ({len(active)} active task(s))</div>',
                unsafe_allow_html=True,
            )
            time.sleep(10)
            st.rerun()

    # ── AI Preview Tab ────────────────────────────────────────────────────
    with tab_preview:
        render_ai_preview(get_all_tasks())


if __name__ == "__main__":
    main()
