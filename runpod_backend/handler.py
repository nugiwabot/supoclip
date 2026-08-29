"""
SupoClip -- RunPod Serverless Worker Handler
Entry point for the RunPod serverless container.

Handles four actions:
  1. "transcribe"           -- Transcribe a hosted video URL and return word-level timestamps
  2. "transcribe_from_url"  -- Download from YouTube/TikTok/IG via yt-dlp then transcribe
  3. "render"               -- Full pipeline: download + reframe + audio mix + captions
  4. "download_and_process" -- Download from URL + full pipeline + upload output clips

Environment Variables (ALL OPTIONAL — uses zero-registration storage by default):
  SUPOCLIP_S3_ENDPOINT     -- S3 endpoint URL
  SUPOCLIP_S3_ACCESS_KEY   -- S3 access key
  SUPOCLIP_S3_SECRET_KEY   -- S3 secret key
  SUPOCLIP_S3_BUCKET       -- S3 bucket name
"""

# ── stdlib only — MUST be importable instantly for fast registration ──────────
import json
import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path

# ── third-party light imports (always present in image) ──────────────────────
import requests
import runpod

# ── Startup banner (unbuffered so RunPod captures it immediately) ─────────────
print("=== SUPOCLIP RUNPOD SERVERLESS WORKER STARTING ===", flush=True)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("supoclip.handler")
logger.info("Handler module loaded — waiting for jobs...")
print("=== HANDLER INITIALIZED ===", flush=True)


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _download_video(url: str, dest: Path) -> Path:
    """Download a video from a direct URL or presigned URL to a local file."""
    logger.info(f"Downloading video: {url[:80]}...")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
    size_mb = dest.stat().st_size / 1_048_576
    logger.info(f"Downloaded: {dest.name} ({size_mb:.1f} MB)")
    return dest


def _upload_file(local_path: Path, remote_key: str = None, presigned_ttl: int = 86400) -> str:
    """
    Upload a rendered video clip to storage.
    Priority: S3 (if credentials provided) → Litterbox (72h) → Uguu.se (48h)
    No signup or credentials required for Litterbox / Uguu fallback.
    """
    local_path = Path(local_path)

    s3_endpoint = os.environ.get("SUPOCLIP_S3_ENDPOINT", "").strip()
    s3_key      = os.environ.get("SUPOCLIP_S3_ACCESS_KEY", "").strip()
    s3_secret   = os.environ.get("SUPOCLIP_S3_SECRET_KEY", "").strip()
    s3_bucket   = os.environ.get("SUPOCLIP_S3_BUCKET", "").strip()

    if s3_key and s3_secret and s3_bucket:
        try:
            import boto3
            from botocore.client import Config

            s3 = boto3.client(
                "s3",
                endpoint_url=s3_endpoint or None,
                aws_access_key_id=s3_key,
                aws_secret_access_key=s3_secret,
                region_name="auto",
                config=Config(signature_version="s3v4"),
            )
            key = remote_key or f"outputs/{local_path.name}"
            s3.upload_file(str(local_path), s3_bucket, key, ExtraArgs={"ContentType": "video/mp4"})
            url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": s3_bucket, "Key": key},
                ExpiresIn=presigned_ttl,
            )
            logger.info(f"Uploaded to S3: s3://{s3_bucket}/{key}")
            return url
        except Exception as e:
            logger.warning(f"S3 upload failed ({e}), falling back to zero-registration storage...")

    # Fallback 1: Litterbox (Catbox) — 72h retention, up to 1 GB
    for attempt in range(1, 4):
        try:
            logger.info(f"Uploading {local_path.name} to Litterbox (attempt {attempt})...")
            with open(local_path, "rb") as f:
                files = {"fileToUpload": (local_path.name, f, "video/mp4")}
                data  = {"reqtype": "fileupload", "time": "72h"}
                resp = requests.post(
                    "https://litterbox.catbox.moe/resources/internals/api.php",
                    data=data, files=files, timeout=180,
                )
            if resp.status_code == 200 and resp.text.strip().startswith("http"):
                url = resp.text.strip()
                logger.info(f"Litterbox upload complete: {url}")
                return url
        except Exception as e:
            logger.warning(f"Litterbox error on attempt {attempt}: {e}")

    # Fallback 2: Uguu.se — 48h retention
    try:
        logger.info(f"Uploading {local_path.name} to Uguu.se...")
        with open(local_path, "rb") as f:
            files = {"files[]": (local_path.name, f, "video/mp4")}
            resp = requests.post("https://uguu.se/upload", files=files, timeout=180)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("success") and data.get("files"):
                url = data["files"][0]["url"]
                logger.info(f"Uguu upload complete: {url}")
                return url
    except Exception as e:
        logger.error(f"Uguu error: {e}")

    raise RuntimeError(f"All storage providers failed for {local_path.name}")


# ---------------------------------------------------------------------------
# Action: Transcribe (from presigned/direct video URL)
# ---------------------------------------------------------------------------

def action_transcribe(job_input: dict) -> dict:
    from core.transcriber import Transcriber

    video_url    = job_input["video_url"]
    task_id      = job_input.get("task_id", "unknown")
    whisper_model = job_input.get("whisper_model", "small")

    logger.info(f"[{task_id}] action_transcribe: {video_url[:80]}")

    with tempfile.TemporaryDirectory(prefix="supoclip_transcribe_") as tmpdir:
        tmpdir     = Path(tmpdir)
        video_path = tmpdir / f"{task_id}_source.mp4"
        _download_video(video_url, video_path)

        transcriber = Transcriber(model_size=whisper_model)
        result      = transcriber.transcribe(video_path)

    return {
        "transcript":             result["text"],
        "timestamped_transcript": result["timestamped_transcript"],
        "words":                  result["words"],
        "duration":               result["duration"],
        "language":               result["language"],
    }


# ---------------------------------------------------------------------------
# Action: Transcribe from URL (YouTube / TikTok / Instagram / etc.)
# ---------------------------------------------------------------------------

def action_transcribe_from_url(job_input: dict) -> dict:
    from core.transcriber import Transcriber
    from core.downloader import download_video

    source_url      = job_input["source_url"]
    task_id         = job_input.get("task_id", "unknown")
    cookies_content = job_input.get("cookies_content", "")
    whisper_model   = job_input.get("whisper_model", "small")

    logger.info(f"[{task_id}] action_transcribe_from_url: {source_url[:80]}")

    with tempfile.TemporaryDirectory(prefix="supoclip_url_transcribe_") as tmpdir:
        tmpdir = Path(tmpdir)

        video_path = download_video(
            url=source_url,
            output_dir=tmpdir,
            cookies_content=cookies_content,
            task_id=task_id,
        )
        size_mb = video_path.stat().st_size / 1_048_576
        logger.info(f"[{task_id}] Downloaded: {video_path.name} ({size_mb:.1f} MB)")

        # NOTE: We do NOT upload the source video here.
        # The render step uses action=download_and_process which re-downloads
        # the video directly from the original URL (YouTube/TikTok/etc.).
        # Litterbox & Uguu block datacenter IPs so uploading would fail anyway.
        # We pass the original source_url back as source_storage_url so the
        # frontend can pass it to the render job.

        transcriber = Transcriber(model_size=whisper_model)
        result      = transcriber.transcribe(video_path)
        logger.info(f"[{task_id}] Transcription done: {len(result.get('words', []))} words, {result.get('duration', 0):.1f}s")

    return {
        "transcript":             result["text"],
        "timestamped_transcript": result["timestamped_transcript"],
        "words":                  result["words"],
        "duration":               result["duration"],
        "language":               result["language"],
        "source_storage_url":     source_url,  # pass original URL back for render step
    }



# ---------------------------------------------------------------------------
# Action: Render (from presigned/direct video URL + pre-computed clip list)
# ---------------------------------------------------------------------------

def action_render(job_input: dict) -> dict:
    from core.transcriber     import Transcriber
    from core.face_tracker    import FaceTracker
    from core.video_editor    import VideoEditor
    from core.audio_mixer     import AudioMixer
    from core.caption_renderer import CaptionRenderer

    video_url          = job_input["video_url"]
    clips_json         = job_input["clips_json"]
    task_id            = job_input.get("task_id", "unknown")
    output_res_str     = job_input.get("output_resolution", "1080x1920")
    whisper_model      = job_input.get("whisper_model", "small")
    enable_face_tracking = job_input.get("enable_face_tracking", True)

    try:
        out_w, out_h = [int(x) for x in output_res_str.lower().split("x")]
    except Exception:
        out_w, out_h = 1080, 1920

    if isinstance(clips_json, str):
        clips_json = json.loads(clips_json)

    output_urls = []

    with tempfile.TemporaryDirectory(prefix="supoclip_render_") as tmpdir:
        tmpdir     = Path(tmpdir)
        video_path = tmpdir / f"{task_id}_source.mp4"
        _download_video(video_url, video_path)

        word_timestamps = job_input.get("word_timestamps")
        if word_timestamps:
            logger.info(f"[{task_id}] Using {len(word_timestamps)} pre-computed word timestamps")
        else:
            logger.info(f"[{task_id}] Transcribing for caption sync...")
            transcriber      = Transcriber(model_size=whisper_model)
            transcription    = transcriber.transcribe(video_path)
            word_timestamps  = transcription["words"]
            logger.info(f"[{task_id}] Got {len(word_timestamps)} word timestamps")

        face_tracker   = FaceTracker() if enable_face_tracking else None
        video_editor   = VideoEditor(
            source_path=video_path,
            face_tracker=face_tracker,
            output_resolution=(out_w, out_h),
        )
        audio_mixer      = AudioMixer()
        caption_renderer = CaptionRenderer()

        for i, clip_info in enumerate(clips_json):
            clip_id          = clip_info.get("clip_id", i + 1)
            high_impact_words = clip_info.get("high_impact_words", [])
            sfx_type         = clip_info.get("sfx_type", "sub_bass")
            clip_start       = float(clip_info["start_time"])
            sfx_ts_abs       = float(clip_info.get("sfx_timestamp", clip_start))
            sfx_offset_rel   = max(0.0, sfx_ts_abs - clip_start)

            try:
                logger.info(f"[{task_id}] Clip {clip_id}/{len(clips_json)}: "
                            f"{clip_info['start_time']}s → {clip_info['end_time']}s")

                raw_path = tmpdir / f"clip_{clip_id}_raw.mp4"
                video_editor.extract_and_reframe_clip(clip_info, raw_path)

                mixed_path = tmpdir / f"clip_{clip_id}_mixed.mp4"
                audio_mixer.process_clip(raw_path, mixed_path, sfx_type=sfx_type, sfx_offset_sec=sfx_offset_rel)

                final_path = tmpdir / f"clip_{clip_id}_final.mp4"
                caption_renderer.add_captions(
                    mixed_path, final_path,
                    word_timestamps=word_timestamps,
                    high_impact_words=high_impact_words,
                    clip_start_time=clip_start,
                    impact_color_idx=i,
                )

                url = _upload_file(final_path, f"outputs/{task_id}/clip_{clip_id:02d}.mp4")
                output_urls.append({
                    "clip_id":    clip_id,
                    "url":        url,
                    "viral_score": clip_info.get("viral_score", 0),
                    "hook_title": clip_info.get("hook_title", ""),
                })
                logger.info(f"[{task_id}] Clip {clip_id} done → {url[:60]}")

            except Exception as e:
                logger.error(f"[{task_id}] Clip {clip_id} FAILED: {e}\n{traceback.format_exc()}")
                output_urls.append({"clip_id": clip_id, "url": None, "error": str(e)})

        video_editor.close()
        if face_tracker:
            face_tracker.close()

    return {
        "output_urls": output_urls,
        "clip_count":  len(clips_json),
        "successful":  sum(1 for c in output_urls if c.get("url")),
    }


# ---------------------------------------------------------------------------
# Action: Download & Process (YouTube / TikTok / Instagram / Facebook)
# ---------------------------------------------------------------------------

def action_download_and_process(job_input: dict) -> dict:
    from core.transcriber      import Transcriber
    from core.face_tracker     import FaceTracker
    from core.video_editor     import VideoEditor
    from core.audio_mixer      import AudioMixer
    from core.caption_renderer import CaptionRenderer
    from core.downloader       import download_video, get_platform_name

    source_url         = job_input["source_url"]
    clips_json         = job_input["clips_json"]
    task_id            = job_input.get("task_id", "unknown")
    cookies_content    = job_input.get("cookies_content", "")
    output_res_str     = job_input.get("output_resolution", "1080x1920")
    whisper_model      = job_input.get("whisper_model", "small")
    enable_face_tracking = job_input.get("enable_face_tracking", True)

    try:
        out_w, out_h = [int(x) for x in output_res_str.lower().split("x")]
    except Exception:
        out_w, out_h = 1080, 1920

    if isinstance(clips_json, str):
        clips_json = json.loads(clips_json)

    platform = get_platform_name(source_url)
    logger.info(f"[{task_id}] download_and_process | platform={platform} | url={source_url[:80]}")

    output_urls = []

    with tempfile.TemporaryDirectory(prefix="supoclip_dlp_") as tmpdir:
        tmpdir = Path(tmpdir)

        logger.info(f"[{task_id}] Downloading via yt-dlp...")
        video_path = download_video(
            url=source_url,
            output_dir=tmpdir,
            cookies_content=cookies_content,
            max_height=out_h,
            task_id=task_id,
        )
        size_mb = video_path.stat().st_size / 1_048_576
        logger.info(f"[{task_id}] Download complete: {video_path.name} ({size_mb:.1f} MB)")

        word_timestamps = job_input.get("word_timestamps")
        if word_timestamps:
            logger.info(f"[{task_id}] Using {len(word_timestamps)} pre-computed word timestamps")
        else:
            logger.info(f"[{task_id}] Transcribing for caption sync...")
            transcriber     = Transcriber(model_size=whisper_model)
            transcription   = transcriber.transcribe(video_path)
            word_timestamps = transcription["words"]
            logger.info(f"[{task_id}] Got {len(word_timestamps)} word timestamps")

        face_tracker   = FaceTracker() if enable_face_tracking else None
        video_editor   = VideoEditor(
            source_path=video_path,
            face_tracker=face_tracker,
            output_resolution=(out_w, out_h),
        )
        audio_mixer      = AudioMixer()
        caption_renderer = CaptionRenderer()

        for i, clip_info in enumerate(clips_json):
            clip_id          = clip_info.get("clip_id", i + 1)
            high_impact_words = clip_info.get("high_impact_words", [])
            sfx_type         = clip_info.get("sfx_type", "sub_bass")
            clip_start       = float(clip_info["start_time"])
            sfx_ts_abs       = float(clip_info.get("sfx_timestamp", clip_start))
            sfx_offset_rel   = max(0.0, sfx_ts_abs - clip_start)

            try:
                logger.info(f"[{task_id}] Clip {clip_id}/{len(clips_json)}: "
                            f"{clip_info['start_time']}s → {clip_info['end_time']}s")

                raw_path = tmpdir / f"clip_{clip_id}_raw.mp4"
                video_editor.extract_and_reframe_clip(clip_info, raw_path)

                mixed_path = tmpdir / f"clip_{clip_id}_mixed.mp4"
                audio_mixer.process_clip(raw_path, mixed_path, sfx_type=sfx_type, sfx_offset_sec=sfx_offset_rel)

                final_path = tmpdir / f"clip_{clip_id}_final.mp4"
                caption_renderer.add_captions(
                    mixed_path, final_path,
                    word_timestamps=word_timestamps,
                    high_impact_words=high_impact_words,
                    clip_start_time=clip_start,
                    impact_color_idx=i,
                )

                url = _upload_file(final_path, f"outputs/{task_id}/clip_{clip_id:02d}.mp4")
                output_urls.append({
                    "clip_id":    clip_id,
                    "url":        url,
                    "viral_score": clip_info.get("viral_score", 0),
                    "hook_title": clip_info.get("hook_title", ""),
                })
                logger.info(f"[{task_id}] Clip {clip_id} done → {url[:60]}")

            except Exception as e:
                logger.error(f"[{task_id}] Clip {clip_id} FAILED: {e}\n{traceback.format_exc()}")
                output_urls.append({"clip_id": clip_id, "url": None, "error": str(e)})

        video_editor.close()
        if face_tracker:
            face_tracker.close()

    return {
        "output_urls":    output_urls,
        "clip_count":     len(clips_json),
        "successful":     sum(1 for c in output_urls if c.get("url")),
        "source_platform": platform,
        "source_url":     source_url,
    }


# ---------------------------------------------------------------------------
# RunPod serverless router — catches ALL exceptions for Zero-Crash Policy
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict:
    job_id    = job.get("id", "unknown")
    job_input = job.get("input", {})
    action    = job_input.get("action", "render")

    logger.info(f"Job received | id={job_id} | action={action}")

    try:
        if action == "transcribe":
            result = action_transcribe(job_input)
        elif action == "transcribe_from_url":
            result = action_transcribe_from_url(job_input)
        elif action == "render":
            result = action_render(job_input)
        elif action == "download_and_process":
            result = action_download_and_process(job_input)
        else:
            return {
                "error": (
                    f"Unknown action: '{action}'. "
                    "Valid actions: 'transcribe', 'transcribe_from_url', 'render', 'download_and_process'"
                )
            }

        logger.info(f"Job {job_id} completed successfully")
        return result

    except KeyError as e:
        msg = f"Missing required field: {e}"
        logger.error(f"Job {job_id} — {msg}")
        return {"error": msg, "error_type": "KeyError"}

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Job {job_id} FAILED:\n{tb}")
        return {
            "error":      str(e),
            "error_type": type(e).__name__,
            "traceback":  tb,
        }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info(f"SupoClip RunPod Worker ready | Python {sys.version.split()[0]} | cwd={os.getcwd()}")
    runpod.serverless.start({"handler": handler})
