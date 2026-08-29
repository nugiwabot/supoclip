"""
SupoClip — URL Video Downloader
Handles video download from YouTube, TikTok, Instagram, Facebook, Twitter/X
via yt-dlp with optional YouTube cookies support (for age-restricted/private videos).

Supports:
- YouTube (including shorts, age-restricted)
- TikTok (public videos)
- Instagram Reels
- Facebook Video
- Twitter/X
- And 1000+ sites supported by yt-dlp

Cookie setup (one-time only):
  Export via browser extension: "Get cookies.txt LOCALLY" (Chrome/Firefox)
  or "Cookie-Editor" > Export > Netscape format
"""

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SUPPORTED_DOMAINS = [
    "youtube.com", "youtu.be",      # YouTube
    "tiktok.com",                   # TikTok
    "instagram.com",                # Instagram Reels
    "facebook.com", "fb.watch",     # Facebook
    "twitter.com", "x.com",         # Twitter/X
    "vimeo.com",                    # Vimeo
    "dailymotion.com",              # Dailymotion
    "twitch.tv",                    # Twitch clips
]


def is_supported_url(url: str) -> bool:
    """Check if URL is from a known supported platform."""
    url_lower = url.lower()
    return any(domain in url_lower for domain in SUPPORTED_DOMAINS)


def get_platform_name(url: str) -> str:
    """Return human-readable platform name from URL."""
    url_lower = url.lower()
    if "youtube.com" in url_lower or "youtu.be" in url_lower:
        return "YouTube"
    elif "tiktok.com" in url_lower:
        return "TikTok"
    elif "instagram.com" in url_lower:
        return "Instagram"
    elif "facebook.com" in url_lower or "fb.watch" in url_lower:
        return "Facebook"
    elif "twitter.com" in url_lower or "x.com" in url_lower:
        return "Twitter/X"
    return "Unknown"


def download_video(
    url: str,
    output_dir: Path,
    cookies_content: Optional[str] = None,
    max_height: int = 1080,
    task_id: str = "unknown",
) -> Path:
    """
    Download a video using yt-dlp and return the local file path.

    Args:
        url:             The video URL (YouTube, TikTok, IG, etc.)
        output_dir:      Directory to save the downloaded video.
        cookies_content: Optional Netscape-format cookies string (from YouTube login).
        max_height:      Max video height (default 1080p).
        task_id:         Task ID for logging.

    Returns:
        Path to the downloaded .mp4 file.

    Raises:
        RuntimeError: If download fails.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    platform = get_platform_name(url)
    logger.info(f"[{task_id}] Downloading {platform} video: {url[:80]}")

    # Build output template
    output_template = str(output_dir / f"{task_id}_source.%(ext)s")

    # Base yt-dlp command
    cmd = [
        "yt-dlp",
        "--no-playlist",              # Single video only
        "--no-warnings",
        "--quiet",
        "--progress",
        # Format: best single file up to max_height
        "-f", f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}][ext=mp4]/best[height<={max_height}]/best",
        "--merge-output-format", "mp4",
        "--output", output_template,
        # Retry logic
        "--retries", "5",
        "--fragment-retries", "5",
        "--retry-sleep", "3",
        # ffmpeg for merging
        "--ffmpeg-location", _get_ffmpeg_path(),
    ]

    # Handle cookies (write to temp file if provided)
    cookie_file = None
    if cookies_content and cookies_content.strip():
        try:
            cookie_file_path = output_dir / f"{task_id}_cookies.txt"
            cookie_file_path.write_text(cookies_content, encoding="utf-8")
            cmd.extend(["--cookies", str(cookie_file_path)])
            logger.info(f"[{task_id}] Cookies loaded ({len(cookies_content)} bytes)")
            cookie_file = cookie_file_path
        except Exception as e:
            logger.warning(f"[{task_id}] Failed to write cookies file: {e}")

    # TikTok/Instagram: add user-agent spoof
    if "tiktok.com" in url.lower() or "instagram.com" in url.lower():
        cmd.extend([
            "--add-header",
            "User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        ])

    cmd.append(url)

    logger.info(f"[{task_id}] Running: {' '.join(cmd[:8])}...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            error_details = stderr or stdout
            raise RuntimeError(
                f"yt-dlp exited with code {result.returncode}: {error_details[-500:]}"
            )

        # Find the downloaded file
        candidates = list(output_dir.glob(f"{task_id}_source.*"))
        if not candidates:
            raise RuntimeError("yt-dlp completed but no output file found.")

        # Prefer .mp4
        mp4_files = [f for f in candidates if f.suffix.lower() == ".mp4"]
        final_path = mp4_files[0] if mp4_files else candidates[0]

        size_mb = final_path.stat().st_size / 1_048_576
        logger.info(
            f"[{task_id}] Download complete: {final_path.name} ({size_mb:.1f} MB)"
        )
        return final_path

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Download timed out after 10 minutes for URL: {url}")

    finally:
        # Clean up cookies file
        if cookie_file and cookie_file.exists():
            try:
                cookie_file.unlink()
            except Exception:
                pass


def get_video_metadata(url: str, cookies_content: Optional[str] = None) -> dict:
    """
    Fetch video metadata (title, duration, uploader) without downloading.
    Used by frontend to preview before processing.
    """
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--skip-download",
        "--print", "%(title)s|%(duration)s|%(uploader)s|%(thumbnail)s",
        "--ffmpeg-location", _get_ffmpeg_path(),
    ]

    if cookies_content and cookies_content.strip():
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(cookies_content)
            cmd.extend(["--cookies", f.name])

    cmd.append(url)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            parts = result.stdout.strip().split("|")
            if len(parts) >= 3:
                return {
                    "title": parts[0],
                    "duration": float(parts[1]) if parts[1].isdigit() else 0,
                    "uploader": parts[2],
                    "thumbnail": parts[3] if len(parts) > 3 else "",
                    "platform": get_platform_name(url),
                }
    except Exception as e:
        logger.warning(f"Failed to get metadata: {e}")

    return {
        "title": "Unknown",
        "duration": 0,
        "uploader": "Unknown",
        "thumbnail": "",
        "platform": get_platform_name(url),
    }


def _get_ffmpeg_path() -> str:
    """Try to locate ffmpeg binary."""
    # Check environment variable first
    ffmpeg_env = os.environ.get("FFMPEG_PATH", "")
    if ffmpeg_env and Path(ffmpeg_env).exists():
        return ffmpeg_env

    # Common paths inside Docker container
    for candidate in ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "ffmpeg"]:
        try:
            result = subprocess.run(
                [candidate, "-version"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                return candidate
        except Exception:
            continue

    return "ffmpeg"  # Rely on PATH
