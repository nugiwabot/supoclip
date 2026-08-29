"""
SupoClip v2.0 — System Validation Script
Checks all dependencies, assets, and configuration before deployment.

Usage:
    python validate.py              # validate everything
    python validate.py --frontend   # frontend only
    python validate.py --backend    # backend only
"""

import sys
import os
import json
import importlib
import subprocess
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
FRONTEND_DIR = ROOT / "vps_frontend"
BACKEND_DIR = ROOT / "runpod_backend"

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "
INFO = "ℹ️ "

errors = []
warnings = []


def check(label: str, condition: bool, hint: str = "", warn_only: bool = False):
    if condition:
        print(f"  {PASS}  {label}")
    else:
        marker = WARN if warn_only else FAIL
        print(f"  {marker}  {label}")
        if hint:
            print(f"       → {hint}")
        if warn_only:
            warnings.append(label)
        else:
            errors.append(label)


def check_import(pkg_name: str, import_name: str = None, warn_only: bool = False):
    import_name = import_name or pkg_name
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "?")
        print(f"  {PASS}  {pkg_name} ({ver})")
        return True
    except ImportError:
        marker = WARN if warn_only else FAIL
        print(f"  {marker}  {pkg_name} — NOT INSTALLED")
        hint = f"pip install {pkg_name}"
        if warn_only:
            warnings.append(pkg_name)
        else:
            errors.append(pkg_name)
        return False


def section(title: str):
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


def validate_frontend():
    section("FRONTEND — Python Packages")
    check_import("streamlit")
    check_import("openai")
    check_import("requests")
    check_import("boto3")
    check_import("dotenv", "dotenv", warn_only=True)

    section("FRONTEND — Source Files")
    for f in [
        "vps_frontend/app.py",
        "vps_frontend/core/db.py",
        "vps_frontend/core/ai_brain.py",
        "vps_frontend/core/runpod_client.py",
        "vps_frontend/core/storage.py",
        "vps_frontend/.streamlit/config.toml",
        "vps_frontend/requirements.txt",
    ]:
        check(f, (ROOT / f).exists(), f"Missing: {ROOT / f}")

    section("FRONTEND — Config File")
    config_path = FRONTEND_DIR / ".supoclip_config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            required_keys = ["runpod_api_key", "runpod_endpoint_id", "llm_api_key"]
            for key in required_keys:
                check(
                    f"Config key: {key}",
                    bool(cfg.get(key, "").strip()),
                    f"Set '{key}' in the Streamlit sidebar",
                    warn_only=True,
                )
        except Exception as e:
            print(f"  {WARN}  Config file exists but invalid JSON: {e}")
    else:
        print(f"  {INFO}  No config file yet — fill in credentials in Streamlit sidebar after launch")


def validate_backend():
    section("BACKEND — Source Files")
    for f in [
        "runpod_backend/handler.py",
        "runpod_backend/core/transcriber.py",
        "runpod_backend/core/face_tracker.py",
        "runpod_backend/core/video_editor.py",
        "runpod_backend/core/audio_mixer.py",
        "runpod_backend/core/caption_renderer.py",
        "runpod_backend/Dockerfile",
        "runpod_backend/requirements.txt",
    ]:
        check(f, (ROOT / f).exists(), f"Missing: {ROOT / f}")

    section("BACKEND — Assets")
    sfx_dir = BACKEND_DIR / "assets" / "sfx"
    music_dir = BACKEND_DIR / "assets" / "music"
    fonts_dir = BACKEND_DIR / "assets" / "fonts"

    sfx_files = list(sfx_dir.glob("*.mp3")) if sfx_dir.exists() else []
    music_files = list(music_dir.glob("*.mp3")) + list(music_dir.glob("*.wav")) if music_dir.exists() else []
    font_files = list(fonts_dir.glob("*.ttf")) if fonts_dir.exists() else []

    check("SFX: whoosh.mp3", (sfx_dir / "whoosh.mp3").exists(),
          "Run: python runpod_backend/download_assets.py", warn_only=True)
    check("SFX: ding.mp3", (sfx_dir / "ding.mp3").exists(),
          "Run: python runpod_backend/download_assets.py", warn_only=True)
    check("SFX: sub_bass.mp3", (sfx_dir / "sub_bass.mp3").exists(),
          "Run: python runpod_backend/download_assets.py", warn_only=True)
    check(f"Music: {len(music_files)} file(s)", len(music_files) > 0,
          "Run: python runpod_backend/download_assets.py", warn_only=True)
    check(f"Fonts: {len(font_files)} file(s)", len(font_files) > 0,
          "Run: python runpod_backend/download_assets.py", warn_only=True)

    section("BACKEND — System Tools")
    # Check FFmpeg
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        ver_line = result.stdout.split("\n")[0]
        print(f"  {PASS}  FFmpeg — {ver_line[:50]}")
    except FileNotFoundError:
        print(f"  {FAIL}  FFmpeg — NOT FOUND")
        print(f"       → Windows: winget install ffmpeg  OR  choco install ffmpeg")
        errors.append("FFmpeg")

    # Check Docker
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        print(f"  {PASS}  Docker — {result.stdout.strip()}")
    except FileNotFoundError:
        print(f"  {WARN}  Docker — NOT FOUND (required for RunPod deployment)")
        print(f"       → Download: https://docker.com/products/docker-desktop")
        warnings.append("Docker")

    section("BACKEND — Python Packages (local test)")
    check_import("runpod", warn_only=True)
    check_import("moviepy", warn_only=True)
    check_import("PIL", "PIL", warn_only=True)
    check_import("cv2", "cv2", warn_only=True)
    check_import("numpy", warn_only=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║       SupoClip v2.0 — System Validation              ║")
    print("╚══════════════════════════════════════════════════════╝")

    if mode in ("all", "--frontend"):
        validate_frontend()
    if mode in ("all", "--backend"):
        validate_backend()

    # Summary
    print(f"\n{'═' * 55}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'═' * 55}")
    if not errors and not warnings:
        print(f"  {PASS}  All checks passed! Ready to deploy.")
    else:
        if errors:
            print(f"\n  {FAIL}  {len(errors)} ERROR(S) — must fix before running:")
            for e in errors:
                print(f"      • {e}")
        if warnings:
            print(f"\n  {WARN}  {len(warnings)} WARNING(S) — optional but recommended:")
            for w in warnings:
                print(f"      • {w}")

    print(f"\n{'─' * 55}")
    print(f"  Next step:")
    if errors:
        print(f"  Fix the errors above, then run this script again.")
    else:
        print(f"  cd vps_frontend && streamlit run app.py")
    print()

    return len(errors)


if __name__ == "__main__":
    sys.exit(main())
