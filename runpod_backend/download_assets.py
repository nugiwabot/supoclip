"""
SupoClip -- Asset Downloader
Auto-downloads free CC0 SFX and a sample lo-fi music track for testing.
Run this once before building the Docker image.

Usage:
    python download_assets.py

Sources:
    - SFX: Generated via FFmpeg (no external dependency needed)
    - Music: Free sample from GitHub (CC0)
"""

import subprocess
import sys
import os
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ASSETS_DIR = Path(__file__).parent / "assets"
SFX_DIR = ASSETS_DIR / "sfx"
MUSIC_DIR = ASSETS_DIR / "music"
FONTS_DIR = ASSETS_DIR / "fonts"


def generate_sfx_with_ffmpeg():
    """
    Generate synthetic SFX using FFmpeg (no internet needed).
    These are functional placeholders — replace with real SFX for production.
    """
    print("[SFX] Generating SFX placeholders with FFmpeg...")

    sfx_commands = {
        # Whoosh: white noise burst with pitch sweep
        "whoosh.mp3": [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "anoisesrc=d=1.0:c=white:r=44100:a=0.8",
            "-af", "afade=t=in:st=0:d=0.05,afade=t=out:st=0.7:d=0.3,atempo=1.0",
            "-ar", "44100", "-ac", "2", "-b:a", "192k",
            str(SFX_DIR / "whoosh.mp3")
        ],
        # Ding: 880Hz sine bell with decay
        "ding.mp3": [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "sine=frequency=880:duration=1.5",
            "-af", "afade=t=in:st=0:d=0.005,afade=t=out:st=0.3:d=1.2,volume=0.9",
            "-ar", "44100", "-ac", "2", "-b:a", "192k",
            str(SFX_DIR / "ding.mp3")
        ],
        # Sub Bass: 60Hz sine burst for impact
        "sub_bass.mp3": [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "sine=frequency=60:duration=1.2",
            "-af", "afade=t=in:st=0:d=0.01,afade=t=out:st=0.6:d=0.6,volume=1.5",
            "-ar", "44100", "-ac", "2", "-b:a", "192k",
            str(SFX_DIR / "sub_bass.mp3")
        ],
    }

    for name, cmd in sfx_commands.items():
        dest = SFX_DIR / name
        if dest.exists():
            print(f"   ✅ {name} already exists, skipping")
            continue
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and dest.exists():
                size_kb = dest.stat().st_size // 1024
                print(f"   [OK] Generated: {name} ({size_kb} KB)")
            else:
                print(f"   [WARN] FFmpeg error for {name}: {result.stderr[-200:]}")
        except FileNotFoundError:
            print("   [ERR] FFmpeg not found. Please install FFmpeg first.")
            print("      Windows: winget install ffmpeg")
            break
        except Exception as e:
            print(f"   [ERR] Error generating {name}: {e}")


def generate_lofi_music():
    """
    Generate a simple 60-second background music track using FFmpeg.
    Uses a mix of low sine waves to simulate lo-fi ambient music.
    """
    dest = MUSIC_DIR / "lofi_background.mp3"
    if dest.exists():
        print(f"   ✅ lofi_background.mp3 already exists, skipping")
        return

    print("[MUS] Generating lo-fi background music placeholder...")
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "aevalsrc='0.3*sin(2*PI*220*t)+0.2*sin(2*PI*330*t)+0.1*sin(2*PI*110*t)':s=44100:d=60",
        "-af", "afade=t=in:st=0:d=2,afade=t=out:st=58:d=2,volume=0.4",
        "-ar", "44100", "-ac", "2", "-b:a", "192k",
        str(dest)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and dest.exists():
            size_kb = dest.stat().st_size // 1024
            print(f"   [OK] Generated: lofi_background.mp3 ({size_kb} KB)")
        else:
            print(f"   [WARN] FFmpeg error: {result.stderr[-300:]}")
    except Exception as e:
        print(f"   [ERR] Error: {e}")


def download_montserrat_font():
    """Download Montserrat Bold font from GitHub."""
    dest = FONTS_DIR / "Montserrat-Bold.ttf"
    if dest.exists():
        print(f"   ✅ Montserrat-Bold.ttf already exists")
        return

    print("[FONT] Downloading Montserrat Bold font...")
    try:
        import urllib.request
        url = "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Bold.ttf"
        urllib.request.urlretrieve(url, dest)
        size_kb = dest.stat().st_size // 1024
        print(f"   [OK] Downloaded: Montserrat-Bold.ttf ({size_kb} KB)")
    except Exception as e:
        print(f"   [WARN] Font download failed: {e}")
        print("   [INFO] Pillow will use a fallback system font.")


def main():
    print("=" * 60)
    print("  SupoClip v2.0 - Asset Setup")
    print("=" * 60)

    # Create directories
    for d in [SFX_DIR, MUSIC_DIR, FONTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"\n[DIR] Asset directories ready: {ASSETS_DIR}")

    print("\n-- SFX ------------------------------------------------------")
    generate_sfx_with_ffmpeg()

    print("\n-- Background Music ------------------------------------------")
    generate_lofi_music()

    print("\n-- Fonts -----------------------------------------------------")
    download_montserrat_font()

    print("\n" + "=" * 60)
    print("  Asset setup complete!")
    print("=" * 60)

    # Summary
    sfx_files = list(SFX_DIR.glob("*.mp3"))
    music_files = list(MUSIC_DIR.glob("*.mp3"))
    font_files = list(FONTS_DIR.glob("*.ttf"))

    print(f"\nSummary:")
    print(f"   SFX files:   {len(sfx_files)} -> {[f.name for f in sfx_files]}")
    print(f"   Music files: {len(music_files)} -> {[f.name for f in music_files]}")
    print(f"   Fonts:       {len(font_files)} -> {[f.name for f in font_files]}")
    print()
    print("Pro tip: Replace the generated SFX with real ones from:")
    print("   -> https://freesound.org (search: whoosh, ding, sub bass)")
    print("   -> https://freemusicarchive.org (lo-fi, ambient)")
    print()


if __name__ == "__main__":
    main()
