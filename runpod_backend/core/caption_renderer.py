"""
SupoClip Backend — CapCut Pro-Style Kinetic Caption Renderer
Generates word-by-word animated subtitles with impact word visual effects.

Features:
- Word-by-word sync from Whisper timestamps
- Dual-tone styling (white/yellow alternating)
- High-impact words: 1.5x scale, neon color, centered (CapCut style)
- Thick black drop-shadow outline for readability on any background
- Montserrat Bold or Impact font rendering via Pillow
"""

import logging
import os
import textwrap
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# ── Typography Constants ────────────────────────────────────────────────────
FONT_DIR = Path("/workspace/assets/fonts")
FONT_FALLBACK_DIR = Path(__file__).parent.parent / "assets" / "fonts"

NORMAL_FONT_SIZE = 62
IMPACT_FONT_SIZE = int(NORMAL_FONT_SIZE * 1.5)   # 93px for high-impact words

# Colors (RGBA)
COLOR_WHITE = (255, 255, 255, 255)
COLOR_YELLOW = (255, 230, 0, 255)
COLOR_NEON_GREEN = (57, 255, 20, 255)
COLOR_BLOOD_RED = (255, 0, 51, 255)
COLOR_BLACK = (0, 0, 0, 255)
COLOR_SHADOW = (20, 20, 20, 200)

# Outline thickness (pixels)
OUTLINE_WIDTH = 4

# Caption vertical position (% from top)
CAPTION_Y_PERCENT = 0.78

# How long to show a word after its end timestamp (in seconds)
WORD_DISPLAY_BUFFER = 0.1

# Impact word highlight colors (alternating between clips)
IMPACT_COLORS = [COLOR_NEON_GREEN, COLOR_BLOOD_RED]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Load best available font at given size."""
    font_names = [
        "Montserrat-Bold.ttf",
        "montserrat-bold.ttf",
        "Impact.ttf",
        "impact.ttf",
        "DejaVuSans-Bold.ttf",
    ]

    for font_dir in [FONT_DIR, FONT_FALLBACK_DIR]:
        for name in font_names:
            path = font_dir / name
            if path.exists():
                try:
                    return ImageFont.truetype(str(path), size)
                except Exception:
                    continue

    # Ultimate fallback: PIL default
    logger.warning("No custom font found, using PIL default")
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def _draw_text_with_outline(
    draw: ImageDraw.ImageDraw,
    text: str,
    position: tuple[int, int],
    font: ImageFont.FreeTypeFont,
    fill_color: tuple,
    outline_color: tuple = COLOR_BLACK,
    outline_width: int = OUTLINE_WIDTH,
) -> None:
    """Draw text with thick outline (for readability on any background)."""
    x, y = position

    # Draw outline by offsetting in all directions
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx != 0 or dy != 0:
                draw.text(
                    (x + dx, y + dy),
                    text,
                    font=font,
                    fill=outline_color,
                    anchor="mm",
                )

    # Draw main text on top
    draw.text(position, text, font=font, fill=fill_color, anchor="mm")


def _make_caption_frame(
    video_width: int,
    video_height: int,
    word: str,
    is_impact: bool,
    impact_color_idx: int = 0,
) -> np.ndarray:
    """
    Generate a single caption frame as RGBA numpy array.

    Args:
        video_width: Frame width in pixels.
        video_height: Frame height in pixels.
        word: The word to display.
        is_impact: Whether this is a high-impact keyword.
        impact_color_idx: Index into IMPACT_COLORS for this clip.

    Returns:
        RGBA numpy array of shape (video_height, video_width, 4).
    """
    img = Image.new("RGBA", (video_width, video_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    if is_impact:
        font = _load_font(IMPACT_FONT_SIZE)
        color = IMPACT_COLORS[impact_color_idx % len(IMPACT_COLORS)]
        # Center position for impact words
        x = video_width // 2
        y = video_height // 2  # Center screen for max visual impact
    else:
        font = _load_font(NORMAL_FONT_SIZE)
        # Alternate white/yellow for non-impact words
        color = COLOR_WHITE
        x = video_width // 2
        y = int(video_height * CAPTION_Y_PERCENT)

    # Draw shadow slightly offset
    shadow_x = x + 3
    shadow_y = y + 3
    draw.text((shadow_x, shadow_y), word, font=font, fill=COLOR_SHADOW, anchor="mm")

    # Draw outlined text
    _draw_text_with_outline(
        draw,
        word.upper() if is_impact else word,
        (x, y),
        font,
        color,
        COLOR_BLACK,
        OUTLINE_WIDTH + (2 if is_impact else 0),
    )

    return np.array(img)


class CaptionRenderer:
    """
    Composites kinetic word captions onto a video clip.

    Usage:
        renderer = CaptionRenderer()
        output_path = renderer.add_captions(
            video_path, output_path, word_timestamps, high_impact_words
        )
    """

    def add_captions(
        self,
        video_path: str | Path,
        output_path: str | Path,
        word_timestamps: list[dict],
        high_impact_words: list[str],
        clip_start_time: float = 0.0,
        impact_color_idx: int = 0,
    ) -> Path:
        """
        Add word-by-word captions to a video clip.

        Args:
            video_path: Path to input video.
            output_path: Path for captioned output video.
            word_timestamps: List of {"word", "start", "end"} from Whisper.
            high_impact_words: Lowercase list of high-impact keywords from AI Brain.
            clip_start_time: Absolute video time of the clip's start (for timestamp filtering).
            impact_color_idx: Determines neon green vs blood red for impact words.

        Returns:
            Path to the output video with captions burned in.
        """
        from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip

        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Adding captions to: {video_path.name}")

        clip = VideoFileClip(str(video_path))
        video_w = int(clip.w)
        video_h = int(clip.h)
        clip_duration = clip.duration

        # Filter word timestamps to only include words within this clip's time range
        clip_end_time = clip_start_time + clip_duration
        impact_set = {w.lower() for w in high_impact_words}

        # Build caption clips for each word
        caption_clips = []
        prev_color = False  # Alternate colors for normal words

        for word_data in word_timestamps:
            word_start_abs = word_data["start"]
            word_end_abs = word_data["end"] + WORD_DISPLAY_BUFFER

            # Skip words outside this clip's time range
            if word_end_abs < clip_start_time or word_start_abs > clip_end_time:
                continue

            # Convert absolute time to clip-relative time
            word_start_rel = max(0.0, word_start_abs - clip_start_time)
            word_end_rel = min(clip_duration, word_end_abs - clip_start_time)

            if word_end_rel <= word_start_rel:
                continue

            word_text = word_data["word"].strip()
            if not word_text:
                continue

            is_impact = word_text.lower() in impact_set

            # Generate the caption frame image
            caption_array = _make_caption_frame(
                video_w, video_h, word_text, is_impact, impact_color_idx
            )

            # Create a MoviePy ImageClip from the RGBA array
            caption_img = ImageClip(caption_array, ismask=False)
            caption_img = caption_img.set_start(word_start_rel)
            caption_img = caption_img.set_end(word_end_rel)
            caption_img = caption_img.set_opacity(1.0)

            caption_clips.append(caption_img)
            prev_color = not prev_color

        if not caption_clips:
            logger.warning(f"No caption clips generated for {video_path.name}")
            # Output unchanged
            clip.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac",
                fps=clip.fps,
                preset="fast",
                ffmpeg_params=["-crf", "18", "-movflags", "+faststart"],
                logger=None,
            )
            clip.close()
            return output_path

        logger.info(f"Compositing {len(caption_clips)} word captions")

        # Composite all caption layers on top of video
        all_layers = [clip] + caption_clips
        final = CompositeVideoClip(all_layers, size=(video_w, video_h))
        final = final.set_audio(clip.audio)

        final.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            fps=clip.fps,
            preset="fast",
            ffmpeg_params=["-crf", "18", "-movflags", "+faststart", "-pix_fmt", "yuv420p"],
            logger=None,
        )

        clip.close()
        final.close()

        logger.info(f"Captioned output written: {output_path.name}")
        return output_path

    def filter_words_for_clip(
        self,
        all_words: list[dict],
        clip_start: float,
        clip_end: float,
    ) -> list[dict]:
        """Return only words that fall within the clip's time range."""
        return [
            w for w in all_words
            if w["start"] < clip_end and w["end"] > clip_start
        ]
