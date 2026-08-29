"""
SupoClip Backend — OpusClip / FujiwaraChoki-Style Kinetic Caption Renderer
Generates dynamic phrase-chunked subtitles with word-by-word karaoke highlighting,
and injects an eye-catching 0.06s Hook Thumbnail Card at t=0 for platform auto-cover.

Features:
- Dynamic phrase grouping (2-3 words per natural phrase)
- Real-time word-by-word active highlight (Neon Yellow / Neon Green) with white text & thick stroke
- High-impact keyword styling (Hormozi / OpusClip viral style)
- Flash Hook Thumbnail Frame at t=0 (0.06s) for TikTok / YouTube Shorts cover detection
- Safe zone positioning (y ~ 75%) to avoid platform UI overlays
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

NORMAL_FONT_SIZE = 68
IMPACT_FONT_SIZE = 78
THUMBNAIL_HOOK_SIZE = 82
BADGE_FONT_SIZE = 34

# Colors (RGBA)
COLOR_WHITE = (255, 255, 255, 255)
COLOR_YELLOW = (255, 230, 0, 255)       # Active word highlight
COLOR_NEON_GREEN = (0, 255, 163, 255)    # Impact word active
COLOR_CORAL_RED = (255, 46, 99, 255)     # Climax word active
COLOR_BLACK = (0, 0, 0, 255)
COLOR_SHADOW = (10, 10, 10, 220)
COLOR_BADGE_BG = (255, 230, 0, 240)
COLOR_BADGE_TEXT = (15, 15, 15, 255)

OUTLINE_WIDTH = 5
CAPTION_Y_PERCENT = 0.75  # Safe zone: lower third, above bottom navigation bar
WORD_DISPLAY_BUFFER = 0.08
THUMBNAIL_DURATION = 0.06  # 1-2 frames at t=0 (0.06s) for social platform cover detection


def _load_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    """Load best available bold font at given size."""
    font_names = [
        "Montserrat-Bold.ttf",
        "montserrat-bold.ttf",
        "Montserrat-Black.ttf",
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

    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def _draw_text_with_stroke(
    draw: ImageDraw.ImageDraw,
    text: str,
    position: tuple[int, int],
    font: ImageFont.FreeTypeFont,
    fill_color: tuple,
    stroke_color: tuple = COLOR_BLACK,
    stroke_width: int = OUTLINE_WIDTH,
    anchor: str = "mm",
) -> None:
    """Draw crisp text with a thick outline and drop shadow."""
    x, y = position
    # Draw drop shadow
    draw.text((x + 4, y + 5), text, font=font, fill=COLOR_SHADOW, anchor=anchor)

    # Draw outline
    for dx in range(-stroke_width, stroke_width + 1):
        for dy in range(-stroke_width, stroke_width + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=stroke_color, anchor=anchor)

    # Draw main text
    draw.text(position, text, font=font, fill=fill_color, anchor=anchor)


def _generate_thumbnail_cover_frame(
    video_w: int,
    video_h: int,
    hook_title: str,
) -> np.ndarray:
    """
    Generate an ultra-engaging 1080x1920 thumbnail frame for the first 0.06s.
    Platforms (TikTok, Reels, Shorts) automatically capture t=0 as the default cover.
    """
    img = Image.new("RGBA", (video_w, video_h), (12, 14, 20, 255))
    draw = ImageDraw.Draw(img)

    # Ambient gradient glow
    for r in range(video_w // 2, 0, -20):
        alpha = int(35 * (1 - r / (video_w // 2)))
        draw.ellipse(
            [video_w // 2 - r, video_h // 2 - r, video_w // 2 + r, video_h // 2 + r],
            fill=(255, 230, 0, alpha),
        )

    # Top Badge: "🔥 MUST WATCH"
    badge_font = _load_font(BADGE_FONT_SIZE)
    badge_text = "🔥  VIRAL HIGHLIGHT  🔥"
    badge_bbox = draw.textbbox((0, 0), badge_text, font=badge_font)
    bw = badge_bbox[2] - badge_bbox[0] + 48
    bh = badge_bbox[3] - badge_bbox[1] + 24
    bx = (video_w - bw) // 2
    by = int(video_h * 0.30)
    draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=16, fill=COLOR_BADGE_BG)
    draw.text((video_w // 2, by + bh // 2), badge_text, font=badge_font, fill=COLOR_BADGE_TEXT, anchor="mm")

    # Hook Title in Center
    hook_clean = hook_title.strip() or "TONTON SAMPAI SELESAI!"
    hook_lines = textwrap.wrap(hook_clean.upper(), width=18)
    hook_font = _load_font(THUMBNAIL_HOOK_SIZE)
    
    line_height = int(THUMBNAIL_HOOK_SIZE * 1.25)
    total_text_h = len(hook_lines) * line_height
    start_y = (video_h - total_text_h) // 2 + 30

    for i, line in enumerate(hook_lines):
        line_y = start_y + i * line_height
        # Alternate line colors: yellow and white
        color = COLOR_YELLOW if i % 2 == 0 else COLOR_WHITE
        _draw_text_with_stroke(
            draw,
            line,
            (video_w // 2, line_y),
            hook_font,
            fill_color=color,
            stroke_color=COLOR_BLACK,
            stroke_width=6,
        )

    # Bottom Callout
    bottom_font = _load_font(32)
    _draw_text_with_stroke(
        draw,
        "▶  TAP TO WATCH CLIP",
        (video_w // 2, int(video_h * 0.72)),
        bottom_font,
        fill_color=COLOR_WHITE,
        stroke_color=COLOR_BLACK,
        stroke_width=3,
    )

    return np.array(img)


def _group_words_into_phrases(
    words: list[dict],
    max_words_per_phrase: int = 3,
    max_pause_sec: float = 0.4,
) -> list[list[dict]]:
    """
    Group word timestamps into natural phrases (2-3 words).
    Breaks phrase on punctuation or pauses > max_pause_sec.
    """
    phrases = []
    current_phrase = []

    for word_info in words:
        word_text = word_info.get("word", "").strip()
        if not word_text:
            continue

        if not current_phrase:
            current_phrase.append(word_info)
            continue

        prev_word = current_phrase[-1]
        gap = word_info.get("start", 0) - prev_word.get("end", 0)
        has_punctuation = bool(prev_word.get("word", "").strip()[-1:] in ".!?,;:")

        if len(current_phrase) >= max_words_per_phrase or gap > max_pause_sec or has_punctuation:
            phrases.append(current_phrase)
            current_phrase = [word_info]
        else:
            current_phrase.append(word_info)

    if current_phrase:
        phrases.append(current_phrase)

    return phrases


def _render_karaoke_phrase_frame(
    video_w: int,
    video_h: int,
    phrase_words: list[dict],
    active_word_idx: int,
    impact_set: set[str],
) -> np.ndarray:
    """
    Render a phrase where all words are visible, but the active word is highlighted.
    This creates the modern OpusClip / CapCut Pro karaoke kinetic effect.
    """
    img = Image.new("RGBA", (video_w, video_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    font = _load_font(NORMAL_FONT_SIZE)
    space_width = draw.textlength(" ", font=font)

    # Measure total width of phrase
    word_widths = []
    for w_info in phrase_words:
        w_text = w_info.get("word", "").strip().upper()
        w_width = draw.textlength(w_text, font=font)
        word_widths.append((w_text, w_width))

    total_phrase_width = sum(w for _, w in word_widths) + space_width * (len(word_widths) - 1)
    start_x = (video_w - total_phrase_width) / 2.0
    center_y = int(video_h * CAPTION_Y_PERCENT)

    # Draw each word in phrase
    current_x = start_x
    for idx, (w_text, w_width) in enumerate(word_widths):
        is_active = (idx == active_word_idx)
        is_impact = (w_text.lower() in impact_set)

        word_center_x = int(current_x + w_width / 2.0)

        if is_active:
            # Active word: Highlighted in vibrant Neon Yellow or Neon Green
            highlight_color = COLOR_NEON_GREEN if is_impact else COLOR_YELLOW
            _draw_text_with_stroke(
                draw,
                w_text,
                (word_center_x, center_y),
                font,
                fill_color=highlight_color,
                stroke_color=COLOR_BLACK,
                stroke_width=OUTLINE_WIDTH + 1,
            )
        else:
            # Inactive word: Solid White with black stroke
            _draw_text_with_stroke(
                draw,
                w_text,
                (word_center_x, center_y),
                font,
                fill_color=COLOR_WHITE,
                stroke_color=COLOR_BLACK,
                stroke_width=OUTLINE_WIDTH,
            )

        current_x += w_width + space_width

    return np.array(img)


class CaptionRenderer:
    """
    Composites kinetic phrase-by-phrase karaoke subtitles & hook thumbnail cover.
    """

    def add_captions(
        self,
        video_path: str | Path,
        output_path: str | Path,
        word_timestamps: list[dict],
        high_impact_words: list[str],
        clip_start_time: float = 0.0,
        hook_title: str = "",
        impact_color_idx: int = 0,
    ) -> Path:
        """
        Add word-synced kinetic subtitles and a 0.06s hook thumbnail cover.
        """
        from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip

        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Adding captions to: {video_path.name} | hook='{hook_title}'")

        clip = VideoFileClip(str(video_path))
        video_w = int(clip.w)
        video_h = int(clip.h)
        clip_duration = clip.duration
        clip_end_time = clip_start_time + clip_duration

        impact_set = {w.lower().strip() for w in high_impact_words if w}

        # Filter words for this clip
        clip_words = []
        for w in word_timestamps:
            if w.get("end", 0) >= clip_start_time and w.get("start", 0) <= clip_end_time:
                clip_words.append({
                    "word": w.get("word", "").strip(),
                    "start": max(0.0, float(w.get("start", 0)) - clip_start_time),
                    "end": min(clip_duration, float(w.get("end", 0)) - clip_start_time + WORD_DISPLAY_BUFFER),
                })

        overlay_layers = []

        # ── 1. Flash Hook Thumbnail Frame at t=0 (0.06s) ──────────────────────
        if hook_title:
            try:
                thumb_arr = _generate_thumbnail_cover_frame(video_w, video_h, hook_title)
                thumb_clip = ImageClip(thumb_arr, ismask=False)
                thumb_clip = thumb_clip.set_start(0.0).set_end(THUMBNAIL_DURATION).set_opacity(1.0)
                overlay_layers.append(thumb_clip)
                logger.info(f"Generated 0.06s hook thumbnail cover for '{hook_title}'")
            except Exception as e:
                logger.warning(f"Thumbnail frame generation failed ({e}), skipping cover.")

        # ── 2. Phrase-Grouped Karaoke Subtitles ──────────────────────────────
        phrases = _group_words_into_phrases(clip_words, max_words_per_phrase=3)
        logger.info(f"Generated {len(phrases)} phrase groups for karaoke subtitles")

        for phrase in phrases:
            if not phrase:
                continue

            for active_idx, active_word in enumerate(phrase):
                w_start = active_word["start"]
                w_end = active_word["end"]
                if w_end <= w_start:
                    continue

                caption_arr = _render_karaoke_phrase_frame(
                    video_w,
                    video_h,
                    phrase_words=phrase,
                    active_word_idx=active_idx,
                    impact_set=impact_set,
                )

                caption_img = ImageClip(caption_arr, ismask=False)
                caption_img = caption_img.set_start(w_start).set_end(w_end).set_opacity(1.0)
                overlay_layers.append(caption_img)

        # Composite video layers
        all_layers = [clip] + overlay_layers
        final = CompositeVideoClip(all_layers, size=(video_w, video_h))
        final = final.set_audio(clip.audio)

        logger.info(f"Rendering captioned clip with {len(overlay_layers)} dynamic visual layers...")
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

        logger.info(f"Captioned output successfully written: {output_path.name}")
        return output_path

