"""
SupoClip Backend — Video Editor
Handles clip extraction, intelligent 9:16 reframing, and smooth transitions.
Uses MoviePy with FFmpeg under the hood.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default output spec
TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
FADE_DURATION = 0.3    # seconds


class VideoEditor:
    """
    Extracts clips from source video and reframes them to 9:16 vertical format.
    Applies smooth fade transitions at clip boundaries.
    """

    def __init__(
        self,
        source_path: str | Path,
        face_tracker=None,
        output_resolution: tuple[int, int] = (TARGET_WIDTH, TARGET_HEIGHT),
    ):
        from moviepy.editor import VideoFileClip

        self.source_path = Path(source_path)
        self.face_tracker = face_tracker
        self.out_w, self.out_h = output_resolution
        self.fade_dur = FADE_DURATION

        logger.info(f"Loading source video: {self.source_path.name}")
        self.source = VideoFileClip(
            str(self.source_path),
            audio=True,
        )
        self.src_w = int(self.source.w)
        self.src_h = int(self.source.h)
        self.src_fps = self.source.fps
        self.src_duration = self.source.duration

        logger.info(
            f"Source: {self.src_w}×{self.src_h} @ {self.src_fps:.2f}fps | {self.src_duration:.1f}s"
        )

        # Face tracker reference (analyzed per clip on-demand for speed)
        self.face_data = []

    def extract_and_reframe_clip(
        self,
        clip_info: dict[str, Any],
        output_path: str | Path,
    ) -> Path:
        """
        Extract a clip segment and reframe it to vertical 9:16 format.

        Args:
            clip_info: Dict from AI Brain with start_time, end_time, etc.
            output_path: Destination path for the output .mp4 file.

        Returns:
            Path to the rendered clip.
        """
        from moviepy.editor import VideoFileClip, CompositeVideoClip, ColorClip
        from moviepy.video.fx.all import fadein, fadeout, resize

        start = float(clip_info["start_time"])
        end = float(clip_info["end_time"])
        output_path = Path(output_path)

        # Safety bounds
        start = max(0.0, start)
        end = min(self.src_duration, end)
        if end - start < 1.0:
            raise ValueError(f"Clip too short: {end - start:.2f}s")

        logger.info(
            f"Extracting clip {clip_info.get('clip_id')}: "
            f"{start:.2f}s → {end:.2f}s ({end-start:.1f}s)"
        )

        # Fast on-demand face tracking for this clip segment only
        clip_face_data = []
        if self.face_tracker is not None:
            clip_face_data = self.face_tracker.analyze_clip(
                self.source_path, start, end, self.src_w, self.src_h
            )

        # --- Step 1: Subclip ---
        raw_clip = self.source.subclip(start, end)

        # --- Step 2: Calculate 9:16 crop ---
        reframed = self._reframe_to_vertical(raw_clip, start, clip_face_data)

        # --- Step 3: Apply fade in/out transitions ---
        clip_with_fades = self._apply_fades(reframed)

        # --- Step 4: Write output ---
        output_path.parent.mkdir(parents=True, exist_ok=True)
        clip_with_fades.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            fps=self.src_fps,
            preset="fast",
            ffmpeg_params=[
                "-crf", "18",
                "-movflags", "+faststart",
                "-pix_fmt", "yuv420p",
            ],
            logger=None,  # Suppress MoviePy's verbose output
        )

        # Cleanup MoviePy clips
        clip_with_fades.close()
        raw_clip.close()
        reframed.close()

        logger.info(f"Clip rendered: {output_path.name} ({output_path.stat().st_size / 1e6:.1f} MB)")
        return output_path

    def _reframe_to_vertical(self, clip, start_time: float, face_data: list = None):
        """
        Crop and resize clip to target 9:16 vertical format.
        Uses face tracking data if available, else center-crop.
        """
        from moviepy.editor import VideoClip
        from moviepy.video.fx.all import resize, crop

        src_w = int(clip.w)
        src_h = int(clip.h)
        target_aspect = self.out_w / self.out_h  # 9/16 = 0.5625

        # Determine crop dimensions
        if src_w / src_h > target_aspect:
            # Wider than 9:16 — crop width
            crop_h = src_h
            crop_w = int(src_h * target_aspect)
        else:
            # Taller or equal — crop height
            crop_w = src_w
            crop_h = int(src_w / target_aspect)

        # Use face tracking or center
        if face_data and self.face_tracker:
            # Dynamic crop: face-centered per frame
            clip_reframed = clip.fl(
                lambda gf, t: self._make_frame_vertical(
                    gf(t), t + start_time, src_w, src_h, crop_w, crop_h, face_data
                ),
                apply_to=["mask"],
            )
        else:
            # Static center crop
            crop_x1 = (src_w - crop_w) // 2
            crop_y1 = (src_h - crop_h) // 2
            clip_reframed = crop(clip, x1=crop_x1, y1=crop_y1, width=crop_w, height=crop_h)

        # Resize to exact output dimensions
        clip_final = clip_reframed.resize((self.out_w, self.out_h))
        return clip_final

    def _make_frame_vertical(
        self,
        frame: np.ndarray,
        abs_time: float,
        src_w: int,
        src_h: int,
        crop_w: int,
        crop_h: int,
        face_data: list,
    ) -> np.ndarray:
        """Per-frame crop function for face-tracking mode."""
        # Get face-centered X position
        crop_x = self.face_tracker.get_crop_x_at_time(
            face_data, abs_time, src_w, crop_w
        )
        crop_y = (src_h - crop_h) // 2

        # Crop
        cropped = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

        # Fast resize to output dimensions
        import cv2
        resized = cv2.resize(cropped, (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)
        return resized

    def _apply_fades(self, clip):
        """Apply fade-in and fade-out to both video and audio."""
        from moviepy.video.fx.all import fadein, fadeout

        fade_dur = min(self.fade_dur, clip.duration / 4)

        # Video fades
        clip_faded = fadein(clip, fade_dur)
        clip_faded = fadeout(clip_faded, fade_dur)

        # Audio fades (if audio track exists)
        if clip_faded.audio is not None:
            clip_faded = clip_faded.audio_fadein(fade_dur).audio_fadeout(fade_dur)

        return clip_faded

    def close(self):
        """Release video resources."""
        try:
            self.source.close()
        except Exception:
            pass
