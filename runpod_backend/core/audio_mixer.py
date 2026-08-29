"""
SupoClip Backend — Professional Audio Mixer
Handles background music overlay with auto-ducking, SFX injection,
and smooth crossfades at all audio boundaries.
"""

import logging
import os
import random
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

ASSETS_MUSIC_DIR = Path("/workspace/assets/music")
ASSETS_SFX_DIR = Path("/workspace/assets/sfx")

# Volume levels
MUSIC_NORMAL_VOLUME = 0.18       # 18% — soft background
MUSIC_DUCKED_VOLUME = 0.06       # 6% — ducked under speech (≈ -25% relative)
VOICE_DUCK_THRESHOLD = 0.015     # RMS threshold for detecting vocal activity
CROSSFADE_MS = 300               # 300ms crossfade between audio segments

# SFX file mapping
SFX_MAP = {
    "whoosh": "whoosh.mp3",
    "ding": "ding.mp3",
    "sub_bass": "sub_bass.mp3",
}


class AudioMixer:
    """
    Professional audio post-processing:
    1. Overlays soft background music
    2. Auto-ducks music when speech is detected
    3. Injects punchy SFX at AI-flagged timestamps
    4. Applies smooth crossfades at clip boundaries
    """

    def __init__(
        self,
        music_dir: Path = ASSETS_MUSIC_DIR,
        sfx_dir: Path = ASSETS_SFX_DIR,
    ):
        self.music_dir = Path(music_dir)
        self.sfx_dir = Path(sfx_dir)
        logger.info(
            f"AudioMixer initialized | music={self.music_dir} | sfx={self.sfx_dir}"
        )

    def process_clip(
        self,
        video_path: str | Path,
        output_path: str | Path,
        sfx_type: str = "sub_bass",
        sfx_offset_sec: float = 0.0,
        enable_music: bool = True,
    ) -> Path:
        """
        Apply full audio processing pipeline to a rendered clip.

        Args:
            video_path: Path to the input video (with original audio).
            output_path: Path for the audio-processed output video.
            sfx_type: Type of SFX to inject ("whoosh", "ding", "sub_bass").
            sfx_offset_sec: Timestamp in video where SFX should play (seconds from clip start).
            enable_music: Whether to add background music.

        Returns:
            Path to the output video with processed audio.
        """
        from moviepy.editor import (
            VideoFileClip,
            AudioFileClip,
            CompositeAudioClip,
        )
        from moviepy.audio.fx.all import audio_fadein, audio_fadeout, volumex

        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Audio processing: {video_path.name}")

        clip = VideoFileClip(str(video_path))
        clip_duration = clip.duration

        audio_tracks = []

        # ── 1. Original vocal audio (master track) ─────────────────────
        if clip.audio is not None:
            original_audio = clip.audio
            # Apply crossfade at boundaries
            original_audio = audio_fadein(original_audio, CROSSFADE_MS / 1000)
            original_audio = audio_fadeout(original_audio, CROSSFADE_MS / 1000)
            audio_tracks.append(original_audio)
            voice_audio = original_audio
        else:
            voice_audio = None
            logger.warning(f"Clip {video_path.name} has no audio track")

        # ── 2. Background Music with Auto-Ducking ──────────────────────
        if enable_music:
            music_track = self._get_music_track(clip_duration)
            if music_track is not None:
                if voice_audio is not None:
                    # Build ducked version
                    ducked_music = self._apply_ducking(
                        music_track,
                        voice_audio,
                        clip_duration,
                    )
                else:
                    ducked_music = volumex(music_track, MUSIC_NORMAL_VOLUME)

                ducked_music = audio_fadein(ducked_music, CROSSFADE_MS / 1000)
                ducked_music = audio_fadeout(ducked_music, CROSSFADE_MS / 1000)
                audio_tracks.append(ducked_music)

        # ── 3. SFX Injection ───────────────────────────────────────────
        sfx_track = self._load_sfx(sfx_type, sfx_offset_sec, clip_duration)
        if sfx_track is not None:
            audio_tracks.append(sfx_track)

        # ── 4. Mix all tracks ──────────────────────────────────────────
        if len(audio_tracks) > 1:
            final_audio = CompositeAudioClip(audio_tracks)
        elif len(audio_tracks) == 1:
            final_audio = audio_tracks[0]
        else:
            final_audio = None

        # ── 5. Write output ────────────────────────────────────────────
        final_clip = clip.set_audio(final_audio)
        final_clip.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            audio_bitrate="192k",
            fps=clip.fps,
            preset="fast",
            ffmpeg_params=["-crf", "18", "-movflags", "+faststart"],
            logger=None,
        )

        clip.close()
        final_clip.close()

        logger.info(f"Audio processing complete: {output_path.name}")
        return output_path

    def _get_music_track(self, duration: float):
        """Load a random music track from assets, looped to fit clip duration."""
        from moviepy.editor import AudioFileClip, afx
        from moviepy.audio.fx.all import audio_loop, volumex

        music_files = list(self.music_dir.glob("*.mp3")) + list(self.music_dir.glob("*.wav"))
        if not music_files:
            logger.warning(f"No music files found in {self.music_dir}")
            return None

        music_file = random.choice(music_files)
        logger.info(f"Using music track: {music_file.name}")

        music = AudioFileClip(str(music_file))

        # Loop if shorter than clip
        if music.duration < duration:
            loops = int(duration / music.duration) + 1
            music = audio_loop(music, nloops=loops)

        # Trim to clip duration
        music = music.subclip(0, duration)
        music = volumex(music, MUSIC_NORMAL_VOLUME)
        return music

    def _apply_ducking(self, music_track, voice_track, duration: float):
        """
        Reduce music volume during speech segments using librosa energy detection.
        Strategy: Analyze vocal RMS energy, generate a volume curve, apply to music.
        """
        from moviepy.audio.fx.all import volumex

        try:
            import librosa
            import numpy as np

            # Export voice to temp array for analysis
            voice_array = voice_track.to_soundarray(fps=16000)
            if voice_array.ndim > 1:
                voice_array = voice_array.mean(axis=1)  # mono

            # Compute frame-level RMS energy
            frame_length = 512
            hop_length = 256
            rms = librosa.feature.rms(
                y=voice_array,
                frame_length=frame_length,
                hop_length=hop_length,
            )[0]

            # Time axis of RMS frames
            rms_times = librosa.frames_to_time(
                np.arange(len(rms)),
                sr=16000,
                hop_length=hop_length,
            )

            # Build volume multiplier function for music track
            def get_volume_at_time(t):
                """Return music volume multiplier at time t."""
                idx = np.searchsorted(rms_times, t, side="right") - 1
                idx = np.clip(idx, 0, len(rms) - 1)
                energy = rms[idx]
                if energy > VOICE_DUCK_THRESHOLD:
                    return MUSIC_DUCKED_VOLUME      # Speech active — duck music
                else:
                    return MUSIC_NORMAL_VOLUME      # Silence — full music

            # Apply dynamic volume using MoviePy fl_time
            # Approximate by cutting into 0.5s segments with different volumes
            from moviepy.editor import concatenate_audioclips, AudioClip
            from moviepy.audio.fx.all import audio_fadein, audio_fadeout

            segment_dur = 0.5
            segments = []
            t = 0.0

            while t < duration:
                seg_end = min(t + segment_dur, duration)
                seg = music_track.subclip(t, seg_end)
                vol = get_volume_at_time(t + segment_dur / 2)
                seg = volumex(seg, vol / MUSIC_NORMAL_VOLUME)  # Normalize relative to base
                segments.append(seg)
                t = seg_end

            if segments:
                ducked = concatenate_audioclips(segments)
                ducked = audio_fadein(ducked, CROSSFADE_MS / 1000)
                ducked = audio_fadeout(ducked, CROSSFADE_MS / 1000)
                return ducked

        except Exception as e:
            logger.warning(f"Ducking failed ({e}), using flat volume")

        # Fallback: static low volume
        return volumex(music_track, MUSIC_DUCKED_VOLUME)

    def _load_sfx(
        self,
        sfx_type: str,
        offset_sec: float,
        clip_duration: float,
    ):
        """Load and position an SFX clip at the specified timestamp."""
        from moviepy.editor import AudioFileClip
        from moviepy.audio.fx.all import volumex

        sfx_filename = SFX_MAP.get(sfx_type, "sub_bass.mp3")
        sfx_path = self.sfx_dir / sfx_filename

        if not sfx_path.exists():
            logger.warning(f"SFX file not found: {sfx_path}")
            return None

        if offset_sec >= clip_duration:
            logger.warning(f"SFX offset {offset_sec:.1f}s >= clip duration {clip_duration:.1f}s, skipping")
            return None

        sfx = AudioFileClip(str(sfx_path))
        sfx = volumex(sfx, 0.7)  # 70% volume for SFX

        # Trim SFX to not exceed clip end
        max_sfx_duration = clip_duration - offset_sec
        if sfx.duration > max_sfx_duration:
            sfx = sfx.subclip(0, max_sfx_duration)

        # Position at offset
        sfx = sfx.set_start(offset_sec)

        logger.info(f"SFX '{sfx_type}' injected at {offset_sec:.2f}s")
        return sfx
