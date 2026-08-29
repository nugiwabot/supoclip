"""
SupoClip Backend — Whisper Transcription Module
Uses faster-whisper (CTranslate2-optimized) for GPU-accelerated speech-to-text
with word-level timestamps required for caption rendering.
"""

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Transcriber:
    """
    GPU-accelerated transcription using faster-whisper.
    
    Returns word-level timestamps in format:
    [{"word": "hello", "start": 0.0, "end": 0.4}, ...]
    """

    def __init__(self, model_size: str = "small", device: str = "auto"):
        from faster_whisper import WhisperModel

        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        compute_type = "float16" if device == "cuda" else "int8"

        logger.info(f"Loading Whisper model: {model_size} | device={device} | compute={compute_type}")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root="/workspace/models",
        )
        self.model_size = model_size
        self.device = device
        logger.info("Whisper model loaded successfully")

    def transcribe(
        self,
        audio_path: str | Path,
        language: str = None,
    ) -> dict[str, Any]:
        """
        Transcribe audio/video file with word-level timestamps.

        Args:
            audio_path: Path to audio or video file.
            language: Optional ISO language code (auto-detects if None).

        Returns:
            Dict with:
                - "text": Full transcript string
                - "language": Detected language code
                - "duration": Total audio duration in seconds
                - "words": List of {"word", "start", "end"} dicts
                - "segments": List of sentence segments with timestamps
        """
        audio_path = str(audio_path)
        logger.info(f"Transcribing: {Path(audio_path).name} | model={self.model_size}")

        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            beam_size=5,
            vad_filter=True,                # Remove silence
            vad_parameters={
                "min_silence_duration_ms": 500,
                "threshold": 0.5,
            },
        )

        all_words = []
        segment_list = []
        full_text_parts = []

        for segment in segments:
            seg_words = []
            if segment.words:
                for w in segment.words:
                    word_data = {
                        "word": w.word.strip(),
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(w.probability, 3),
                    }
                    all_words.append(word_data)
                    seg_words.append(word_data)

            segment_list.append({
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": segment.text.strip(),
                "words": seg_words,
            })
            full_text_parts.append(segment.text.strip())

        duration = info.duration
        full_text = " ".join(full_text_parts)

        # Build transcript with timestamps for AI Brain
        timestamped_lines = []
        for seg in segment_list:
            timestamped_lines.append(
                f"[{seg['start']:.1f}s → {seg['end']:.1f}s] {seg['text']}"
            )
        timestamped_transcript = "\n".join(timestamped_lines)

        logger.info(
            f"Transcription complete | duration={duration:.1f}s | "
            f"words={len(all_words)} | language={info.language}"
        )

        return {
            "text": full_text,
            "timestamped_transcript": timestamped_transcript,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": round(duration, 3),
            "words": all_words,
            "segments": segment_list,
        }

    def get_transcript_for_ai(self, transcription_result: dict) -> str:
        """
        Format transcript with timestamps for the AI Brain prompt.
        Includes segment-level timing for accurate clip boundary detection.
        """
        return transcription_result.get("timestamped_transcript", "")
