"""
SupoClip Backend — MediaPipe Face Tracker
Provides intelligent speaker-centered crop coordinates for 9:16 reframing.
Falls back to center-crop when no face detected.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Sample rate: analyze 1 frame every N seconds
SAMPLE_INTERVAL_SEC = 1.0
# Smoothing window for jitter reduction
SMOOTHING_WINDOW = 5


class FaceTracker:
    """
    Detects face positions in video frames using MediaPipe.
    Returns smooth time-series of face center X positions for smart cropping.
    """

    def __init__(self):
        try:
            import mediapipe.solutions.face_detection as mp_face_detection
            self.mp_face_detection = mp_face_detection
            self.detector = self.mp_face_detection.FaceDetection(
                model_selection=1,          # 1 = full-range model (better for video)
                min_detection_confidence=0.5,
            )
            self.available = True
            logger.info("MediaPipe face detector initialized")
        except BaseException as e:
            logger.warning(f"MediaPipe not available ({e}) — using center-crop fallback")
            self.available = False
            self.detector = None

    def analyze_video(
        self,
        video_path: str | Path,
        video_width: int,
        video_height: int,
    ) -> list[dict]:
        """
        Sample video frames and detect face positions.

        Args:
            video_path: Path to the source video.
            video_width: Native video width in pixels.
            video_height: Native video height in pixels.

        Returns:
            List of {"time": float, "center_x": int, "confidence": float} dicts.
            center_x is in pixels from left edge.
        """
        if not self.available:
            return self._center_fallback(video_width)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return self._center_fallback(video_width)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        sample_step = max(1, int(fps * SAMPLE_INTERVAL_SEC))

        results = []
        frame_idx = 0

        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps
            cx = self._detect_face_center_x(frame, video_width)
            results.append({
                "time": round(timestamp, 3),
                "center_x": cx,
            })
            frame_idx += sample_step

        cap.release()
        logger.info(
            f"Face analysis complete | {len(results)} samples | duration={duration:.1f}s"
        )

        if not results:
            return self._center_fallback(video_width)

        return self._smooth_results(results, video_width)

    def _detect_face_center_x(self, frame: np.ndarray, video_width: int) -> int:
        """Detect the center X of the primary (largest/most confident) face."""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection_result = self.detector.process(rgb_frame)

            if not detection_result.detections:
                return video_width // 2

            # Pick the highest-confidence detection
            best = max(
                detection_result.detections,
                key=lambda d: d.score[0],
            )
            bbox = best.location_data.relative_bounding_box
            # Center X of the face bounding box
            face_cx_rel = bbox.xmin + bbox.width / 2
            face_cx_px = int(face_cx_rel * video_width)
            return max(0, min(video_width, face_cx_px))

        except Exception as e:
            logger.debug(f"Face detection error on frame: {e}")
            return video_width // 2

    def _smooth_results(
        self,
        results: list[dict],
        video_width: int,
    ) -> list[dict]:
        """Apply rolling average to reduce jittery crop movement."""
        if len(results) < SMOOTHING_WINDOW:
            return results

        cx_values = [r["center_x"] for r in results]
        smoothed = []

        for i, r in enumerate(results):
            start = max(0, i - SMOOTHING_WINDOW // 2)
            end = min(len(cx_values), i + SMOOTHING_WINDOW // 2 + 1)
            window = cx_values[start:end]
            # Clamp center to valid crop region
            avg_cx = int(np.mean(window))
            smoothed.append({**r, "center_x": avg_cx})

        return smoothed

    def _center_fallback(self, video_width: int) -> list[dict]:
        """Return a single center-crop data point as fallback."""
        return [{"time": 0.0, "center_x": video_width // 2}]

    def get_crop_x_at_time(
        self,
        face_data: list[dict],
        timestamp: float,
        video_width: int,
        crop_width: int,
    ) -> int:
        """
        Get the crop X origin (left edge) for a given timestamp.

        Args:
            face_data: Result from analyze_video().
            timestamp: Time in seconds to query.
            video_width: Source video width.
            crop_width: Width of the crop window.

        Returns:
            X pixel position of the crop's left edge.
        """
        if not face_data:
            return (video_width - crop_width) // 2

        # Find nearest sample
        nearest = min(face_data, key=lambda d: abs(d["time"] - timestamp))
        cx = nearest["center_x"]

        # Center crop around face
        crop_x = cx - crop_width // 2
        # Clamp to valid range
        crop_x = max(0, min(video_width - crop_width, crop_x))
        return crop_x

    def close(self):
        """Release MediaPipe resources."""
        if self.available and self.detector:
            self.detector.close()
