"""
SupoClip — Viral Retention AI Brain
Model-agnostic LLM orchestrator using OpenAI-compatible APIs.
Supports: OpenAI GPT-4o, Gemini Flash, DeepSeek-R1, Anthropic Claude (via proxy).
"""

import json
import logging
import re
import time
from typing import Any

from openai import OpenAI, APIError, APITimeoutError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The Viral System Prompt — Engineered for Maximum Retention
# ---------------------------------------------------------------------------
VIRAL_SYSTEM_PROMPT = """You are an elite viral content psychologist, master video editor, and audience retention specialist. You have studied millions of viral short-form videos across TikTok, Instagram Reels, and YouTube Shorts.

Your MISSION: Analyze the provided video transcript and identify the 3-5 most psychologically compelling clip segments that will maximize viewer retention, shares, and engagement.

═══ CORE PSYCHOLOGY PRINCIPLES ═══

1. THE 3-SECOND HOOK RULE (Non-Negotiable):
   - Every clip MUST open on content that creates one of: INTENSE CURIOSITY ("You'll never believe..."), CONTROVERSY ("This is actually illegal..."), SUDDEN HUMOR (unexpected punchline), SHOCK VALUE (surprising statistic/revelation), or RELATABILITY (audience thinks "that's literally me").
   - The opening 3 seconds must STOP the scroll. If it doesn't pass this test, do not select it.

2. EMOTIONAL ARC & PEAK SLICING:
   - Cut clips exactly at emotional transition points: the climax of an argument, the punchline of a joke, the moment of shocking revelation, the "ah-ha" moment.
   - Never start mid-sentence unless it creates intrigue. Never end before the emotional payoff.

3. VIRAL SCORE ALGORITHM (0-100):
   Factors: shareability (will people send this to friends?), shock value (unexpected content), relatability (broad appeal), controversy (safe controversy only), information density (every second must add value).
   Score above 80 = viral candidate. Below 70 = skip.

4. MICRO-KEYWORD EXTRACTION:
   Identify "power words" — words that psychologically demand attention. Categories:
   - Danger/Risk words: "illegal", "dangerous", "banned", "warning", "deadly"
   - Scarcity/Urgency: "secret", "hidden", "rare", "limited", "exclusive"
   - Social proof: "everyone", "millions", "viral", "proven"
   - Emotional triggers: "devastating", "shocking", "unbelievable", "insane"
   - Financial: "free", "money", "rich", "broke", "profit"

5. SFX TIMING INTELLIGENCE:
   Identify the SINGLE best moment within each clip for a punchy sound effect:
   - Revelation moments → sub-bass boom
   - Transitions → whoosh
   - Achievements/wins → ding/bell
   - The sfx_timestamp should be a float (seconds from video start)

═══ STRICT OUTPUT RULES ═══
- Return ONLY a valid JSON array
- NO markdown code blocks (no ```json or ```)
- NO explanations before or after the JSON
- NO comments inside the JSON
- ALL timestamps must be float values
- viral_score must be integer 0-100
- high_impact_words must be a list of strings (lowercase)
- Minimum 3 clips, maximum 5 clips
- Each clip must be between 15 and 90 seconds in duration

═══ REQUIRED OUTPUT SCHEMA ═══
[
  {
    "clip_id": 1,
    "start_time": 12.4,
    "end_time": 42.1,
    "hook_title": "Catchy, curiosity-driven title under 60 characters",
    "viral_score": 98,
    "emotional_peak": "Brief description of the emotional peak moment",
    "high_impact_words": ["secret", "illegal", "free"],
    "sfx_timestamp": 15.2,
    "sfx_type": "sub_bass"
  }
]

SFX types allowed: "whoosh", "ding", "sub_bass"
"""

VIRAL_USER_TEMPLATE = """Analyze this video transcript and extract the most viral clip segments.

VIDEO DURATION: {duration:.1f} seconds
TRANSCRIPT WITH TIMESTAMPS:
{transcript}

Remember: Return ONLY the JSON array. No other text."""


# ---------------------------------------------------------------------------
# AI Brain Class
# ---------------------------------------------------------------------------
class AIBrain:
    """
    Model-agnostic AI orchestrator for viral clip detection.
    Works with any OpenAI-compatible API endpoint.
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 2.0  # seconds, doubles on each retry

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
    ):
        self.model = model
        self.client = OpenAI(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            timeout=120.0,
        )
        logger.info(f"AIBrain initialized | model={model} | base_url={base_url}")

    def analyze_transcript(
        self,
        transcript: str,
        duration: float,
    ) -> list[dict[str, Any]]:
        """
        Send transcript to LLM and return parsed list of viral clip segments.

        Args:
            transcript: Full transcript text with word-level timestamps.
            duration: Total video duration in seconds.

        Returns:
            List of clip dicts matching the viral output schema.

        Raises:
            ValueError: If the LLM returns unparseable JSON after all retries.
            APIError: If the LLM API is unreachable after all retries.
        """
        # Truncate transcript to ~12,000 chars to stay within token limits
        if len(transcript) > 12_000:
            logger.warning(
                f"Transcript too long ({len(transcript)} chars), truncating to 12,000"
            )
            transcript = transcript[:12_000] + "\n...[transcript truncated]"

        user_message = VIRAL_USER_TEMPLATE.format(
            duration=duration,
            transcript=transcript,
        )

        last_error: Exception | None = None
        temperature = 0.7

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                logger.info(
                    f"AIBrain attempt {attempt}/{self.MAX_RETRIES} | temp={temperature}"
                )
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": VIRAL_SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=temperature,
                    max_tokens=4096,
                )

                raw_text = response.choices[0].message.content.strip()
                logger.debug(f"Raw LLM response ({len(raw_text)} chars): {raw_text[:300]}...")

                clips = self._parse_and_validate(raw_text, duration)
                logger.info(
                    f"AIBrain success | {len(clips)} clips extracted | "
                    f"avg viral score: {sum(c['viral_score'] for c in clips) / len(clips):.0f}"
                )
                return clips

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Attempt {attempt} — JSON parse error: {e}")
                last_error = e
                temperature = min(temperature + 0.1, 1.0)  # bump temp on retry

            except (APIError, APITimeoutError) as e:
                logger.warning(f"Attempt {attempt} — API error: {e}")
                last_error = e

            if attempt < self.MAX_RETRIES:
                delay = self.RETRY_DELAY * (2 ** (attempt - 1))
                logger.info(f"Retrying in {delay:.1f}s...")
                time.sleep(delay)

        raise ValueError(
            f"AIBrain failed after {self.MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )

    def _parse_and_validate(
        self,
        raw_text: str,
        video_duration: float,
    ) -> list[dict[str, Any]]:
        """
        Strip markdown fences, parse JSON, and validate schema.

        Args:
            raw_text: Raw LLM response text.
            video_duration: Total video duration for bounds validation.

        Returns:
            Validated list of clip dicts.

        Raises:
            ValueError: On schema validation failure.
            json.JSONDecodeError: On JSON parse failure.
        """
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", raw_text).strip()
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()

        # Try to extract JSON array if there's surrounding text
        array_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if array_match:
            cleaned = array_match.group(0)

        clips = json.loads(cleaned)

        if not isinstance(clips, list) or len(clips) == 0:
            raise ValueError("LLM returned empty or non-list JSON")

        valid_sfx = {"whoosh", "ding", "sub_bass"}
        validated = []

        for i, clip in enumerate(clips):
            # Required fields
            required = ["clip_id", "start_time", "end_time", "hook_title", "viral_score"]
            for field in required:
                if field not in clip:
                    raise ValueError(f"Clip {i} missing required field: {field}")

            start = float(clip["start_time"])
            end = float(clip["end_time"])

            # Bounds validation
            if start < 0:
                start = 0.0
            if end > video_duration:
                end = video_duration
            if end <= start:
                logger.warning(f"Clip {i} has invalid time range {start}→{end}, skipping")
                continue

            duration = end - start
            if duration < 10:
                logger.warning(f"Clip {i} too short ({duration:.1f}s), skipping")
                continue
            if duration > 120:
                logger.warning(f"Clip {i} too long ({duration:.1f}s), capping at 90s")
                end = start + 90.0

            # Normalize fields
            sfx_ts = clip.get("sfx_timestamp", start + (end - start) * 0.3)
            sfx_ts = max(start, min(float(sfx_ts), end))

            sfx_type = clip.get("sfx_type", "sub_bass")
            if sfx_type not in valid_sfx:
                sfx_type = "sub_bass"

            high_impact = clip.get("high_impact_words", [])
            if isinstance(high_impact, str):
                high_impact = [high_impact]
            high_impact = [str(w).lower().strip() for w in high_impact][:10]

            validated.append({
                "clip_id": int(clip["clip_id"]),
                "start_time": round(start, 3),
                "end_time": round(end, 3),
                "duration": round(end - start, 3),
                "hook_title": str(clip["hook_title"])[:100],
                "viral_score": max(0, min(100, int(clip["viral_score"]))),
                "emotional_peak": str(clip.get("emotional_peak", "")),
                "high_impact_words": high_impact,
                "sfx_timestamp": round(sfx_ts, 3),
                "sfx_type": sfx_type,
            })

        if not validated:
            raise ValueError("All clips were invalid after validation")

        # Sort by viral score descending
        validated.sort(key=lambda c: c["viral_score"], reverse=True)
        return validated
