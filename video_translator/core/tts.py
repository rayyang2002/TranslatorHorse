"""TTS engine abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import time
import wave
from typing import Optional

from utils.logger import logger


TTS_SEGMENT_RETRIES = 3
SILENCE_SAMPLE_RATE = 44100
SILENCE_CHANNELS = 2
SILENCE_SAMPLE_WIDTH = 2


class TTSEngine(ABC):
    """Base class for TTS engines."""

    @abstractmethod
    def generate(
        self,
        text: str,
        output_path: str | Path,
        voice: Optional[str] = None,
        speed: float = 1.0,
        **kwargs,
    ) -> Path:
        """Generate audio for a text segment."""
        raise NotImplementedError

    @abstractmethod
    def get_voices(self) -> list[str]:
        """Return available voices."""
        raise NotImplementedError

    @abstractmethod
    def get_supported_languages(self) -> list[str]:
        """Return supported language codes."""
        raise NotImplementedError

    @classmethod
    def create(cls, engine_name: str = "edge-tts", **kwargs) -> "TTSEngine":
        """Create a concrete TTS engine instance."""
        if engine_name == "edge-tts":
            from .tts_edge import EdgeTTSEngine

            return EdgeTTSEngine()
        if engine_name == "indextts2":
            from .tts_indextts2 import IndexTTS2Engine

            return IndexTTS2Engine(**kwargs)
        raise ValueError(f"Unsupported TTS engine: {engine_name}")


def generate_subtitle_audio(
    segments: list[dict],
    output_path: str | Path,
    engine: TTSEngine,
    voice: Optional[str] = None,
    speed: float = 1.0,
    reference_audio: Optional[str] = None,
    emotion: str = "natural",
    timing_strategy: str = "natural_with_light_stretch",
    cancel_check: Callable[[], None] | None = None,
    target_lang: Optional[str] = None,
) -> list[tuple[dict, Path]]:
    """Generate one audio file per subtitle segment.

    When ``target_lang`` is provided, each segment is synthesized with a per-
    segment speed hint (Phase 1 of the A-tier aligner) so the TTS engine speaks
    slightly faster/slower at synthesis time rather than being brutally
    compressed later. ``speed`` is kept as a global user multiplier and combined
    with the hint.
    """
    from core.tts_aligner import estimate_speech_duration, plan_tts_speed

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    audio_files: list[tuple[dict, Path]] = []
    for index, segment in enumerate(segments):
        if cancel_check:
            cancel_check()
        text = (segment.get("translated_text") or segment.get("text") or "").strip()
        if not text:
            continue

        suffix = ".mp3" if engine.__class__.__name__ == "EdgeTTSEngine" else ".wav"
        segment_path = output_path / f"seg_{index:04d}{suffix}"
        silence_path = output_path / f"seg_{index:04d}_silence.wav"
        metadata_path = output_path / f"seg_{index:04d}.json"
        start = float(segment.get("start", 0))
        end = float(segment.get("end", start))
        target_duration = max(0.0, end - start)

        # Phase 1: derive a synthesis-time speed hint from the slot vs. the
        # estimated natural duration. Stays 1.0 when target_lang is unknown
        # (keeps legacy callers/tests untouched).
        tts_speed = speed
        if target_lang and target_duration > 0:
            estimated = estimate_speech_duration(text, target_lang)
            hint, _reason = plan_tts_speed(target_duration, estimated)
            tts_speed = max(0.5, min(2.0, speed * hint))

        expected_metadata = _build_segment_metadata(
            engine=engine,
            text=text,
            start=start,
            end=end,
            voice=voice,
            speed=tts_speed,
            reference_audio=reference_audio,
            emotion=emotion,
            timing_strategy=timing_strategy,
            target_lang=target_lang,
            target_duration=target_duration,
        )

        cached_audio = _matching_cached_audio(
            metadata_path=metadata_path,
            expected_metadata=expected_metadata,
            segment_path=segment_path,
            silence_path=silence_path,
        )
        if cached_audio:
            segment_path = cached_audio
            logger.info("TTS segment exists, skipping: %s", segment_path.name)
        else:
            _remove_stale_segment_files(segment_path, silence_path, metadata_path)
            try:
                _generate_with_retries(
                    engine=engine,
                    text=text,
                    segment_path=segment_path,
                    voice=voice,
                    speed=tts_speed,
                    reference_audio=reference_audio,
                    emotion=emotion,
                    target_duration=target_duration,
                    timing_strategy=timing_strategy,
                    cancel_check=cancel_check,
                )
                _write_segment_metadata(
                    metadata_path,
                    expected_metadata,
                    output_file=segment_path.name,
                    fallback="none",
                )
            except Exception as exc:
                _write_silence_wav(silence_path, max(target_duration, 0.1))
                segment_path = silence_path
                _write_segment_metadata(
                    metadata_path,
                    expected_metadata,
                    output_file=silence_path.name,
                    fallback="silence",
                )
                logger.warning(
                    "TTS segment %s failed after %s attempts; using silence. Error: %s",
                    index,
                    TTS_SEGMENT_RETRIES,
                    exc,
                )
        if cancel_check:
            cancel_check()
        audio_files.append((segment, segment_path))

    logger.info(f"TTS generated {len(audio_files)} subtitle segments")
    return audio_files


def _generate_with_retries(
    engine: TTSEngine,
    text: str,
    segment_path: Path,
    voice: Optional[str],
    speed: float,
    reference_audio: Optional[str],
    emotion: str,
    target_duration: float,
    timing_strategy: str,
    cancel_check: Callable[[], None] | None = None,
) -> Path:
    last_error: Exception | None = None
    for attempt in range(1, TTS_SEGMENT_RETRIES + 1):
        if cancel_check:
            cancel_check()
        _remove_empty_or_partial_file(segment_path)
        try:
            engine.generate(
                text,
                segment_path,
                voice=voice,
                speed=speed,
                reference_audio=reference_audio,
                emotion=emotion,
                target_duration=target_duration,
                timing_strategy=timing_strategy,
            )
            if not segment_path.exists() or segment_path.stat().st_size == 0:
                raise RuntimeError("TTS generated an empty audio file.")
            return segment_path
        except Exception as exc:
            last_error = exc
            _remove_empty_or_partial_file(segment_path)
            if attempt < TTS_SEGMENT_RETRIES:
                logger.warning("TTS segment failed, retrying %s/%s: %s", attempt, TTS_SEGMENT_RETRIES, exc)
                time.sleep(min(attempt, 3))
    raise RuntimeError(str(last_error) if last_error else "TTS segment failed.")


def _remove_empty_or_partial_file(path: Path) -> None:
    if path.exists():
        try:
            path.unlink()
        except OSError:
            logger.warning("Could not remove partial TTS file: %s", path)


def _valid_audio_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _build_segment_metadata(
    *,
    engine: TTSEngine,
    text: str,
    start: float,
    end: float,
    voice: Optional[str],
    speed: float,
    reference_audio: Optional[str],
    emotion: str,
    timing_strategy: str,
    target_lang: Optional[str],
    target_duration: float,
) -> dict:
    """Build the cache identity for one synthesized subtitle segment.

    The cache intentionally stores a hash of the spoken text instead of the
    text itself. Segment files are only reused when every timing and synthesis
    parameter matches, preventing old ``seg_0000.mp3`` files from a previous
    video being mixed into a new video's timeline.
    """
    return {
        "schema": 1,
        "engine": engine.__class__.__name__,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "start_ms": round(float(start) * 1000),
        "end_ms": round(float(end) * 1000),
        "target_duration_ms": round(float(target_duration) * 1000),
        "voice": voice or "",
        "speed": round(float(speed), 4),
        "reference_audio": _reference_audio_cache_key(reference_audio),
        "emotion": emotion or "",
        "timing_strategy": timing_strategy or "",
        "target_lang": target_lang or "",
    }


def _reference_audio_cache_key(reference_audio: Optional[str]) -> str:
    if not reference_audio:
        return ""
    path = Path(reference_audio)
    try:
        stat = path.stat()
        identity = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    except OSError:
        identity = str(reference_audio)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _matching_cached_audio(
    *,
    metadata_path: Path,
    expected_metadata: dict,
    segment_path: Path,
    silence_path: Path,
) -> Path | None:
    if not metadata_path.exists():
        return None
    try:
        cached = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    metadata = cached.get("metadata")
    if metadata != expected_metadata:
        return None

    output_file = cached.get("output_file", "")
    candidates = {
        segment_path.name: segment_path,
        silence_path.name: silence_path,
    }
    cached_path = candidates.get(output_file)
    if cached_path and _valid_audio_file(cached_path):
        return cached_path
    return None


def _write_segment_metadata(
    metadata_path: Path,
    metadata: dict,
    *,
    output_file: str,
    fallback: str,
) -> None:
    payload = {
        "metadata": metadata,
        "output_file": output_file,
        "fallback": fallback,
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _remove_stale_segment_files(segment_path: Path, silence_path: Path, metadata_path: Path) -> None:
    for path in (segment_path, silence_path, metadata_path):
        if path.exists():
            try:
                path.unlink()
            except OSError:
                logger.warning("Could not remove stale TTS cache file: %s", path)


def _write_silence_wav(path: Path, duration: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = max(1, int(duration * SILENCE_SAMPLE_RATE))
    frame_bytes = b"\x00" * SILENCE_SAMPLE_WIDTH * SILENCE_CHANNELS
    chunk_size = SILENCE_SAMPLE_RATE
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(SILENCE_CHANNELS)
        wav.setsampwidth(SILENCE_SAMPLE_WIDTH)
        wav.setframerate(SILENCE_SAMPLE_RATE)
        remaining = frame_count
        while remaining > 0:
            count = min(remaining, chunk_size)
            wav.writeframes(frame_bytes * count)
            remaining -= count
    return path
