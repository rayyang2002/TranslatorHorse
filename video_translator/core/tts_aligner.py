"""Language-agnostic dubbing alignment (A-tier).

This module decides, for each translated subtitle segment, how its TTS audio
should be placed and time-stretched so that the dub is always:

  * natural and intelligible (speed never leaves a bounded, inaudible range),
  * basically synchronized with the subtitles (segment start aligned, no bleed
    into unrelated segments),
  * uniform across every language pair — zh<->en, ja->fr, ar->en, etc.

The strategy is driven entirely by measured audio duration vs. available time,
never by hard-coded per-language branches. A plan is produced up front so that:

  Phase 1 (TTS generation): each segment carries a ``tts_speed`` hint derived
    from an *estimate* of its natural duration. Edge-TTS (and any cooperating
    engine) uses it to speak slightly faster/slower at synthesis time, which
    sounds far better than post-hoc atempo compression.

  Phase 2 (mix): after the real audio duration is known, any remaining overflow
    is absorbed by (a) borrowing time from neighbouring gaps first, and only
    then (b) a capped atempo that never exceeds ``MAX_SPEED``. Whatever still
    overflows is allowed to drift slightly into the next gap (A-tier contract),
    never accelerated further.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from utils.logger import logger


# --- A-tier global constraints (language-agnostic) ---------------------------

MIN_SPEED = 0.85          # slowest TTS/atempo speed (fills gaps on short targets)
MAX_SPEED = 1.30          # fastest speed — the hard guarantee of intelligibility
MIN_SEGMENT_DURATION = 0.8  # segments shorter than this get merged into a neighbour
GAP_BORROW_RATIO = 0.6    # at most this fraction of adjacent silence may be borrowed


# --- Speech-density table (per target language, no if/else branching) --------

# Seconds of natural TTS audio per "unit" of text for each target language.
# A "unit" is one spoken syllable-like token: for latin/cyrillic alphabets we
# count vowel groups; for CJK each character is one unit. The exact numbers are
# calibration constants — what matters is that they exist for every supported
# language, so the estimator never special-cases a direction.
SPEECH_DENSITY: dict[str, float] = {
    "zh": 0.26,   # ~0.26s per character
    "ja": 0.24,
    "ko": 0.24,
    "vi": 0.25,
    "th": 0.27,
    "en": 0.27,   # ~0.27s per syllable (vowel group)
    "de": 0.28,
    "nl": 0.27,
    "af": 0.27,
    "sv": 0.27,
    "da": 0.26,
    "fi": 0.24,
    "fr": 0.24,
    "es": 0.24,
    "pt": 0.24,
    "it": 0.24,
    "ro": 0.25,
    "pl": 0.26,
    "cs": 0.25,
    "hu": 0.25,
    "el": 0.27,
    "ru": 0.25,
    "tr": 0.26,
    "ar": 0.27,
    "hi": 0.30,
    "bn": 0.30,
    "ur": 0.29,
    "mr": 0.30,
    "pa": 0.29,
    "ta": 0.30,
    "te": 0.30,
    "id": 0.24,
}
DEFAULT_DENSITY = 0.27


_CJK_RANGES = (
    (0x3040, 0x30FF),   # Hiragana / Katakana
    (0x3400, 0x4DBF),   # CJK Extension A
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0xAC00, 0xD7AF),   # Hangul Syllables
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
)


def estimate_speech_units(text: str) -> int:
    """Count spoken units in ``text``.

    For CJK text every character counts as one unit; otherwise we count vowel
    groups (a reasonable syllable proxy for latin/cyrillic/etc. scripts). This
    is intentionally crude and universal — it is only used to pick a sensible
    TTS speed hint, and the real audio duration always overrides it later.
    """
    text = (text or "").strip()
    if not text:
        return 0

    units = 0
    prev_vowel = False
    for ch in text:
        code = ord(ch)
        if any(lo <= code <= hi for lo, hi in _CJK_RANGES):
            units += 1
            prev_vowel = False
            continue
        low = ch.lower()
        if low in "aeiouyаоэуыеяёию":
            if not prev_vowel:
                units += 1
                prev_vowel = True
        else:
            prev_vowel = False
    return max(units, 1)


def estimate_speech_duration(text: str, language: str) -> float:
    """Estimate natural TTS duration for ``text`` in ``language`` seconds."""
    density = SPEECH_DENSITY.get((language or "").lower(), DEFAULT_DENSITY)
    return estimate_speech_units(text) * density


# --- Plan types --------------------------------------------------------------

@dataclass
class SegmentPlan:
    """Alignment plan for one subtitle segment.

    Attributes:
        start / end:        adjusted placement on the timeline (after gap
                            borrowing). In the common case these equal the
                            original subtitle times.
        tts_speed:          speed hint handed to the TTS engine at synthesis
                            time (Phase 1). 1.0 = natural.
        atempo:             final atempo applied during the mix (Phase 2). 1.0
                            means no time-stretch. Capped to [MIN_SPEED, MAX_SPEED].
        trim_to:            hard cut point on the placed audio (seconds from its
                            start). ``None`` keeps the whole clip.
        reason:             short human-readable note for logging/diagnostics.
    """

    start: float
    end: float
    tts_speed: float = 1.0
    atempo: float = 1.0
    trim_to: Optional[float] = None
    reason: str = "natural"


@dataclass
class AlignmentPlan:
    plans: list[SegmentPlan] = field(default_factory=list)


# --- Phase 1: pre-generation speed hints -------------------------------------

def plan_tts_speed(slot: float, estimated_natural: float) -> tuple[float, str]:
    """Pick a TTS speed hint for a segment given its slot and estimated duration.

    Returns ``(tts_speed, reason)`` where ``tts_speed`` is the multiplier handed
    to the TTS engine (1.0 = natural). The hint is clamped to the same bounds
    used in Phase 2 so the two phases agree on what "intelligible" means.
    """
    slot = max(0.05, slot)
    ratio = estimated_natural / slot

    if ratio <= 1.0:
        # Target is shorter than the estimate (e.g. en->zh): speak a touch slower
        # so the clip fills more of the slot instead of leaving a big gap.
        target_fill = max(1.0 / ratio, 1.0)
        tts_speed = 1.0 / min(target_fill, 1.0 / MIN_SPEED)
        tts_speed = max(MIN_SPEED, tts_speed)
        if abs(tts_speed - 1.0) < 0.02:
            return 1.0, "natural"
        return round(tts_speed, 3), "slow-fill"

    # Target is longer (e.g. zh->en): ask TTS to speed up, but never past MAX_SPEED.
    tts_speed = min(ratio, MAX_SPEED)
    if tts_speed <= 1.02:
        return 1.0, "natural"
    return round(tts_speed, 3), "speed-up"


# --- Phase 2: post-generation placement --------------------------------------

def plan_final_placement(
    segments: list[dict],
    actual_durations: list[float],
    *,
    max_speed: float = MAX_SPEED,
    min_speed: float = MIN_SPEED,
    gap_borrow_ratio: float = GAP_BORROW_RATIO,
) -> AlignmentPlan:
    """Build the final placement plan using real measured audio durations.

    ``segments[i]`` must have ``start``/``end``; ``actual_durations[i]`` is the
    measured length of its TTS clip. The plan borrows from neighbouring gaps
    before applying a capped atempo, and never lets speed exceed ``max_speed``.
    """
    if len(segments) != len(actual_durations):
        raise ValueError("segments and actual_durations length mismatch")

    plans: list[SegmentPlan] = []
    n = len(segments)

    for i, segment in enumerate(segments):
        orig_start = float(segment.get("start", 0) or 0)
        orig_end = float(segment.get("end", orig_start) or orig_start)
        slot = max(0.0, orig_end - orig_start)
        actual = max(0.0, actual_durations[i])

        if actual <= 0.0:
            plans.append(SegmentPlan(start=orig_start, end=orig_end, reason="empty-clip"))
            continue

        if actual <= slot + 0.001:
            # Fits comfortably (with or without a tail gap). No stretch needed.
            plans.append(SegmentPlan(
                start=orig_start,
                end=orig_end,
                trim_to=slot if slot > 0 else None,
                reason="fits",
            ))
            continue

        # Overflow: try to borrow from adjacent gaps first (no speed change).
        prev_plan_end = plans[i - 1].end if i > 0 else 0.0
        next_seg_start = float(segments[i + 1].get("start", orig_end) or orig_end) if i + 1 < n else None

        gap_before = max(0.0, orig_start - prev_plan_end)
        gap_after = (max(0.0, next_seg_start - orig_end) if next_seg_start is not None else 0.0)
        total_borrowable = (gap_before + gap_after) * gap_borrow_ratio

        overflow = actual - slot
        borrowed = min(total_borrowable, overflow)

        # Split the borrowed time between the two sides proportionally to the
        # available gaps; fall back to 50/50 when neither gap exists.
        if total_borrowable > 1e-6:
            share_before = (gap_before * gap_borrow_ratio) / total_borrowable
        else:
            share_before = 0.0
        borrow_before = borrowed * share_before
        borrow_after = borrowed - borrow_before

        new_start = max(prev_plan_end if i > 0 else 0.0, orig_start - borrow_before)
        new_end = orig_end + borrow_after
        if next_seg_start is not None:
            new_end = min(new_end, next_seg_start)
        new_slot = max(0.0, new_end - new_start)

        remaining = max(0.0, actual - new_slot)

        if remaining <= 0.001:
            plans.append(SegmentPlan(
                start=new_start,
                end=new_end,
                trim_to=new_slot if new_slot > 0 else None,
                reason="borrowed-gap",
            ))
            continue

        # Still overflowing after borrowing: apply a capped atempo. Never exceed
        # max_speed — anything left over becomes a tail drift (A-tier contract).
        needed_speed = actual / new_slot if new_slot > 0 else max_speed
        atempo = min(needed_speed, max_speed)
        atempo = max(min_speed, atempo)

        if atempo <= 1.02:
            # Even at natural speed it (almost) fits now; just place it.
            plans.append(SegmentPlan(
                start=new_start,
                end=new_end,
                trim_to=new_slot if new_slot > 0 else None,
                reason="borrowed-gap",
            ))
            continue

        sped_duration = actual / atempo
        if sped_duration > new_slot:
            # Cap reached: trim to the slot so we don't bleed past the next
            # subtitle's start; the audible tail drifts but stays bounded.
            trim_to = new_slot if new_slot > 0 else None
        else:
            trim_to = sped_duration

        plans.append(SegmentPlan(
            start=new_start,
            end=new_end,
            atempo=round(atempo, 4),
            trim_to=trim_to,
            reason="capped-speed",
        ))

    return AlignmentPlan(plans=plans)


# --- Pre-scan: merge tiny segments -------------------------------------------

def merge_short_segments(
    segments: list[dict],
    *,
    min_duration: float = MIN_SEGMENT_DURATION,
) -> list[dict]:
    """Merge runs of consecutive tiny segments into single segments.

    Whisper (especially ``base``) often emits 0.3-0.7s fragments, which become
    impossible-to-fit slots for any target language. Merging neighbours removes
    that pressure without dropping content. Both ``text`` and
    ``translated_text`` are concatenated when present.

    The output is a new list; the input is not mutated.
    """
    if not segments:
        return []

    def _absorb(prev: dict, segment: dict) -> None:
        prev["end"] = max(float(prev.get("end", 0) or 0), float(segment.get("end", 0) or 0))
        if segment.get("text"):
            prev["text"] = (prev.get("text", "") + segment["text"]).strip()
        if segment.get("translated_text"):
            prev["translated_text"] = (
                prev.get("translated_text", "") + " " + segment["translated_text"]
            ).strip()

    merged: list[dict] = []
    for segment in segments:
        start = float(segment.get("start", 0) or 0)
        end = float(segment.get("end", start) or start)
        duration = end - start

        # Absorb a short segment into the running merge as long as the running
        # merge has not already grown past ~4x the floor (keeps merges local).
        if merged and duration < min_duration:
            prev = merged[-1]
            running_span = float(prev.get("end", 0) or 0) - float(prev.get("start", 0) or 0)
            if running_span < min_duration * 4:
                _absorb(prev, segment)
                continue

        merged.append(dict(segment))

    # Second pass: any leftover standalone short segment that could not be
    # merged forward gets merged backward into the previous one.
    if len(merged) >= 2:
        last = merged[-1]
        last_start = float(last.get("start", 0) or 0)
        last_end = float(last.get("end", last_start) or last_start)
        if last_end - last_start < min_duration:
            _absorb(merged[-2], last)
            merged.pop()

    if len(merged) != len(segments):
        logger.info(
            "Merged %d short segments into %d (min_duration=%.2fs)",
            len(segments),
            len(merged),
            min_duration,
        )
    return merged


__all__ = [
    "MIN_SPEED",
    "MAX_SPEED",
    "MIN_SEGMENT_DURATION",
    "GAP_BORROW_RATIO",
    "SegmentPlan",
    "AlignmentPlan",
    "estimate_speech_units",
    "estimate_speech_duration",
    "plan_tts_speed",
    "plan_final_placement",
    "merge_short_segments",
]
