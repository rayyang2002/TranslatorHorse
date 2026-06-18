"""Unit tests for the language-agnostic A-tier dubbing aligner."""

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "video_translator"
sys.path.insert(0, str(PACKAGE))


def test_estimate_units_counts_cjk_per_character():
    from core.tts_aligner import estimate_speech_units

    assert estimate_speech_units("我都没有做图") == 6
    assert estimate_speech_units("日本語") == 3


def test_estimate_units_counts_vowel_groups_for_latin():
    from core.tts_aligner import estimate_speech_units

    # "I haven't made images" -> I, a, e, a, e, i, a, e -> 8 vowel groups
    assert estimate_speech_units("I haven't made images") == 8


def test_estimate_duration_is_language_independent_lookup():
    from core.tts_aligner import estimate_speech_duration

    en = estimate_speech_duration("hello", "en")
    zh = estimate_speech_duration("你好", "zh")
    assert en > 0 and zh > 0
    # Unknown language still returns a sane estimate (no crash, no branch).
    assert estimate_speech_duration("hello", "xx") > 0


def test_plan_tts_speed_natural_when_target_short_enough():
    from core.tts_aligner import plan_tts_speed

    # estimate exactly matches slot -> natural
    speed, reason = plan_tts_speed(slot=2.0, estimated_natural=2.0)
    assert speed == 1.0
    assert reason == "natural"


def test_plan_tts_speed_speeds_up_but_never_exceeds_max():
    from core.tts_aligner import MAX_SPEED, plan_tts_speed

    # zh->en like case: estimate 3.4s into a 2.0s slot -> ratio 1.7
    speed, reason = plan_tts_speed(slot=2.0, estimated_natural=3.4)
    assert reason == "speed-up"
    assert 1.0 < speed <= MAX_SPEED
    assert speed == pytest.approx(1.3, abs=0.01)  # capped at MAX_SPEED


def test_plan_tts_speed_slows_down_for_short_targets():
    from core.tts_aligner import MIN_SPEED, plan_tts_speed

    # en->zh like case: estimate 1.0s into a 2.0s slot -> slow to fill
    speed, reason = plan_tts_speed(slot=2.0, estimated_natural=1.0)
    assert reason == "slow-fill"
    assert MIN_SPEED <= speed < 1.0


def test_final_placement_fits_without_stretch():
    from core.tts_aligner import plan_final_placement

    segments = [{"start": 0.0, "end": 2.0}, {"start": 3.0, "end": 5.0}]
    durations = [1.8, 1.5]  # both fit their slots
    plan = plan_final_placement(segments, durations)

    assert len(plan.plans) == 2
    for p in plan.plans:
        assert p.atempo == 1.0
        assert p.reason == "fits"


def test_final_placement_capped_speed_for_massive_overflow():
    """The core fix: a hugely overlong clip must NOT be sped past MAX_SPEED."""
    from core.tts_aligner import MAX_SPEED, plan_final_placement

    # 4s clip in a 1s slot with no neighbouring gaps to borrow.
    segments = [{"start": 0.0, "end": 1.0}]
    durations = [4.0]
    plan = plan_final_placement(segments, durations)

    p = plan.plans[0]
    assert p.atempo <= MAX_SPEED
    assert p.atempo == pytest.approx(MAX_SPEED, abs=0.01)
    # Trim keeps the clip inside its slot — no bleed past the (single) segment.
    assert p.trim_to == pytest.approx(1.0, abs=0.01)


def test_final_placement_borrows_gap_before_speeding_up():
    """Overflow is absorbed from neighbouring silence before any speed-up."""
    from core.tts_aligner import plan_final_placement

    # 3s clip in a 2s slot, with a 3s gap before it that can be borrowed from.
    segments = [{"start": 5.0, "end": 7.0}]
    durations = [3.0]
    plan = plan_final_placement(segments, durations)

    p = plan.plans[0]
    # 1s overflow, borrowable = 5.0 * 0.6 = 3.0 -> borrow 1.0, slot grows to 3.0,
    # clip now fits with no speed-up.
    assert p.atempo == 1.0
    assert p.reason == "borrowed-gap"
    assert p.start == pytest.approx(4.0, abs=0.01)  # shifted back by 1.0


def test_final_placement_does_not_borrow_into_next_segment_start():
    """A borrowed tail must never push into the next segment's start time."""
    from core.tts_aligner import plan_final_placement

    segments = [
        {"start": 0.0, "end": 1.0},   # tiny slot
        {"start": 1.2, "end": 2.0},   # only 0.2s gap after the first
    ]
    durations = [3.0, 0.5]  # first clip badly overflows
    plan = plan_final_placement(segments, durations)

    p0 = plan.plans[0]
    # new_end clamped to next segment start (1.2); borrow from after is tiny.
    assert p0.end <= 1.2 + 1e-6
    assert p0.atempo <= 1.30  # capped even after partial borrowing


def test_merge_short_segments_combines_tiny_fragments():
    from core.tts_aligner import merge_short_segments

    segments = [
        {"start": 0.0, "end": 0.4, "text": "我", "translated_text": "I"},
        {"start": 0.4, "end": 0.7, "text": "都", "translated_text": "all"},
        {"start": 0.7, "end": 3.0, "text": "没有做图", "translated_text": "haven't made images"},
    ]
    merged = merge_short_segments(segments)

    # The first two tiny segments (0.3s and 0.4s, both < 0.8s threshold) are
    # merged into one. The third is 2.3s — well above the floor — so it stays.
    assert len(merged) == 2
    assert merged[0]["start"] == 0.0
    assert merged[0]["end"] == 0.7
    assert merged[0]["text"] == "我都"
    assert merged[1]["end"] == 3.0
    assert merged[1]["text"] == "没有做图"


def test_merge_short_segments_combines_all_tiny_when_contiguous():
    from core.tts_aligner import merge_short_segments

    segments = [
        {"start": 0.0, "end": 0.3, "text": "我", "translated_text": "I"},
        {"start": 0.3, "end": 0.6, "text": "都没有", "translated_text": "haven't"},
        {"start": 0.6, "end": 0.9, "text": "做图", "translated_text": "made images"},
    ]
    merged = merge_short_segments(segments)

    # All three are < 0.8s and the running merge stays < 4× floor.
    assert len(merged) == 1
    assert merged[0]["start"] == 0.0
    assert merged[0]["end"] == 0.9
    assert merged[0]["text"] == "我都没有做图"
    assert "haven't" in merged[0]["translated_text"]


def test_merge_short_segments_keeps_long_segments_intact():
    from core.tts_aligner import merge_short_segments

    segments = [
        {"start": 0.0, "end": 2.0, "text": "hello world", "translated_text": "你好世界"},
        {"start": 2.5, "end": 4.5, "text": "second line", "translated_text": "第二行"},
    ]
    merged = merge_short_segments(segments)
    assert merged == segments


def test_merge_short_segments_does_not_mutate_input():
    from core.tts_aligner import merge_short_segments

    segments = [
        {"start": 0.0, "end": 0.3, "text": "a", "translated_text": "A"},
        {"start": 0.3, "end": 2.0, "text": "b", "translated_text": "B"},
    ]
    original_end = segments[1]["end"]
    merge_short_segments(segments)
    assert segments[1]["end"] == original_end


def test_alignment_works_identically_across_language_pairs():
    """The whole point: identical algorithm for any direction.

    We feed the same slot/overflow math and assert the plan only depends on
    durations, not on which languages are involved.
    """
    from core.tts_aligner import plan_final_placement

    segments = [{"start": 0.0, "end": 1.5}]
    durations = [4.0]  # identical overflow regardless of language pair

    plan_a = plan_final_placement(segments, durations)
    # Re-running with the same numbers must yield the same plan. This stands in
    # for "zh->en, ja->fr, ar->en all produce identical treatment".
    plan_b = plan_final_placement(segments, durations)

    assert plan_a.plans[0].atempo == plan_b.plans[0].atempo
    assert plan_a.plans[0].trim_to == plan_b.plans[0].trim_to
    assert plan_a.plans[0].atempo <= 1.30


def test_empty_clip_yields_silent_plan():
    from core.tts_aligner import plan_final_placement

    segments = [{"start": 1.0, "end": 2.0}]
    plan = plan_final_placement(segments, [0.0])
    assert plan.plans[0].reason == "empty-clip"
    assert plan.plans[0].atempo == 1.0
