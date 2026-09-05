from pathlib import Path

import pytest

from osu_mania_renderer_v2.beatmap import parse_beatmap
from osu_mania_renderer_v2.errors import BeatmapParseError, NotAManiaError
from osu_mania_renderer_v2.models import HoldNote


def test_parse_ao_infinity_hard(fixtures_dir: Path):
    bm = parse_beatmap(fixtures_dir / "ao_infinity_hard.osu")
    assert bm.key_count == 4
    assert bm.audio_filename == "audio.mp3"
    assert bm.artist == "Seiryu"
    assert bm.title == "AO-INFINITY"
    assert bm.difficulty == "Hard"
    assert bm.beatmap_id == 2071816
    assert len(bm.notes) > 100  # real map has lots
    # First note's column is in range and time is non-negative.
    n0 = bm.notes[0]
    assert 0 <= n0.column < bm.key_count
    assert n0.time_ms >= 0


def test_parse_notes_are_sorted_by_time(fixtures_dir: Path):
    bm = parse_beatmap(fixtures_dir / "ao_infinity_hard.osu")
    times = [n.time_ms for n in bm.notes]
    assert times == sorted(times)


def test_parse_holds_have_end_time(fixtures_dir: Path):
    bm = parse_beatmap(fixtures_dir / "ao_infinity_hard.osu")
    holds = [n for n in bm.notes if isinstance(n, HoldNote)]
    if holds:  # most maps have at least one hold
        h = holds[0]
        assert h.end_time_ms > h.time_ms


def test_non_mania_rejected(tmp_path: Path):
    p = tmp_path / "std.osu"
    p.write_text("osu file format v14\n[General]\nMode: 0\n[HitObjects]\n")
    with pytest.raises(NotAManiaError):
        parse_beatmap(p)


def test_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        parse_beatmap(tmp_path / "nope.osu")


def test_malformed_raises(tmp_path: Path):
    p = tmp_path / "bad.osu"
    p.write_text("this is not an osu file")
    with pytest.raises(BeatmapParseError):
        parse_beatmap(p)
