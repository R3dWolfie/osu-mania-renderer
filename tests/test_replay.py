from pathlib import Path

import pytest

from osu_mania_renderer_v2.errors import NotAManiaError
from osu_mania_renderer_v2.replay import parse_replay


def test_parse_mania_replay(fixtures_dir: Path):
    r = parse_replay(fixtures_dir / "ao_infinity_hard.osr")
    assert r.mode == 3
    assert r.player_name
    assert r.beatmap_md5
    assert 0.0 <= r.accuracy <= 100.0
    assert r.max_combo > 0
    assert len(r.key_events) > 0
    # KeyEvents must be sorted by time and have a non-negative time.
    assert r.key_events[0].time_ms >= 0
    times = [e.time_ms for e in r.key_events]
    assert times == sorted(times)
    # Regression: ensure keypress bitmask actually decodes (osrparse 7.x .keys).
    assert any(e.keys_held != 0 for e in r.key_events)


def test_reject_std_replay(fixtures_dir: Path):
    with pytest.raises(NotAManiaError):
        parse_replay(fixtures_dir / "std_replay.osr")


def test_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        parse_replay(tmp_path / "nope.osr")
