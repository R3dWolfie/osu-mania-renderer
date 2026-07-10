import lzma
import struct
from pathlib import Path

import pytest

from osu_mania_renderer_v2.errors import NotAManiaError
from osu_mania_renderer_v2.replay import _recover_leadin_offset, parse_replay


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


# --- osu!stable lead-in fix (replay._recover_leadin_offset) -----------------
# osrparse silently drops the up-to-two leading (256,-500) placeholder frames
# WITHOUT accumulating their deltas, discarding the audio lead-in / intro-skip
# that osu!'s LegacyScoreDecoder folds into the running clock. Accumulating the
# survivors from 0 then shifts EVERY press by the whole lead-in -> a clean play
# renders as combo-67 / 584-miss garbage. The committed real fixture
# stable_leadin.osr carries a 2717ms intro-skip that the OLD -5000 guard did
# NOT catch (its cancel delta is > -5000), so it was mistimed by the whole
# 2717ms; it now lands exactly where osu! puts it.
_STABLE_LEADIN_MS = 2717


def _uleb_string(s: str) -> bytes:
    b = s.encode("utf-8")
    out = bytearray([0x0b])
    n = len(b)
    while True:
        byte = n & 0x7F
        n >>= 7
        out.append(byte | 0x80 if n else byte)
        if not n:
            break
    return bytes(out) + b


def _make_osr(frames: str, mode: int = 3) -> bytes:
    blob = lzma.compress(frames.encode("ascii"), format=lzma.FORMAT_ALONE)
    out = bytearray()
    out.append(mode)
    out += struct.pack("<i", 20260101)          # stable game version
    out += _uleb_string("beatmapmd5")
    out += _uleb_string("player")
    out += _uleb_string("replaymd5")
    out += struct.pack("<6h", 0, 0, 0, 0, 0, 0)
    out += struct.pack("<i", 0)
    out += struct.pack("<h", 0)
    out += struct.pack("<b", 0)
    out += struct.pack("<i", 0)
    out += _uleb_string("")
    out += struct.pack("<q", 0)
    out += struct.pack("<i", len(blob))
    out += blob
    out += struct.pack("<q", 0)
    return bytes(out)


def test_recover_leadin_synthetic(tmp_path: Path):
    # two (256,-500) placeholders summing to 2342 -> that IS the lead-in; the
    # first real frame is not a <-5000 cancel, so this is the catastrophic case.
    p = tmp_path / "s.osr"
    p.write_bytes(_make_osr("0|256|-500|0,2342|256|-500|0,14|0|18|0,20|0|18|0,-12345|0|0|9999,"))
    assert _recover_leadin_offset(p) == 2342


def test_recover_leadin_lazer_synthetic(tmp_path: Path):
    p = tmp_path / "l.osr"
    p.write_bytes(_make_osr("0|1|18|0,16|1|18|0,17|1|18|0,-12345|0|0|9999,"))
    assert _recover_leadin_offset(p) == 0


def test_recover_leadin_single_placeholder(tmp_path: Path):
    p = tmp_path / "one.osr"
    p.write_bytes(_make_osr("0|256|-500|0,50|1|18|0,-12345|0|0|9999,"))
    assert _recover_leadin_offset(p) == 0


def test_recover_leadin_failsoft(tmp_path: Path):
    p = tmp_path / "junk.osr"
    p.write_bytes(b"not an osr file at all")
    assert _recover_leadin_offset(p) == 0


def test_lazer_replay_zero_offset(fixtures_dir: Path):
    # the existing lazer fixture carries no placeholder frames -> byte-identical.
    assert _recover_leadin_offset(fixtures_dir / "ao_infinity_hard.osr") == 0


def test_real_stable_leadin_recovered(fixtures_dir: Path):
    assert _recover_leadin_offset(fixtures_dir / "stable_leadin.osr") == _STABLE_LEADIN_MS


def test_parse_seeds_clock_matches_osu(fixtures_dir: Path):
    """End-to-end on a real 2717ms-intro-skip stable replay: the parsed press
    clock must start at the osu!-correct time (seed + first survivor delta),
    which the OLD from-0 logic got wrong by the whole lead-in."""
    from osrparse import Replay
    fx = fixtures_dir / "stable_leadin.osr"
    seed = _recover_leadin_offset(fx)
    r = Replay.from_path(fx)
    deltas = [int(e.time_delta) for e in (r.replay_data or [])
              if int(e.time_delta) != -12345]
    expected_first = max(0, seed + deltas[0])
    old_first = max(0, 0 if deltas[0] < -5000 else deltas[0])

    info = parse_replay(fx)
    times = [e.time_ms for e in info.key_events]
    assert times == sorted(times)
    assert all(t >= 0 for t in times)
    assert min(times) == expected_first
    assert old_first != expected_first     # regression: old logic was wrong here
