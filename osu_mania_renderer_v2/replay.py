"""Parse osu!mania .osr files via osrparse, including the keypress timeline."""
from __future__ import annotations

from pathlib import Path

from osrparse import GameMode, Replay

from osu_mania_renderer_v2.errors import NotAManiaError, ReplayParseError
from osu_mania_renderer_v2.models import KeyEvent, ReplayInfo


def parse_replay(path: Path) -> ReplayInfo:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        r = Replay.from_path(path)
    except Exception as e:
        raise ReplayParseError(f"osrparse failed: {e}") from e

    mode = r.mode.value if isinstance(r.mode, GameMode) else int(r.mode)
    if mode != 3:
        raise NotAManiaError(mode)

    events = _decode_key_events(r)

    total = (
        r.count_300 + r.count_100 + r.count_50 + r.count_miss
        + r.count_katu + r.count_geki
    )
    if total == 0:
        accuracy = 0.0
    else:
        # Match osu!'s displayed mania accuracy (305 weight for geki/320,
        # 305 denominator). The older 300/300 formula reads ~0.7 % high
        # vs what the score page actually shows.
        weighted = (
            50 * r.count_50 + 100 * r.count_100 + 200 * r.count_katu
            + 300 * r.count_300 + 305 * r.count_geki
        )
        accuracy = round((weighted / (305 * total)) * 100, 4)

    return ReplayInfo(
        mode=mode,
        beatmap_md5=r.beatmap_hash,
        player_name=r.username,
        replay_md5=r.replay_hash,
        mods=int(r.mods),
        key_events=tuple(events),
        score=int(r.score),
        accuracy=accuracy,
        max_combo=int(r.max_combo),
        count_geki=int(r.count_geki),
        count_300=int(r.count_300),
        count_katu=int(r.count_katu),
        count_100=int(r.count_100),
        count_50=int(r.count_50),
        count_miss=int(r.count_miss),
        grade=_grade(accuracy, r),
    )


def _decode_key_events(r: Replay) -> list[KeyEvent]:
    """Convert osrparse ReplayEventMania entries to absolute-time KeyEvents."""
    out: list[KeyEvent] = []
    t = 0
    for ev in r.replay_data or []:
        delta = int(getattr(ev, "time_delta", 0))
        t += delta
        # osrparse 7.x: ReplayEventMania.keys is the bitmask of held columns.
        keys = int(getattr(ev, "keys", 0))
        out.append(KeyEvent(time_ms=max(t, 0), keys_held=keys))
    # Deduplicate same-time entries by keeping the latest.
    dedup: dict[int, KeyEvent] = {}
    for e in out:
        dedup[e.time_ms] = e
    return sorted(dedup.values(), key=lambda e: e.time_ms)


def _grade(accuracy: float, r: Replay) -> str:
    total = (
        r.count_geki + r.count_300 + r.count_katu + r.count_100
        + r.count_50 + r.count_miss
    )
    if total == 0:
        return "D"
    if r.count_300 == 0 and r.count_katu == 0 and r.count_100 == 0 \
            and r.count_50 == 0 and r.count_miss == 0:
        return "SS"
    # osu!mania grading is purely accuracy-based — no "no misses" or
    # "≤1 % 50s" requirement for S (those rules apply to osu!standard).
    if accuracy >= 95.0:
        return "S"
    if accuracy >= 90.0:
        return "A"
    if accuracy >= 80.0:
        return "B"
    if accuracy >= 70.0:
        return "C"
    return "D"
