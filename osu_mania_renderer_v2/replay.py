"""Parse osu!mania .osr files via osrparse, including the keypress timeline."""
from __future__ import annotations

import lzma
import struct
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

    events = _decode_key_events(r, path)

    total = (
        r.count_300 + r.count_100 + r.count_50 + r.count_miss
        + r.count_katu + r.count_geki
    )
    # Mania accuracy weighting must match what osu! / the website card shows:
    # stable replays weight the rainbow-300 (geki) at 320, lazer (and stable +
    # Score V2) at 305. Detect via game_version (lazer writes 9-digit
    # ≥30000000) and the Score V2 mod (1<<29) — identical to the bot's
    # osr_parser so the in-render accuracy lands on the same number.
    mw = 305 if (int(getattr(r, "game_version", 0) or 0) >= 30000000
                 or (int(r.mods) & (1 << 29))) else 320
    if total == 0:
        accuracy = 0.0
    else:
        weighted = (
            50 * r.count_50 + 100 * r.count_100 + 200 * r.count_katu
            + 300 * r.count_300 + mw * r.count_geki
        )
        accuracy = round((weighted / (mw * total)) * 100, 4)

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
        mania_max_weight=mw,
    )


_SEED_DELTA = -12345  # final frame carries the RNG seed in X, time_delta -12345


def _recover_leadin_offset(path: Path) -> int:
    """Recover the replay-clock lead-in that osrparse silently discards.

    osu!stable replays begin with up to two placeholder frames at the
    off-screen sentinel position (256, -500). osu!'s own LegacyScoreDecoder
    ACCUMULATES each frame's time delta into the running clock *before* it
    drops those placeholders (``lastTime += diff`` precedes the
    ``i < 2 && (256,-500)`` skip), so the second placeholder's delta carries
    the audio lead-in / intro-skip offset. osrparse instead ``continue``s past
    these frames (osrparse/replay.py: ``if i < 2 and float(x) == 256 and
    float(y) == -500: continue``), throwing their deltas away — the lead-in
    never reaches ``Replay.replay_data``. Accumulating that stream from 0 then
    starts the clock too early by the whole lead-in, so every object is
    sampled before the player reached it (mass over-miss on stable replays
    whose intro-skip is not cancelled by a <-5000 ms first frame).

    Reproduce osu!'s accumulation: read the raw replay-data blob and sum the
    deltas of exactly the leading placeholder frames osrparse strips. Returns 0
    when there are none (lazer replays carry no placeholder frames; a clean
    stable play carries a ~0 ms lead-in), so already-aligned replays are left
    byte-identical. Fail-soft: any decode problem returns 0 (the pre-fix
    behaviour) rather than raising.
    """
    try:
        data = Path(path).read_bytes()
        off = 0

        def _skip_string() -> None:
            nonlocal off
            tag = data[off]
            off += 1
            if tag == 0x00:
                return
            if tag != 0x0b:
                raise ValueError(f"bad string tag {tag}")
            length = shift = 0
            while True:
                byte = data[off]
                off += 1
                length |= (byte & 0x7F) << shift
                if not (byte & 0x80):
                    break
                shift += 7
            off += length

        off += 1                       # mode (byte)
        off += 4                       # game version (int32)
        _skip_string()                 # beatmap md5
        _skip_string()                 # player name
        _skip_string()                 # replay md5
        off += 2 * 6                   # 300/100/50/geki/katu/miss (6 shorts)
        off += 4                       # score (int32)
        off += 2                       # max combo (short)
        off += 1                       # perfect (byte)
        off += 4                       # mods (int32)
        _skip_string()                 # life-bar graph
        off += 8                       # timestamp (int64)
        rlen = struct.unpack_from("<i", data, off)[0]
        off += 4                       # replay-data length (int32)
        raw = lzma.decompress(data[off:off + rlen],
                              format=lzma.FORMAT_AUTO).decode("ascii", "replace")

        lead = 0
        for i, group in enumerate(raw.rstrip(",").split(",")):
            if not group:
                continue
            fields = group.split("|")
            delta = int(fields[0])
            if delta == _SEED_DELTA:   # RNG seed (never a leading frame) — stop
                break
            # osrparse strips only the first two frames, and only when they sit
            # at the (256, -500) sentinel; mirror that set exactly.
            if i < 2 and float(fields[1]) == 256.0 and float(fields[2]) == -500.0:
                lead += delta
                continue
            break                      # first real frame: nothing more to strip
        return lead
    except Exception:  # noqa: BLE001 - never let a header quirk break parsing
        return 0


def _decode_key_events(r: Replay, path: Path) -> list[KeyEvent]:
    """Convert osrparse ReplayEventMania entries to absolute-time KeyEvents.

    Two .osr quirks must be handled or the whole press timeline desyncs:
      * the last frame is an RNG-seed sentinel (time_delta -12345) and must be
        skipped, never accumulated;
      * osrparse silently DROPS the up-to-two leading (256, -500) placeholder
        frames WITHOUT accumulating their deltas, discarding the audio lead-in
        / intro-skip that osu!'s LegacyScoreDecoder folds into the running
        clock (``lastTime += diff`` precedes the skip). Seed the clock with
        that recovered lead-in (see _recover_leadin_offset) and then accumulate
        every gameplay delta verbatim — exactly as osu! does. This replaces the
        old ``-5000`` first-frame guard, which only APPROXIMATED the missing
        lead-in by zeroing a large-negative first delta: that guard shifted
        every press by the whole intro-skip on replays whose lead-in was not a
        <-5000 ms cancel (a clean play rendering as combo-67 / 584-miss
        garbage) and, once the lead-in is seeded, would wrongly re-cancel the
        legitimate large-negative first delta that returns the clock to ~0. The
        seed is 0 for lazer and already-aligned stable plays, so those stay
        byte-identical."""
    out: list[KeyEvent] = []
    t = _recover_leadin_offset(path)
    for ev in r.replay_data or []:
        delta = int(getattr(ev, "time_delta", 0))
        if delta == _SEED_DELTA:
            continue
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
