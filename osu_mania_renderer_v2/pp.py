"""Performance-points helpers powered by rosu-pp-py.

Computes:
  * `pp`     — the player's actual PP for this replay
  * `max_pp` — the map's theoretical SS-FC PP (100% acc, full max combo)

Stays out of the hot render loop — `compute_pp` is called once at render
start and the result is baked into every frame's SceneState.
"""
from __future__ import annotations

import logging
from pathlib import Path

from osu_mania_renderer_v2.models import ReplayInfo

log = logging.getLogger("osu_mania_renderer_v2.pp")


def _mods_to_acronyms(mods_bitfield: int) -> str:
    """Translate osu! mod bits → comma-separated acronyms rosu-pp accepts.
    Excludes key-count mods (1K..9K) — rosu-pp reads keys from the beatmap."""
    table = (
        (1 << 0,  "NF"), (1 << 1,  "EZ"), (1 << 3,  "HD"),
        (1 << 4,  "HR"), (1 << 5,  "SD"), (1 << 6,  "DT"),
        (1 << 8,  "HT"), (1 << 9,  "NC"), (1 << 10, "FL"),
        (1 << 14, "PF"), (1 << 20, "FI"), (1 << 21, "RD"),
        (1 << 25, "KC"), (1 << 29, "V2"), (1 << 30, "MR"),
    )
    out: list[str] = []
    has_nc = bool(mods_bitfield & (1 << 9))
    for bit, name in table:
        if not (mods_bitfield & bit):
            continue
        if name == "DT" and has_nc:  # NC implies DT in the bitfield
            continue
        out.append(name)
    return ",".join(out)


def compute_pp(osu_path: Path, replay: ReplayInfo) -> tuple[float, float]:
    """Return (player_pp, max_fc_pp). Returns (0.0, 0.0) on any failure
    (rosu-pp missing, unsupported map, suspicious beatmap…)."""
    try:
        import rosu_pp_py as rosu  # type: ignore[import-not-found]
    except ImportError:
        log.warning("rosu_pp_py not installed; PP set to 0")
        return 0.0, 0.0

    try:
        bmap = rosu.Beatmap(path=str(osu_path))
        if bmap.is_suspicious():
            return 0.0, 0.0
        mods = _mods_to_acronyms(replay.mods)
        # rosu-pp 4.x's `lazer` flag is counter-intuitive for mania:
        #   lazer=True  → 74 pp (matches osu!'s score page)
        #   lazer=False → 90 pp (the older "stable" formula, no longer
        #                        what the website displays for mania)
        # The osu! site adopted lazer mania PP when lazer went official,
        # so we always use it now regardless of how the score was recorded.
        lazer = True
        perf_kwargs = dict(
            lazer=lazer,
            n_geki=int(replay.count_geki),
            n300=int(replay.count_300),
            n_katu=int(replay.count_katu),
            n100=int(replay.count_100),
            n50=int(replay.count_50),
            misses=int(replay.count_miss),
            combo=int(replay.max_combo),
        )
        if mods:
            perf_kwargs["mods"] = mods
        player_perf = rosu.Performance(**perf_kwargs).calculate(bmap)

        # Max possible: SS-FC with the same mods, in the same scoring
        # system (stable, per the comment above).
        max_kwargs = dict(lazer=lazer, accuracy=100.0)
        if mods:
            max_kwargs["mods"] = mods
        max_perf = rosu.Performance(**max_kwargs).calculate(bmap)
        return float(player_perf.pp or 0.0), float(max_perf.pp or 0.0)
    except Exception as e:  # noqa: BLE001
        log.warning("pp_compute_failed", extra={"error": str(e)})
        return 0.0, 0.0
