"""Apply osu!mania mods to a beatmap. Returns a modded beatmap + audio rate + visual flags."""

from __future__ import annotations

import enum
from dataclasses import dataclass, replace

from osu_mania_renderer_v2.models import BeatmapInfo, HoldNote, Note, ReplayInfo, VisualMods


class Mod(enum.IntFlag):
    """osu! mod bitmask values, ordered by bit position."""

    NF = 1 << 0
    EZ = 1 << 1
    HD = 1 << 3
    HR = 1 << 4
    SD = 1 << 5
    DT = 1 << 6
    HT = 1 << 8
    NC = 1 << 9
    FL = 1 << 10
    PF = 1 << 14
    K4 = 1 << 15
    K5 = 1 << 16
    K6 = 1 << 17
    K7 = 1 << 18
    K8 = 1 << 19
    FI = 1 << 20
    RD = 1 << 21
    K9 = 1 << 24
    KC = 1 << 25
    K1 = 1 << 26
    K3 = 1 << 27
    K2 = 1 << 28
    V2 = 1 << 29
    MR = 1 << 30


@dataclass(frozen=True)
class ModResult:
    beatmap: BeatmapInfo
    audio_rate: float
    visual_mods: VisualMods
    warnings: tuple[str, ...] = ()


# Display order roughly matches in-game ordering (difficulty-affecting first,
# then visual, then misc). Speed mods are mutually exclusive (DT > NC > HT).
_DISPLAY_ORDER: tuple[tuple[Mod, str], ...] = (
    (Mod.EZ, "EZ"),
    (Mod.NF, "NF"),
    (Mod.HT, "HT"),
    (Mod.DT, "DT"),
    (Mod.NC, "NC"),
    (Mod.HR, "HR"),
    (Mod.SD, "SD"),
    (Mod.PF, "PF"),
    (Mod.HD, "HD"),
    (Mod.FI, "FI"),
    (Mod.FL, "FL"),
    (Mod.MR, "MR"),
    (Mod.RD, "RD"),
    (Mod.KC, "KC"),
    (Mod.V2, "V2"),
)


def mod_acronyms(mods_bitfield: int, key_count: int) -> tuple[str, ...]:
    """Replay mod bitfield → display-ordered list of pill labels.

    Always emits the key-count pill first (e.g., "4K"). Then any gameplay
    mods present in the bitfield, in canonical display order. NC implies DT
    in osu!'s bitfield, so we drop the duplicate DT label.
    """
    out: list[str] = [f"{key_count}K"]
    has_nc = bool(mods_bitfield & Mod.NC)
    for bit, name in _DISPLAY_ORDER:
        if not (mods_bitfield & bit):
            continue
        if name == "DT" and has_nc:
            continue  # NC supersedes DT
        out.append(name)
    return tuple(out)


def apply_mods(beatmap: BeatmapInfo, replay: ReplayInfo) -> ModResult:
    mods = replay.mods
    warnings: list[str] = []

    # Speed.
    if mods & Mod.DT or mods & Mod.NC:
        audio_rate = 1.5
    elif mods & Mod.HT:
        audio_rate = 0.75
    else:
        audio_rate = 1.0

    # Apply speed to note times.
    notes = _rescale_times(beatmap.notes, audio_rate)
    total = int(beatmap.total_duration_ms / audio_rate)

    # Mirror.
    if mods & Mod.MR:
        notes = _mirror(notes, beatmap.key_count)

    # Random — explicitly unsupported.
    if mods & Mod.RD:
        warnings.append("Random (RD) is not supported; rendering as NM column order")

    # Key Coop — explicitly unsupported.
    if mods & Mod.KC:
        warnings.append("Key Coop (KC) is not supported; rendering as single playfield")

    visual = VisualMods(
        hidden=bool(mods & Mod.HD),
        fade_in=bool(mods & Mod.FI),
        flashlight=bool(mods & Mod.FL),
        score_v2=bool(mods & Mod.V2),
    )

    modded = replace(beatmap, notes=tuple(notes), total_duration_ms=total)
    return ModResult(
        beatmap=modded,
        audio_rate=audio_rate,
        visual_mods=visual,
        warnings=tuple(warnings),
    )


def _rescale_times(notes: tuple, rate: float) -> list:
    if rate == 1.0:
        return list(notes)
    out: list = []
    for n in notes:
        if isinstance(n, HoldNote):
            out.append(
                HoldNote(
                    column=n.column,
                    time_ms=int(n.time_ms / rate),
                    end_time_ms=int(n.end_time_ms / rate),
                )
            )
        else:
            out.append(Note(column=n.column, time_ms=int(n.time_ms / rate)))
    return out


def _mirror(notes: list, key_count: int) -> list:
    out: list = []
    for n in notes:
        new_col = (key_count - 1) - n.column
        if isinstance(n, HoldNote):
            out.append(HoldNote(column=new_col, time_ms=n.time_ms, end_time_ms=n.end_time_ms))
        else:
            out.append(Note(column=new_col, time_ms=n.time_ms))
    return out
