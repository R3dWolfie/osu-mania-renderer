"""Apply osu!mania mods to a beatmap. Returns a modded beatmap + audio rate + visual flags."""

from __future__ import annotations

import enum
from dataclasses import dataclass, replace

from osu_mania_renderer_v2.beatmap.models import BeatmapInfo, HoldNote, Note, ReplayInfo, VisualMods


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
    AT = 1 << 11
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
class LegacyModIcon:
    """One stable-style gameplay mod icon derived from the replay bitfield."""

    acronym: str
    asset_name: str | None


# Stable's Player iterates the legacy flags in enum-value order and loads
# ``selection-mod-{enum name lowercased}``.  V2/MR are newer than that asset
# contract, so they intentionally retain generated fallbacks.
_LEGACY_MOD_ICON_ORDER: tuple[tuple[Mod, str, str | None], ...] = (
    (Mod.NF, "NF", "nofail"),
    (Mod.EZ, "EZ", "easy"),
    (Mod.HD, "HD", "hidden"),
    (Mod.HR, "HR", "hardrock"),
    (Mod.SD, "SD", "suddendeath"),
    (Mod.DT, "DT", "doubletime"),
    (Mod.HT, "HT", "halftime"),
    (Mod.NC, "NC", "nightcore"),
    (Mod.FL, "FL", "flashlight"),
    (Mod.AT, "AT", "autoplay"),
    (Mod.PF, "PF", "perfect"),
    (Mod.K4, "4K", "key4"),
    (Mod.K5, "5K", "key5"),
    (Mod.K6, "6K", "key6"),
    (Mod.K7, "7K", "key7"),
    (Mod.K8, "8K", "key8"),
    (Mod.FI, "FI", "fadein"),
    (Mod.RD, "RD", "random"),
    (Mod.K9, "9K", "key9"),
    (Mod.KC, "KC", "keycoop"),
    (Mod.K1, "1K", "key1"),
    (Mod.K3, "3K", "key3"),
    (Mod.K2, "2K", "key2"),
    (Mod.V2, "V2", None),
    (Mod.MR, "MR", None),
)

LEGACY_MOD_SKIN_ASSET_NAMES: tuple[str, ...] = tuple(
    asset_name
    for _bit, _acronym, asset_name in _LEGACY_MOD_ICON_ORDER
    if asset_name is not None
)


def legacy_mod_icons(mods_bitfield: int) -> tuple[LegacyModIcon, ...]:
    """Return only ACTUAL replay mods in stable gameplay display order.

    In particular this never fabricates a key-count icon from beatmap metadata.
    Nightcore and Perfect suppress their implied lower-tier icons exactly as
    stable's ``Player`` does.
    """
    has_nc = bool(mods_bitfield & Mod.NC)
    has_pf = bool(mods_bitfield & Mod.PF)
    out: list[LegacyModIcon] = []
    for bit, acronym, asset_name in _LEGACY_MOD_ICON_ORDER:
        if not (mods_bitfield & bit):
            continue
        if bit == Mod.DT and has_nc:
            continue
        if bit == Mod.SD and has_pf:
            continue
        out.append(LegacyModIcon(acronym=acronym, asset_name=asset_name))
    return tuple(out)


def actual_mod_acronyms(mods_bitfield: int) -> tuple[str, ...]:
    """Return display acronyms for replay-authored mods only.

    Unlike :func:`mod_acronyms`, this never fabricates a native key-count
    label. An explicit key mod such as Key4 remains visible.
    """
    return tuple(icon.acronym for icon in legacy_mod_icons(mods_bitfield))


@dataclass(frozen=True)
class ModResult:
    beatmap: BeatmapInfo
    audio_rate: float
    visual_mods: VisualMods
    warnings: tuple[str, ...] = ()
    # True when the speed mod also shifts PITCH with the rate (Nightcore).
    # Stable semantics: DT/HT are pitch-PRESERVING tempo changes (BASS FX
    # tempo); only NC plays the track resampled so pitch rises with speed.
    # Drives the atempo-vs-asetrate branch in encode.build_ffmpeg_cmd.
    audio_pitch: bool = False


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

    # Speed. NC implies the DT bit; only NC pitches the audio with the rate
    # (stable: DT/HT are pitch-preserving tempo changes, NC is a resample).
    if mods & Mod.DT or mods & Mod.NC:
        audio_rate = 1.5
    elif mods & Mod.HT:
        audio_rate = 0.75
    else:
        audio_rate = 1.0
    audio_pitch = bool(mods & Mod.NC)

    # Apply speed to note times.
    notes = _rescale_times(beatmap.notes, audio_rate)
    total = int(beatmap.total_duration_ms / audio_rate)
    # Break periods ride the same clock (MAP → REAL/video time) so the
    # background dim envelope's break glides line up with the rescaled notes.
    if audio_rate != 1.0:
        breaks = tuple((int(a / audio_rate), int(b / audio_rate))
                       for a, b in getattr(beatmap, "breaks", ()))
    else:
        breaks = tuple(getattr(beatmap, "breaks", ()))

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

    modded = replace(beatmap, notes=tuple(notes), total_duration_ms=total,
                     breaks=breaks)
    return ModResult(
        beatmap=modded,
        audio_rate=audio_rate,
        visual_mods=visual,
        warnings=tuple(warnings),
        audio_pitch=audio_pitch,
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
