"""Shared constants + helpers for wiki render elements, decoupled from
FrameRenderer. Functions take the FrameContext so elements stay
self-contained while reading the parsed [Mania] section / skin.ini.
Values mirror the legacy FrameRenderer exactly (byte-identical migration).
"""
from __future__ import annotations

RECEPTOR_HEIGHT_REL_COL = 1.0
STAGE_LIGHT_DURATION_MS = 200
HIT_LIGHT_DURATION_MS = 320

# ── Mod icons (lazer ModIcon / OsuColour.ForModType) ─────────────────────
# Flat-topped hexagon tinted by the mod's ModType; acronym drawn in fill×0.1.
_MOD_TYPE_FILL = {
    "reduction":  (0.698, 1.000, 0.400),  # #b2ff66 DifficultyReduction
    "increase":   (1.000, 0.400, 0.400),  # #ff6666 DifficultyIncrease
    "conversion": (0.549, 0.400, 1.000),  # #8c66ff Conversion
    "automation": (0.400, 0.800, 1.000),  # #66ccff Automation
    "fun":        (1.000, 0.400, 0.671),  # #ff66ab Fun
    "system":     (1.000, 0.800, 0.133),  # #ffcc22 System
}
# osu! mod acronym → ModType. Unknown acronyms fall back to conversion-purple.
_MOD_ACRONYM_TYPE = {
    # DifficultyReduction
    "EZ": "reduction", "NF": "reduction", "HT": "reduction", "DC": "reduction",
    # DifficultyIncrease
    "HR": "increase", "SD": "increase", "PF": "increase", "DT": "increase",
    "NC": "increase", "HD": "increase", "FI": "increase", "FL": "increase",
    "AC": "increase",
    # Conversion (mania-relevant)
    "MR": "conversion", "RD": "conversion", "DS": "conversion", "IN": "conversion",
    "HO": "conversion", "CS": "conversion", "CO": "conversion",
    # Automation
    "AT": "automation", "CN": "automation", "SO": "automation", "RX": "automation",
    "AP": "automation",
    # Fun
    "WU": "fun", "WD": "fun", "MU": "fun", "AS": "fun", "NS": "fun", "BR": "fun",
    "TC": "fun", "BU": "fun", "SY": "fun", "DP": "fun", "MG": "fun", "RP": "fun",
    "FR": "fun", "AD": "fun",
    # System
    "TD": "system",
}


def mod_fill_colour(acronym: str) -> tuple[float, float, float]:
    """ModType fill colour for a mod acronym (lazer OsuColour.ForModType)."""
    return _MOD_TYPE_FILL[_MOD_ACRONYM_TYPE.get(acronym, "conversion")]


def is_keycount_acronym(acronym: str) -> bool:
    """True for the mania key-count pseudo-mod (1K..18K). lazer does NOT show
    it as a gameplay mod icon for native mania maps."""
    return len(acronym) >= 2 and acronym[-1] == "K" and acronym[:-1].isdigit()

# Argon default-skin logic (accent palette + is_argon_default / argon_accent /
# _skin_provides_mania) moved to the argon/ module — parity with taiko/catch.
# Import from osu_mania_renderer_v2.argon.

# Per-judgment hit-light RGB (0-255).
JUDGMENT_LIGHT: dict[str, tuple[int, int, int]] = {
    "geki": (150, 215, 255),   # 320 light blue
    "300":  (80, 150, 240),    # 300 blue
    "katu": (100, 220, 130),   # 200 green
    "100":  (240, 220, 90),    # 100 yellow
    "50":   (240, 160, 80),    # 50  orange
}


def note_anim_fps(ctx, frame_count: int) -> float:
    """FPS for tap/hold-head/tail animations: [General] AnimationFramerate
    (>0), -1 → derive (fps == frame_count), else 60."""
    si = ctx.fr.skin_ini
    if si is not None and si.animation_framerate:
        af = si.animation_framerate
        if af > 0:
            return float(af)
        if af == -1 and frame_count > 1:
            return float(frame_count)
    return 60.0


def stage_light_fps(ctx, frame_count: int) -> float:
    """[Mania] LightFramePerSecond (>0) → -1 derive → [General]
    AnimationFramerate → 60."""
    section = ctx.mania_section
    if section is not None and section.light_frame_per_second is not None:
        v = section.light_frame_per_second
        if v > 0:
            return float(v)
        if v == -1 and frame_count > 1:
            return float(frame_count)
    si = ctx.fr.skin_ini
    if si is not None and si.animation_framerate:
        af = si.animation_framerate
        if af > 0:
            return float(af)
        if af == -1 and frame_count > 1:
            return float(frame_count)
    return 60.0


def stage_light_tint(ctx, col: int) -> tuple[float, float, float]:
    """ColourLight{N} (1-indexed → 0-indexed) → white."""
    section = ctx.mania_section
    if section is not None:
        rgb = section.colour_light.get(col + 1)
        if rgb is None:
            rgb = section.colour_light.get(col)
        if rgb is not None:
            return rgb[0] / 255, rgb[1] / 255, rgb[2] / 255
    return 1.0, 1.0, 1.0
