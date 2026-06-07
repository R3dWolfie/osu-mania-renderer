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

# ── Argon default skin (osu!lazer's default) ─────────────────────────────
# Per-column accent palette + mapping, ported from lazer's
# ManiaArgonSkinTransformer. Used to tint the bundled Argon default notes /
# columns / receptors per column. User-uploaded skins keep their own look.
ARGON_YELLOW = (255, 197, 40)
ARGON_ORANGE = (252, 109, 1)
ARGON_PINK = (213, 35, 90)
ARGON_PURPLE = (203, 60, 236)
ARGON_CYAN = (72, 198, 255)
ARGON_GREEN = (100, 192, 92)
ARGON_SPECIAL = (169, 106, 255)
_ARGON_CYCLE = (ARGON_YELLOW, ARGON_ORANGE, ARGON_PINK, ARGON_PURPLE, ARGON_CYAN, ARGON_GREEN)
# Explicit per-keycount tables (from lazer source) for the common modes.
_ARGON_BY_KEYS: dict[int, tuple[tuple[int, int, int], ...]] = {
    4: (ARGON_YELLOW, ARGON_ORANGE, ARGON_PINK, ARGON_PURPLE),
    7: (ARGON_PINK, ARGON_ORANGE, ARGON_PINK, ARGON_SPECIAL,
        ARGON_PINK, ARGON_ORANGE, ARGON_PINK),
}


def _skin_provides_mania(ctx) -> bool:
    """True if the USER skin supplies any mania content — a [Mania] section in
    skin.ini, or any user/beatmap mania sprite (note/key/stage). Cached per
    render."""
    cache = ctx.persistent.setdefault("_skinmeta", {})
    if "provides" in cache:
        return cache["provides"]
    provides = ctx.mania_section is not None
    if not provides:
        atlas = ctx.atlas
        for kind in ("note_tap", "note_hold_head", "note_hold_body", "receptor_off"):
            if any(atlas.column_source(kind, c) in ("user", "beatmap")
                   for c in range(ctx.key_count)):
                provides = True
                break
    if not provides:
        atlas = ctx.atlas
        provides = any(
            atlas.global_source(g) in ("user", "beatmap")
            for g in ("stage_left", "stage_right", "playfield_frame",
                      "stage_light", "hit_light")
        )
    cache["provides"] = provides
    return provides


def is_argon_default(ctx, col: int) -> bool:
    """The Argon default look applies ONLY when NO user skin provides mania
    content. A selected legacy skin — even one missing some sprites (Chitanda
    ships keys but no notes; FREEDOM DiVE references files it doesn't include)
    — renders legacy with the CLASSIC bundle fallbacks and its OWN elements,
    never Argon. Mirrors lazer's skin → DefaultLegacySkin chain (Argon is only
    the no-skin default). Skin-level, so it's uniform across columns."""
    return not _skin_provides_mania(ctx)


def argon_accent(col: int, key_count: int) -> tuple[int, int, int]:
    """Argon per-column accent RGB. Explicit table for 4K/7K; else cycle the
    6 base colours with the special-purple on an odd-key centre column."""
    table = _ARGON_BY_KEYS.get(key_count)
    if table is not None and 0 <= col < len(table):
        return table[col]
    if key_count % 2 == 1 and col == key_count // 2:
        return ARGON_SPECIAL
    return _ARGON_CYCLE[col % len(_ARGON_CYCLE)]

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
