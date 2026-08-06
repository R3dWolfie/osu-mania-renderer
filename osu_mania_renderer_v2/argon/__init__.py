"""Argon default-skin logic for mania (osu!lazer's default look).

Whether the Argon default applies (only when no user skin supplies mania
content) + the per-column accent palette, ported from lazer's
ManiaArgonSkinTransformer. Consolidated here from ``render/element_common.py``
so mania's ``argon/`` mirrors taiko's and catch's ``argon/`` modules. The
definitions are byte-identical to the previous inline ones.
"""
from __future__ import annotations

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
