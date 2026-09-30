"""Pure geometry helpers for legacy osu!mania note sprites."""
from __future__ import annotations


def legacy_note_draw_y(
    anchor_y: int | float,
    sprite_height: int | float,
    *,
    upside_down: bool,
    is_tail: bool = False,
) -> int:
    """Return the GL lower-Y coordinate for an edge-anchored legacy note."""
    # Source authority:
    # osu.Game.Rulesets.Mania/Skinning/Legacy/LegacyNotePiece.cs
    # osu.Game.Rulesets.Mania/Skinning/Legacy/LegacyHoldNoteTailPiece.cs
    # Tap/head use the scrolling direction's edge anchor; tail intentionally
    # uses the opposite edge.
    top_edge_is_anchor = upside_down != is_tail
    draw_y = anchor_y - sprite_height if top_edge_is_anchor else anchor_y
    return int(draw_y)
