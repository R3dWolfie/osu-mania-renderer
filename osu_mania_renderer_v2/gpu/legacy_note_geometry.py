"""Pure geometry helpers for legacy osu!mania note sprites."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


def legacy_note_height(
    aspect: float,
    *,
    minimum_column_width: float,
    configured_width: float | None,
    render_height: float,
) -> int:
    """Stable/lazer scale note Y using WidthForNoteHeightScale or min width.

    The configured width is in skin.ini's 480-reference units. Source aspect
    is ScaleAdjust invariant; X continues to fill the actual column width.
    Applies to taps and both hold caps, never to the hold body's native Y.
    """
    width = minimum_column_width
    if configured_width is not None and isfinite(configured_width) and configured_width > 0:
        width = configured_width * render_height / 480.0
    return max(1, int(width / aspect if aspect > 0 else width))


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


@dataclass(frozen=True)
class LegacyHoldGeometry:
    """Edge-anchored cap rects and their centre-to-centre body interval."""

    head_draw_y: int
    tail_draw_y: int
    body_y: float
    body_height: float
    clip_min_y: float | None = None
    clip_max_y: float | None = None


def legacy_hold_geometry(
    y_head: int | float,
    y_tail: int | float,
    head_height: int | float,
    tail_height: int | float,
    *,
    upside_down: bool,
    body_head_y: int | float | None = None,
    clip_head_y: int | float | None = None,
) -> LegacyHoldGeometry:
    """Combine legacy cap edge anchors with lazer's half-cap body overlap."""
    head_draw_y = legacy_note_draw_y(
        y_head, head_height, upside_down=upside_down,
    )
    tail_draw_y = legacy_note_draw_y(
        y_tail, tail_height, upside_down=upside_down, is_tail=True,
    )
    head_centre = head_draw_y + head_height / 2.0
    if body_head_y is not None:
        head_centre = legacy_note_draw_y(
            body_head_y, head_height, upside_down=upside_down,
        ) + head_height / 2.0
    tail_centre = tail_draw_y + tail_height / 2.0
    boundary = None
    if clip_head_y is not None:
        # Stable FreezeNote: HitPosition - SpriteHeight(head)/2, converted
        # from top-down reference coordinates to this GL Y-up direction.
        boundary = clip_head_y + (-head_height / 2.0 if upside_down else head_height / 2.0)
    return LegacyHoldGeometry(
        head_draw_y=head_draw_y,
        tail_draw_y=tail_draw_y,
        body_y=min(head_centre, tail_centre),
        body_height=abs(tail_centre - head_centre),
        clip_min_y=boundary if not upside_down else None,
        clip_max_y=boundary if upside_down else None,
    )
