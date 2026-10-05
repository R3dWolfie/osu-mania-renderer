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
) -> float:
    """Return the GL lower-Y coordinate for an edge-anchored legacy note."""
    # HC and LN share FlipOrigin(BottomCentre). TextureGlSingle.Draw uses
    # abs(scale) for the rectangle and a negative scale ONLY to invert UVs.
    # Rear art has a different sign, not a different anchor.
    draw_y = anchor_y - sprite_height if upside_down else anchor_y
    return draw_y


@dataclass(frozen=True)
class LegacyHoldGeometry:
    """Edge-anchored cap rects and their centre-to-centre body interval."""

    head_draw_y: float
    tail_draw_y: float
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
    body_head_height: float | None = None,
    clip_head_height: float | None = None,
    full_body_length: float | None = None,
) -> LegacyHoldGeometry:
    """HM places the body at the scrolling head centre, using full note length."""
    head_draw_y = legacy_note_draw_y(
        y_head, head_height, upside_down=upside_down,
    )
    tail_draw_y = legacy_note_draw_y(
        y_tail, tail_height, upside_down=upside_down, is_tail=True,
    )
    head_centre = head_draw_y + head_height / 2.0
    if body_head_y is not None:
        body_height = head_height if body_head_height is None else body_head_height
        head_centre = legacy_note_draw_y(
            body_head_y, body_height, upside_down=upside_down,
        ) + body_height / 2.0
    scrolling_head = y_head if body_head_y is None else body_head_y
    body_length = abs(y_tail - scrolling_head) if full_body_length is None else full_body_length
    body_y = head_centre - body_length if upside_down else head_centre
    boundary = None
    if clip_head_y is not None:
        # Stable FreezeNote: HitPosition - SpriteHeight(head)/2, converted
        # from top-down reference coordinates to this GL Y-up direction.
        clip_height = head_height if clip_head_height is None else clip_head_height
        boundary = clip_head_y + (-clip_height / 2.0 if upside_down else clip_height / 2.0)
    return LegacyHoldGeometry(
        head_draw_y=head_draw_y,
        tail_draw_y=tail_draw_y,
        body_y=body_y,
        body_height=body_length,
        clip_min_y=boundary if not upside_down else None,
        clip_max_y=boundary if upside_down else None,
    )
