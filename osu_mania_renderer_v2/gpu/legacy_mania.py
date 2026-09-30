"""Pure legacy Mania presentation helpers shared by both draw paths."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

LEGACY_NOTE_BODY_STRETCH = 0
LEGACY_NOTE_BODY_REPEAT_TOP = 2
LEGACY_NOTE_BODY_REPEAT_BOTTOM = 3
LEGACY_NOTE_BODY_REPEAT_TOP_AND_BOTTOM = 4
_VALID_NOTE_BODY_STYLES = {
    LEGACY_NOTE_BODY_STRETCH,
    LEGACY_NOTE_BODY_REPEAT_TOP,
    LEGACY_NOTE_BODY_REPEAT_BOTTOM,
    LEGACY_NOTE_BODY_REPEAT_TOP_AND_BOTTOM,
}

LEGACY_STAGE_LIGHT_RELEASE_MS = 250.0
LEGACY_HOLD_BODY_FRAME_MS = 30.0


class _ManiaBodyStyleConfig(Protocol):
    note_body_style: int | None
    note_body_style_by_column: dict[int, int]


class _ManiaStageLightConfig(Protocol):
    light_frame_per_second: int | None


def legacy_doubled_alpha_colour(
    colour: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    """Apply stable/lazer's legacy colour alpha compatibility quirk.

    ``ApplyWithDoubledAlpha`` applies authored alpha to both the drawable and
    its colour. RGB is not altered, including when authored alpha is zero.
    """
    red, green, blue, alpha = colour
    normalised_alpha = alpha / 255.0
    return (
        red / 255.0,
        green / 255.0,
        blue / 255.0,
        normalised_alpha * normalised_alpha,
    )


def legacy_disallow_zero_alpha_colour(
    colour: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    """Normalise a post-construction legacy tint, forcing zero alpha to one."""
    red, green, blue, alpha = colour
    return (
        red / 255.0,
        green / 255.0,
        blue / 255.0,
        alpha / 255.0 if alpha else 1.0,
    )


def legacy_note_body_style(
    section: _ManiaBodyStyleConfig | None,
    column: int,
    legacy_version: float,
) -> int:
    """Resolve stable's versioned per-column/global hold-body style."""
    if legacy_version < 2.5:
        return LEGACY_NOTE_BODY_STRETCH

    configured: int | None = None
    if section is not None:
        # NoteBodyStyleN is zero-indexed in stable (unlike ColourN).
        configured = section.note_body_style_by_column.get(column)
        if configured is None:
            configured = section.note_body_style

    # Modern legacy skins default to RepeatBottom. Stable's undocumented
    # value 1 and all malformed/out-of-range values use the same deterministic
    # compatibility fallback rather than crashing or silently stretching.
    if configured not in _VALID_NOTE_BODY_STYLES:
        return LEGACY_NOTE_BODY_REPEAT_BOTTOM
    return configured


@dataclass(frozen=True)
class LegacyHoldBodySegment:
    """One destination/source slice of a repeated hold-body texture.

    Source fractions are measured bottom-to-top, matching the renderer's GL
    coordinate convention. A partial segment therefore crops the source UVs
    while retaining the natural destination scale.
    """

    y: float
    height: float
    source_bottom: float
    source_top: float


def legacy_hold_body_segments(
    body_y: float,
    body_height: float,
    tile_height: float,
    style: int,
) -> tuple[LegacyHoldBodySegment, ...]:
    """Lay out stable-aligned repeat slices without squashing partial tiles."""
    if body_height <= 0 or tile_height <= 0:
        return ()

    if style == LEGACY_NOTE_BODY_STRETCH:
        return (LegacyHoldBodySegment(body_y, body_height, 0.0, 1.0),)

    if style == LEGACY_NOTE_BODY_REPEAT_TOP:
        phase = (-body_height) % tile_height
    elif style == LEGACY_NOTE_BODY_REPEAT_TOP_AND_BOTTOM:
        phase = ((tile_height - body_height) / 2.0) % tile_height
    else:
        # RepeatBottom, including the compatibility fallback.
        phase = 0.0

    segments: list[LegacyHoldBodySegment] = []
    cursor = 0.0
    source_offset = phase
    epsilon = 1e-9
    while cursor < body_height - epsilon:
        segment_height = min(tile_height - source_offset, body_height - cursor)
        segments.append(LegacyHoldBodySegment(
            y=body_y + cursor,
            height=segment_height,
            source_bottom=source_offset / tile_height,
            source_top=(source_offset + segment_height) / tile_height,
        ))
        cursor += segment_height
        source_offset = 0.0
    return tuple(segments)


def legacy_hold_body_frame(
    *,
    active: bool,
    elapsed_active_ms: float,
    frame_count: int,
) -> int:
    """Select a 30 ms legacy body frame, resetting to zero when inactive."""
    if not active or frame_count <= 1 or elapsed_active_ms <= 0:
        return 0
    return int(elapsed_active_ms / LEGACY_HOLD_BODY_FRAME_MS) % frame_count


@dataclass(frozen=True)
class LegacyStageLightPresentation:
    visible: bool
    frame: int
    alpha: float
    vertical_scale: float


def legacy_stage_light_fps(
    section: _ManiaStageLightConfig | None,
) -> float:
    """Resolve current lazer's legacy Mania stage-light frame rate."""
    if section is None or section.light_frame_per_second is None:
        return 60.0
    value = section.light_frame_per_second
    return float(value if value > 0 else 24)


def legacy_stage_light_presentation(
    *,
    held: bool,
    release_age_ms: float | None,
    time_ms: float,
    frame_count: int,
    fps: float,
) -> LegacyStageLightPresentation:
    """Calculate presentation from immutable replay-derived input evidence."""
    # lazer's legacy animation starts against the current gameplay clock and
    # keeps running whether the light is hidden, held, or fading out. Input
    # transitions affect only opacity and vertical scale.
    frame = (
        int(time_ms * fps / 1000.0) % frame_count
        if frame_count > 1 and fps > 0
        else 0
    )

    if held:
        return LegacyStageLightPresentation(True, frame, 1.0, 1.0)

    if (
        release_age_ms is None
        or release_age_ms < 0
        or release_age_ms >= LEGACY_STAGE_LIGHT_RELEASE_MS
    ):
        return LegacyStageLightPresentation(False, 0, 0.0, 0.0)

    amount = 1.0 - release_age_ms / LEGACY_STAGE_LIGHT_RELEASE_MS
    return LegacyStageLightPresentation(True, frame, amount, amount)


@dataclass(frozen=True)
class LegacyStageLightGeometry:
    x: float
    y: float
    width: float
    height: float
    anchor_y: float


def legacy_stage_light_geometry(
    *,
    column_x: float,
    column_width: float,
    native_size: tuple[float, float],
    light_position: float,
    render_height: float,
    upside_down: bool,
    vertical_scale: float,
) -> LegacyStageLightGeometry:
    """Keep native legacy Y scale and collapse toward the LightPosition edge."""
    _native_width, native_height = native_size
    full_height = (
        native_height * render_height / 768.0
        if native_height > 0
        else 0.0
    )
    height = full_height * max(0.0, min(1.0, vertical_scale))
    position = 480.0 - light_position if upside_down else light_position
    anchor_y = render_height - position * render_height / 480.0
    y = anchor_y - height if upside_down else anchor_y
    return LegacyStageLightGeometry(
        x=column_x,
        y=y,
        width=column_width,
        height=height,
        anchor_y=anchor_y,
    )
