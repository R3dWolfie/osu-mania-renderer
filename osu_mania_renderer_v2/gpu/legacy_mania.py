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

    # Enum.Parse permits unnamed integers. Preserve raw intent; the draw
    # switch (not the parser/resolver) supplies RepeatBottom for unknown values.
    return configured if configured is not None else LEGACY_NOTE_BODY_REPEAT_BOTTOM


def legacy_key_flip(section, column: int, *, upside_down: bool) -> bool:
    """SkinMania.GetFlipVertical(Key): column > global > true, no version gate."""
    if not upside_down:
        return False
    if section is None:
        return True
    value = section.key_flip_by_column.get(column)
    if value is None:
        value = section.key_flip_when_upside_down
    return value if value is not None else True


def legacy_note_flip(section, column: int, part: str = "", *,
                     upside_down: bool, legacy_version: float) -> bool:
    """Final source UV sign from HC/LN, including the rear's logical inversion."""
    part = part.upper()
    default = True
    if part == "H":
        default = legacy_note_flip(section, column, upside_down=upside_down,
                                   legacy_version=legacy_version)
    resolved = False
    if upside_down and legacy_version >= 2.5:
        value = section.note_flip_by_column.get((column, part)) if section else None
        if value is None and section:
            value = section.note_flip_when_upside_down.get(part)
        resolved = value if value is not None else default
    return not resolved if part == "T" else resolved


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


def legacy_clip_y_segment(
    segment: LegacyHoldBodySegment,
    *,
    minimum_y: float | None = None,
    maximum_y: float | None = None,
) -> LegacyHoldBodySegment | None:
    """Intersect a mask with an existing quad, retaining its source-V slope."""
    if segment.height <= 0:
        return None
    bottom = max(segment.y, minimum_y) if minimum_y is not None else segment.y
    top = min(segment.y + segment.height, maximum_y) if maximum_y is not None else segment.y + segment.height
    if top <= bottom:
        return None
    source_delta = segment.source_top - segment.source_bottom
    return LegacyHoldBodySegment(
        bottom, top - bottom,
        segment.source_bottom + source_delta * (bottom - segment.y) / segment.height,
        segment.source_bottom + source_delta * (top - segment.y) / segment.height,
    )


def legacy_hold_body_segments(
    body_y: float,
    body_height: float,
    tile_height: float,
    style: int,
    *,
    upside_down: bool = False,
    flip_vertical: bool | None = None,
    height_ratio: float = 1.0,
) -> tuple[LegacyHoldBodySegment, ...]:
    """Lay out stable's source phase, then mirror into GL Y-up for upscroll.

    RepeatBottom/default sets DrawTop=0: image row zero starts at the tail,
    so a short downscroll body uses the TOP of the source. RepeatTop aligns
    its last source row to the head. Tile height follows native design Y
    scale (stage height / 768), independently of the column's X stretch.
    """
    if body_height <= 0 or tile_height <= 0:
        return ()

    if flip_vertical is None:
        flip_vertical = upside_down
    if style == LEGACY_NOTE_BODY_STRETCH:
        bounds = (1.0, 0.0) if flip_vertical else (0.0, 1.0)
        return (LegacyHoldBodySegment(body_y, body_height, *bounds),)

    native_height = tile_height / height_ratio
    source_length = body_height / height_ratio
    # HM reads DrawHeight / HeightRatio before setting the manual wrapped
    # DrawHeight. Integer DrawTop truncation occurs in source space.
    if style == LEGACY_NOTE_BODY_REPEAT_TOP:
        draw_top = int(native_height - source_length)
    elif style == LEGACY_NOTE_BODY_REPEAT_TOP_AND_BOTTOM:
        draw_top = int((native_height - source_length) / 2.0)
    else:
        draw_top = 0
    # GL bottom->top runs opposite source rows before a vertical flip.
    phase = (-(draw_top + source_length) * height_ratio) % tile_height

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
    if upside_down:
        segments = list(LegacyHoldBodySegment(
            y=body_y + body_height - (segment.y - body_y) - segment.height,
            height=segment.height,
            source_bottom=segment.source_top,
            source_top=segment.source_bottom,
        ) for segment in reversed(segments))
    if flip_vertical != upside_down:
        segments = [LegacyHoldBodySegment(s.y, s.height, 1 - s.source_bottom,
                                           1 - s.source_top) for s in segments]
    return tuple(segments)


def legacy_hold_body_frame(
    *,
    active: bool,
    elapsed_active_ms: float,
    frame_count: int,
) -> int:
    """Select a 30 ms body frame from its accumulated active clock.

    The caller pauses that clock while inactive; pAnimation retains its frame.
    """
    if frame_count <= 1 or elapsed_active_ms <= 0:
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
    """Resolve stable's Mania stage-light frame rate."""
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
    press_age_ms: float | None = None,
    release_duration_ms: float = 250.0,
) -> LegacyStageLightPresentation:
    """Calculate presentation from immutable replay-derived input evidence."""
    # CI resets animation on every key-down. Exact replay ages also preserve
    # subframe press/release pairs and deterministic direct seeks.
    elapsed = max(0.0, press_age_ms or 0.0)
    frame = int(elapsed * fps / 1000.0) % frame_count if frame_count > 1 and fps > 0 else 0

    if held:
        return LegacyStageLightPresentation(True, frame, 1.0, 1.0)

    if (
        release_age_ms is None
        or release_age_ms < 0
        or release_age_ms >= release_duration_ms
    ):
        return LegacyStageLightPresentation(False, 0, 0.0, 0.0)

    amount = 1.0 - release_age_ms / release_duration_ms
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
