"""Pure port of TheAussie's horizontal Mania HEM geometry and transforms.

Authority: replay/shared/hit-error-meter-{geometry,timeline}.js and the
Mania runtime's Perfect/Good/Meh window projection. Coordinates are local,
top-down HUD pixels; the painter alone converts these to OpenGL coordinates.
R3D bottom-anchors the whole component with the human-reviewed HUD margin.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

REFERENCE_HEIGHT = 720.0
BOTTOM_MARGIN = 8.0
ICON_SIZE = 16.0
ICON_BAR_GAP = 6.0
BAR_LENGTH = 200.0
COLUMN_SIZE = 14.0
BAND_SIZE = 2.0
TICK_THICKNESS = 4.0
CENTRE_MARKER_SIZE = 8.0
CHEVRON_SIZE = 8.0
NATURAL_HEIGHT = COLUMN_SIZE + CHEVRON_SIZE
CHEVRON_STROKE_WIDTH = 2.0
EDGE_FADE_SIZE = 6.0
MARKER_MAX_CONCURRENT = 50
MARKER_LIFETIME_MS = 5100
MOVING_AVERAGE_MOVE_DURATION_MS = 800.0

BLUE = (0x66 / 255, 0xCC / 255, 1.0)
GREEN = (0x88 / 255, 0xB3 / 255, 0.0)
YELLOW = (1.0, 0xCC / 255, 0x22 / 255)
CENTRE_DARK_BLUE = (0x47 / 255, 0x8F / 255, 0xB3 / 255)
HIT_RESULT_COLOURS = {
    "geki": BLUE,
    "300": BLUE,
    "katu": GREEN,
    "100": GREEN,
    "50": YELLOW,
}


@dataclass(frozen=True)
class HitErrorMeterGeometry:
    scale: float
    component_x: float
    component_top: float
    natural_width: float
    natural_height: float
    axis_start: float
    axis_centre: float
    cross_centre: float
    early_icon_centre: tuple[float, float]
    late_icon_centre: tuple[float, float]

    def axis_point(self, position: float) -> float:
        return self.axis_start + max(0.0, min(1.0, position)) * BAR_LENGTH * self.scale

    def band_rect(self, fraction: float) -> tuple[float, float, float, float]:
        span = BAR_LENGTH * max(0.0, min(1.0, fraction)) * self.scale
        return (
            self.axis_centre - span / 2,
            self.cross_centre - BAND_SIZE * self.scale / 2,
            span,
            BAND_SIZE * self.scale,
        )

    def marker_segment(self, position: float, width_scale: float):
        axis = self.axis_point(position)
        span = COLUMN_SIZE * max(0.0, width_scale) * self.scale
        return ((axis, self.cross_centre - span / 2), (axis, self.cross_centre + span / 2))

    def chevron_points(self, position: float):
        axis = self.axis_point(position)
        tip = (COLUMN_SIZE + 1) * self.scale
        base = tip + (CHEVRON_SIZE - 2) * self.scale
        return ((axis - 4 * self.scale, base), (axis, tip), (axis + 4 * self.scale, base))


def hit_error_meter_geometry(render_width: int, render_height: int) -> HitErrorMeterGeometry:
    scale = render_height / REFERENCE_HEIGHT
    outset = ICON_SIZE + ICON_BAR_GAP
    return HitErrorMeterGeometry(
        scale=scale,
        component_x=(render_width - 1280 * scale) / 2 + 540 * scale,
        component_top=(REFERENCE_HEIGHT - NATURAL_HEIGHT - BOTTOM_MARGIN) * scale,
        natural_width=(2 * outset + BAR_LENGTH) * scale,
        natural_height=NATURAL_HEIGHT * scale,
        axis_start=outset * scale,
        axis_centre=(outset + BAR_LENGTH / 2) * scale,
        cross_centre=COLUMN_SIZE / 2 * scale,
        early_icon_centre=(ICON_SIZE / 2 * scale, COLUMN_SIZE / 2 * scale),
        late_icon_centre=(
            (outset + BAR_LENGTH + ICON_BAR_GAP + ICON_SIZE / 2) * scale,
            COLUMN_SIZE / 2 * scale,
        ),
    )


@dataclass(frozen=True)
class HitWindowBand:
    judgment: str
    window_ms: float
    relative_length: float
    colour: tuple[float, float, float]


def hit_window_bands(windows: tuple[float, ...]) -> tuple[HitWindowBand, ...]:
    """Perfect/Good/Meh -> the viewer's w300/w100/w50, narrowest first."""
    if len(windows) != 5 or not isfinite(windows[-1]) or windows[-1] <= 0:
        return ()
    return tuple(
        HitWindowBand(
            judgment,
            float(window),
            max(0.0, min(1.0, window / windows[-1])) if isfinite(window) else 0.0,
            colour,
        )
        for judgment, window, colour in (
            ("geki", windows[0], BLUE),
            ("katu", windows[2], GREEN),
            ("50", windows[4], YELLOW),
        )
    )


def hit_error_offset_position(offset_ms: float, max_hit_window: float) -> float:
    if not isfinite(offset_ms) or not isfinite(max_hit_window) or max_hit_window <= 0:
        return 0.5
    return max(0.0, min(1.0, (offset_ms / max_hit_window + 1.0) / 2.0))


def hit_error_edge_alpha(axis: float, span: float, fade_size: float) -> float:
    """Continuous linear alpha at the widest band's outer tips only."""
    fade = min(fade_size, span / 2)
    if span <= 0 or axis < 0 or axis > span:
        return 0.0
    return min(1.0, axis / fade, (span - axis) / fade) if fade > 0 else 1.0


@dataclass(frozen=True)
class HitErrorTickState:
    alpha: float
    width_fraction: float


def hit_error_tick_state(age_ms: float) -> HitErrorTickState:
    if age_ms < 0 or age_ms >= MARKER_LIFETIME_MS:
        return HitErrorTickState(0.0, 0.0)
    if age_ms < 100:
        eased = 1 - (1 - age_ms / 100) ** 5
        return HitErrorTickState(0.6 * eased, eased)
    progress = (age_ms - 100) / 5000
    return HitErrorTickState(0.6 * (1 - progress), 1 - progress**5)


def next_hit_error_ema(old_average: float, offset_ms: float) -> float:
    return old_average * 0.9 + offset_ms * 0.1


@dataclass(frozen=True)
class HitErrorChevronTransition:
    time_ms: float
    start_position: float
    target_position: float


def hit_error_chevron_position(
    transition: HitErrorChevronTransition | None, time_ms: float
) -> float | None:
    if transition is None:
        return None
    progress = max(0.0, min(1.0, (time_ms - transition.time_ms) / MOVING_AVERAGE_MOVE_DURATION_MS))
    eased = 1 - (1 - progress) ** 5
    return (
        transition.start_position + (transition.target_position - transition.start_position) * eased
    )


def next_hit_error_chevron_transition(
    previous: HitErrorChevronTransition | None, time_ms: float, ema_ms: float, max_hit_window: float
) -> HitErrorChevronTransition:
    current = hit_error_chevron_position(previous, time_ms)
    return HitErrorChevronTransition(
        time_ms,
        0.5 if current is None else current,
        hit_error_offset_position(ema_ms, max_hit_window),
    )
