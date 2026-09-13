"""HUD elements — lazer-faithful (per r3drenderer_lazer_fidelity).

Health = the legacy scorebar (scorebar-bg/colour/marker, top-left) when the
skin ships it, else the source-shaped Argon health path. Score/accuracy use
the skin score font. Argon uses the shared dual edge hit-error meters and a
separate website-controlled UR readout; obsolete R3D timing chrome stays off.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import moderngl

from osu_mania_renderer_v2.beatmap.mods import actual_mod_acronyms
from osu_mania_renderer_v2.gpu.argon_health import (
    argon_health_path_geometry,
    argon_health_path_renderer,
)
from osu_mania_renderer_v2.gpu.argon_wedge import (
    argon_wedge_geometry,
    argon_wedge_renderer,
)
from osu_mania_renderer_v2.wiki_elements._common import (
    is_argon_default,
    mod_fill_colour,
)

# Argon HUD source geometry. osu!lazer authors this overlay on its 1280x720
# gameplay surface; arbitrary 16:9 outputs scale by render_height / 720.
ARGON_REFERENCE_WIDTH = 1280.0
ARGON_REFERENCE_HEIGHT = 720.0
ARGON_HEALTH_POSITION = (50.0, 20.0)
ARGON_HEALTH_WIDTH = 300.0
ARGON_HEALTH_BAR_HEIGHT = 30.0
ARGON_HEALTH_MAIN_PATH_RADIUS = 10.0
ARGON_HEALTH_GLOW_PATH_RADIUS = 40.0
ARGON_HEALTH_MAIN_GLOW_PORTION = 0.6
ARGON_HEALTH_GLOW_PORTION = (
    ARGON_HEALTH_GLOW_PATH_RADIUS
    - ARGON_HEALTH_MAIN_PATH_RADIUS * (1.0 - ARGON_HEALTH_MAIN_GLOW_PORTION)
) / ARGON_HEALTH_GLOW_PATH_RADIUS
ARGON_HEALTH_GLOW_PADDING = (
    ARGON_HEALTH_MAIN_PATH_RADIUS - ARGON_HEALTH_GLOW_PATH_RADIUS
)
ARGON_HEALTH_CONNECTOR = (0.0, 30.0, 45.0, 3.0)
ARGON_SCORE_POSITION = (250.0, 50.0)
ARGON_ACCURACY_MARGIN = (20.0, 20.0)
ARGON_PP_POSITION = (20.0, 72.0)
ARGON_PP_SCALE = 0.8

# Shared BarHitErrorMeter constants from osu!lazer/the audited Standard HUD.
ARGON_HEM_ICON_SIZE = 16.0
ARGON_HEM_ICON_BAR_GAP = 6.0
ARGON_HEM_BAR_LENGTH = 200.0
ARGON_HEM_COLUMN_SIZE = 14.0
ARGON_HEM_BAND_SIZE = 2.0
ARGON_HEM_TICK_THICKNESS = 4.0
ARGON_HEM_CENTRE_MARKER_SIZE = 8.0
ARGON_HEM_CHEVRON_SIZE = 8.0
ARGON_HEM_CHEVRON_STROKE = 2.0
ARGON_HEM_EDGE_FADE_SIZE = 6.0


@dataclass(frozen=True)
class ArgonHudGeometry:
    scale: float
    health_rect: tuple[float, float, float, float]
    health_bar_height: float
    health_main_radius: float
    health_glow_radius: float
    health_glow_rect: tuple[float, float, float, float]
    health_path_points: tuple[tuple[float, float], ...]
    connector_rect: tuple[float, float, float, float]
    score_anchor: tuple[float, float]
    accuracy_anchor: tuple[float, float]
    pp_anchor: tuple[float, float]


@dataclass(frozen=True)
class ArgonHitErrorMeterInstance:
    left: float
    top: float
    mirrored: bool


@dataclass(frozen=True)
class ArgonHitErrorMeterGeometry:
    scale: float
    width: float
    height: float
    axis_start: float
    axis_centre: float
    cross_centre: float
    bar_length: float
    column_size: float
    band_size: float
    tick_thickness: float
    centre_marker_size: float
    chevron_size: float
    chevron_stroke: float
    edge_fade_size: float
    instances: tuple[ArgonHitErrorMeterInstance, ArgonHitErrorMeterInstance]


@dataclass(frozen=True)
class HudComponentVisibility:
    health: bool
    hit_error_meter: bool
    unstable_rate: bool
    score: bool
    mods: bool
    scoreboard: bool
    performance_points: bool


def hud_component_visibility(options) -> HudComponentVisibility:
    """Resolve independent website HUD settings plus the global HUD gate."""
    hud_on = getattr(options, "hud_opacity", 1.0) > 0.0
    return HudComponentVisibility(
        health=hud_on and getattr(options, "show_hp_bar", True),
        hit_error_meter=(
            hud_on and getattr(options, "show_hit_error_meter", True)
        ),
        unstable_rate=(
            hud_on
            and getattr(options, "show_unstable_rate", True)
            and getattr(options, "show_ur_bar", True)
        ),
        score=hud_on and getattr(options, "show_score", True),
        mods=hud_on and getattr(options, "show_mods", True),
        scoreboard=hud_on and getattr(options, "show_scoreboard", True),
        performance_points=(
            hud_on and getattr(options, "show_pp_counter", False)
        ),
    )


def argon_hud_geometry(render_width: int, render_height: int) -> ArgonHudGeometry:
    """Resolve source-backed Argon HUD anchors in output top-left pixels."""
    scale = render_height / ARGON_REFERENCE_HEIGHT
    hx, hy = (value * scale for value in ARGON_HEALTH_POSITION)
    health_height = (
        ARGON_HEALTH_BAR_HEIGHT + ARGON_HEALTH_MAIN_PATH_RADIUS * 2.0
    ) * scale
    health_width = ARGON_HEALTH_WIDTH * scale
    glow_padding = ARGON_HEALTH_GLOW_PADDING * scale
    glow_rect = (
        hx + glow_padding,
        hy + glow_padding,
        health_width - glow_padding * 2.0,
        health_height - glow_padding * 2.0,
    )
    source_path = argon_health_path_geometry(
        ARGON_HEALTH_WIDTH,
        ARGON_HEALTH_BAR_HEIGHT + ARGON_HEALTH_MAIN_PATH_RADIUS * 2.0,
        ARGON_HEALTH_MAIN_PATH_RADIUS,
    )
    path = tuple(
        (hx + x * scale, hy + y * scale)
        for x, y in (
            source_path.start,
            source_path.top_arc_start,
            source_path.top_arc_end,
            source_path.slash_end,
            source_path.bottom_arc_end,
            source_path.end,
        )
    )
    return ArgonHudGeometry(
        scale=scale,
        health_rect=(hx, hy, health_width, health_height),
        health_bar_height=ARGON_HEALTH_BAR_HEIGHT * scale,
        health_main_radius=ARGON_HEALTH_MAIN_PATH_RADIUS * scale,
        health_glow_radius=ARGON_HEALTH_GLOW_PATH_RADIUS * scale,
        health_glow_rect=glow_rect,
        health_path_points=path,
        connector_rect=tuple(value * scale for value in ARGON_HEALTH_CONNECTOR),
        score_anchor=(ARGON_SCORE_POSITION[0] * scale,
                      ARGON_SCORE_POSITION[1] * scale),
        accuracy_anchor=(render_width - ARGON_ACCURACY_MARGIN[0] * scale,
                         ARGON_ACCURACY_MARGIN[1] * scale),
        pp_anchor=(render_width - ARGON_PP_POSITION[0] * scale,
                   ARGON_PP_POSITION[1] * scale),
    )


def argon_hit_error_meter_geometry(
    render_width: int,
    render_height: int,
) -> ArgonHitErrorMeterGeometry:
    """Two vertical meters anchored to the opposing centre edges."""
    scale = render_height / ARGON_REFERENCE_HEIGHT
    width = (ARGON_HEM_COLUMN_SIZE + ARGON_HEM_CHEVRON_SIZE) * scale
    height = (
        ARGON_HEM_ICON_SIZE * 2.0
        + ARGON_HEM_ICON_BAR_GAP * 2.0
        + ARGON_HEM_BAR_LENGTH
    ) * scale
    top = (render_height - height) * 0.5
    return ArgonHitErrorMeterGeometry(
        scale=scale,
        width=width,
        height=height,
        axis_start=(ARGON_HEM_ICON_SIZE + ARGON_HEM_ICON_BAR_GAP) * scale,
        axis_centre=height * 0.5,
        cross_centre=(ARGON_HEM_CHEVRON_SIZE
                      + ARGON_HEM_COLUMN_SIZE * 0.5) * scale,
        bar_length=ARGON_HEM_BAR_LENGTH * scale,
        column_size=ARGON_HEM_COLUMN_SIZE * scale,
        band_size=ARGON_HEM_BAND_SIZE * scale,
        tick_thickness=ARGON_HEM_TICK_THICKNESS * scale,
        centre_marker_size=ARGON_HEM_CENTRE_MARKER_SIZE * scale,
        chevron_size=ARGON_HEM_CHEVRON_SIZE * scale,
        chevron_stroke=ARGON_HEM_CHEVRON_STROKE * scale,
        edge_fade_size=ARGON_HEM_EDGE_FADE_SIZE * scale,
        instances=(
            ArgonHitErrorMeterInstance(0.0, top, False),
            ArgonHitErrorMeterInstance(render_width - width, top, True),
        ),
    )


def argon_hit_error_axis_y(
    geometry: ArgonHitErrorMeterGeometry,
    offset_ms: float,
    max_hit_window: float,
) -> float:
    """Map early (negative) hits toward the top and late hits downward."""
    if max_hit_window <= 0:
        position = 0.5
    else:
        position = max(0.0, min(
            1.0, (offset_ms / max_hit_window + 1.0) * 0.5,
        ))
    return geometry.axis_start + position * geometry.bar_length


# Argon counter geometry. The argon-counter glyphs are 240px square boxes with
# the digit content inset ~31px per side; ARGON_OVERLAP pulls the boxes together
# so the visible digits sit tight/condensed like lazer's counter. The HUD is
# authored as native 240px textures; callers provide the final 720-space scale.
ARGON_OVERLAP = 60          # native px of box-to-box overlap (out of 240);
                            # ~content-width advance so digits just touch (the
                            # ~31px inset per side is the doubled padding cancelled)
ARGON_WIRE_ALPHA = 0.15     # dim "wireframes" template behind the live digits


def _draw_mod_icons(
    ctx,
    right_x: float,
    top_y: float,
    *,
    ui_scale: float | None = None,
) -> None:
    """lazer ModDisplay: active mods as flat-topped hexagons (mod_hex tinted
    by ModType), acronym centred in fill×0.1. Top-right, horizontal flow,
    rightmost on top, resting overlap. Only replay-authored mods are shown;
    the native Mania key-count is never fabricated as a mod."""
    if not hud_component_visibility(getattr(ctx, "options", None)).mods:
        return
    s = ctx.scene
    mods = actual_mod_acronyms(int(getattr(s, "replay_mods", 0)))
    if not mods:
        return
    s2 = ui_scale if ui_scale is not None else ctx.height / 768.0
    box = max(24, int(48 * s2))          # MOD_ICON_SIZE 80 × MOD_ICON_SCALE 0.6
    step = max(10, int(33 * s2))         # 48 − (25×0.6) overlap
    total_w = box + (len(mods) - 1) * step
    left = right_x - total_w
    for i, m in enumerate(mods):
        fx = left + i * step
        fill = mod_fill_colour(m)
        # Hexagon drawn into a square box (the letterbox encodes its aspect).
        _draw_tl(ctx, "mod_hex", fx, top_y, box, box, (*fill, 1.0))
        # Acronym centred, dark tint (lerp black→fill at 0.1).
        fg = (int(fill[0] * 0.1 * 255), int(fill[1] * 0.1 * 255),
              int(fill[2] * 0.1 * 255), 255)
        tex, tw, th = ctx.text(m, max(10, int(box * 0.42)), fg)
        cx = fx + box / 2.0
        cy_gl = ctx.height - top_y - box / 2.0      # hexagon centre (GL)
        ctx.draw_external(tex, int(cx - tw / 2), int(cy_gl - th / 2), tw, th, 1.0)


def _draw_tl(ctx, name_or_idx, left, top, w, h, tint, *, is_idx=False, direct=False):
    """Draw a sprite positioned by its TOP-LEFT corner in screen pixels
    (y measured downward from the top). Converts to the engine's GL
    bottom-left origin. `direct=True` draws via the full-res direct path
    (for wide sprites like the scorebar). Skips degenerate sizes."""
    if w <= 0 or h <= 0:
        return
    gl_y = ctx.height - top - h
    if direct:
        ctx.draw_direct(name_or_idx, int(left), int(gl_y), int(w), int(h), tint)
    elif is_idx:
        ctx.draw_sprite_idx(name_or_idx, int(left), int(gl_y), int(w), int(h), tint)
    else:
        ctx.draw_sprite(name_or_idx, int(left), int(gl_y), int(w), int(h), tint)


def _draw_argon_wedges(ctx) -> None:
    """Draw both source rounded-gradient wedges with their 0.8 X shear."""
    fr = ctx.fr
    gl = fr.rc.ctx
    geometry = argon_wedge_geometry(fr.rc.height)
    renderer = argon_wedge_renderer(fr)
    fr._flush_sprite_batch()
    try:
        # The wedge shader emits premultiplied cyan/alpha.
        gl.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
        for position in geometry.source_positions:
            renderer.draw(position, geometry, (fr.rc.width, fr.rc.height))
    finally:
        gl.blend_func = (
            moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA,
        )


def _draw_capsule_segment(ctx, start, end, thickness, tint) -> None:
    """Draw one top-left-space rounded segment using the bundled capsule."""
    length = math.dist(start, end)
    if length <= 0 or thickness <= 0:
        return
    centre_x = (start[0] + end[0]) * 0.5
    centre_y_gl = ctx.height - (start[1] + end[1]) * 0.5
    angle = math.degrees(math.atan2(start[1] - end[1], end[0] - start[0]))
    width = length + thickness
    ctx.fr._draw_direct(
        "argon_hp",
        centre_x - width * 0.5,
        centre_y_gl - thickness * 0.5,
        width,
        thickness,
        tint=tint,
        rotation_deg=angle,
    )


def _draw_argon_health_paths(ctx, geometry: ArgonHudGeometry, hp: float) -> None:
    """Draw source Argon background/glow/main shader quads from ``scene.hp``."""
    hp = max(0.0, min(1.0, float(hp)))
    fr = ctx.fr
    gl = fr.rc.ctx
    viewport = (fr.rc.width, fr.rc.height)
    renderer = argon_health_path_renderer(fr)
    fr._flush_sprite_batch()
    try:
        # The ported shaders emit premultiplied RGB, matching the source.
        gl.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
        renderer.draw_background(
            geometry.health_rect,
            viewport,
            (ARGON_HEALTH_WIDTH, 50.0),
        )
        if hp <= 0.0:
            return

        gl.blend_func = (moderngl.ONE, moderngl.ONE)
        renderer.draw_bar(
            geometry.health_glow_rect,
            viewport,
            (360.0, 110.0),
            progress=hp,
            radius=ARGON_HEALTH_GLOW_PATH_RADIUS,
            glow_portion=ARGON_HEALTH_GLOW_PORTION,
            bar_colour=(1.0, 1.0, 1.0, 1.0),
            glow_colour=(0x7E / 255, 0xD7 / 255, 0xFD / 255, 0.5),
            gradient_left=(1.0, 1.0, 1.0, 0.8),
            gradient_right=(1.0, 1.0, 1.0, 1.0),
        )
        renderer.draw_bar(
            geometry.health_rect,
            viewport,
            (ARGON_HEALTH_WIDTH, 50.0),
            progress=hp,
            radius=ARGON_HEALTH_MAIN_PATH_RADIUS,
            glow_portion=ARGON_HEALTH_MAIN_GLOW_PORTION,
            bar_colour=(1.0, 1.0, 1.0, 1.0),
            glow_colour=(0x7E / 255, 0xD7 / 255, 0xFD / 255, 0.5),
            gradient_left=(1.0, 1.0, 1.0, 1.0),
            gradient_right=(1.0, 1.0, 1.0, 1.0),
        )
    finally:
        gl.blend_func = (
            moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA,
        )


def _draw_argon_health_display(ctx, geometry: ArgonHudGeometry, hp: float) -> None:
    """Draw the continuous source health path and its independent connector.

    The source's retained hit/miss colour transforms require judgement-action
    history not carried by SceneState. This deterministic presentation keeps
    that animation deferred while preserving the shader path, fill direction,
    background/glow/main composition, connector, colours, and bounds.
    """
    _draw_argon_health_paths(ctx, geometry, hp)
    x, top, width, height = geometry.connector_rect
    _draw_tl(ctx, "column_bg", x, top, width, height, (1, 1, 1, 0.95))


def _scorebar_fill_colour(hp: float) -> tuple[float, float, float]:
    """lazer LegacyHealthDisplay.getFillColour (new style): white at full,
    fading white→black approaching 0.5, then black→red approaching 0."""
    if hp >= 0.5:
        return (1.0, 1.0, 1.0)
    if hp >= 0.2:
        t = (0.5 - hp) / 0.3            # 0 at .5 → 1 at .2
        v = 1.0 - t
        return (v, v, v)
    t = (0.2 - hp) / 0.2                # 0 at .2 → 1 at 0
    return (t, 0.0, 0.0)


# --- R3D web-viewer chrome: not part of lazer. The wiki path reproduces
# lazer exactly (per r3drenderer_lazer_fidelity), so these are no-ops here.
# The legacy render_mania keeps the R3D look as the fallback renderer.
def hit_error_popups(*, element, skin, assets, variables, ctx) -> None:
    return  # lazer mania has no per-column hit-error number popups


def hit_strip(*, element, skin, assets, variables, ctx) -> None:
    if (
        not hud_component_visibility(ctx.options).hit_error_meter
    ):
        return
    if is_argon_default(ctx, 0):
        _argon_hit_error(ctx)
    else:
        ctx.fr._draw_lazer_hit_error_meter(ctx.scene)


def _fmt_time(ms: float) -> str:
    s = max(0, int(ms // 1000))
    return f"{s // 60}:{s % 60:02d}"


def _argon_song_progress(ctx) -> None:
    """lazer ArgonSongProgressBar: a thin bar across the bottom (0.9 width,
    centred) with elapsed time at the bottom-left and remaining at the right."""
    fr = ctx.fr
    s = ctx.scene
    rc = fr.rc
    A = rc.height / 1080.0
    p = max(0.0, min(1.0, s.song_progress))
    pad = 14 * A
    bar_h = max(3, int(7 * A))
    bw = rc.width * 0.9
    bx = (rc.width - bw) / 2.0
    top = rc.height - pad - bar_h          # from-top y of the bar
    _draw_tl(ctx, "column_bg", bx, top, bw, bar_h, (0.32, 0.33, 0.38, 0.55))
    _draw_tl(ctx, "column_bg", bx, top, bw * p, bar_h, (1, 1, 1, 0.95))
    # cyan head glow at the fill edge.
    _draw_tl(ctx, "column_bg", bx + bw * p, top, max(2, int(5 * A)), bar_h,
             (0.49, 0.84, 0.99, 0.6))
    # Elapsed (left) + remaining (right) time, just above the bar.
    elapsed = max(0.0, float(s.t_ms))
    total = elapsed / p if p > 0.01 else elapsed
    et, ew, eh = fr._cached_text(_fmt_time(elapsed), 20, (235, 235, 245, 255))
    fr._draw_external_texture(et, x=int(bx),
                              y=int(rc.height - top + 2 * A), w=ew, h=eh, alpha=0.9)
    rt, rw, rh = fr._cached_text("-" + _fmt_time(max(0.0, total - elapsed)), 20,
                                 (235, 235, 245, 255))
    fr._draw_external_texture(rt, x=int(bx + bw - rw),
                              y=int(rc.height - top + 2 * A), w=rw, h=rh, alpha=0.9)


def progress_bar(*, element, skin, assets, variables, ctx) -> None:
    if not ctx.options.show_progress_bar:
        return
    if is_argon_default(ctx, 0):
        _argon_song_progress(ctx)
    else:
        ctx.fr._draw_progress_bar(ctx.scene)


def _argon_hit_error(ctx) -> None:
    """Draw Argon's source pair of vertically-oriented BarHitErrorMeters."""
    from osu_mania_renderer_v2.gpu.renderer import (
        LAZER_HIT_RESULT_COLOURS,
        lazer_hit_error_tick_state,
        lazer_hit_window_bands,
    )

    fr = ctx.fr
    s = ctx.scene
    rc = fr.rc
    geometry = argon_hit_error_meter_geometry(rc.width, rc.height)
    bands = lazer_hit_window_bands(s.hit_error_windows)
    if not bands:
        return
    max_window = bands[-1].window_ms

    def rect(
        instance, local_x, local_top, width, height, tint,
        *, sprite="column_bg",
    ):
        if instance.mirrored:
            local_x = geometry.width - local_x - width
        _draw_tl(
            ctx, sprite,
            instance.left + local_x,
            instance.top + local_top,
            width, height, tint,
        )

    def point(instance, local_x, local_y):
        if instance.mirrored:
            local_x = geometry.width - local_x
        return instance.left + local_x, instance.top + local_y

    def draw_band(instance, band, *, fade_edges=False):
        extent = geometry.bar_length * band.relative_length
        top = geometry.axis_centre - extent * 0.5
        local_x = geometry.cross_centre - geometry.band_size * 0.5
        if not fade_edges:
            rect(instance, local_x, top, geometry.band_size, extent,
                 (*band.colour, 1.0))
            return
        fade = min(geometry.edge_fade_size, extent * 0.5)
        solid = max(0.0, extent - fade * 2.0)
        if solid > 0:
            rect(instance, local_x, top + fade, geometry.band_size, solid,
                 (*band.colour, 1.0))
        slices = 6
        for index in range(slices):
            y0 = fade * index / slices
            y1 = fade * (index + 1) / slices
            alpha = (index + 0.5) / slices
            rect(instance, local_x, top + y0, geometry.band_size, y1 - y0,
                 (*band.colour, alpha))
            rect(instance, local_x, top + extent - y1,
                 geometry.band_size, y1 - y0, (*band.colour, alpha))

    for instance in geometry.instances:
        draw_band(instance, bands[-1], fade_edges=True)
        for band in reversed(bands[:-1]):
            draw_band(instance, band)
        marker_colour = bands[0].colour
        size = geometry.centre_marker_size
        marker_x = geometry.cross_centre - size * 0.5
        marker_top = geometry.axis_centre - size * 0.5
        rect(instance, marker_x, marker_top, size, size,
             (*marker_colour, 1.0), sprite="note_circle")

    # Tick capsules use their real scene ages/offsets/results and additive
    # blending. The same source data is presented on both mirrored meters.
    fr._flush_sprite_batch()
    gl = rc.ctx
    gl.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
    try:
        for instance in geometry.instances:
            for event in s.hit_error_events[-50:]:
                state = lazer_hit_error_tick_state(event.age_ms)
                if state.alpha <= 0 or state.width_fraction <= 0:
                    continue
                y = argon_hit_error_axis_y(
                    geometry, event.offset_ms, max_window,
                )
                width = geometry.column_size * state.width_fraction
                local_x = geometry.cross_centre - width * 0.5
                colour = LAZER_HIT_RESULT_COLOURS.get(
                    event.judgment, (1.0, 1.0, 1.0),
                )
                rect(
                    instance, local_x, y - geometry.tick_thickness * 0.5,
                    width, geometry.tick_thickness,
                    (*colour, state.alpha),
                )
        fr._flush_sprite_batch()
    finally:
        gl.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

    for instance in geometry.instances:
        # Darkened centre foreground above the judgment ticks.
        inner = geometry.centre_marker_size * 0.5
        colour = tuple(channel * 0.7 for channel in bands[0].colour)
        rect(
            instance,
            geometry.cross_centre - inner * 0.5,
            geometry.axis_centre - inner * 0.5,
            inner, inner, (*colour, 1.0), sprite="note_circle",
        )

        # Original text fallback for unavailable hare/tortoise art. E/L stay
        # within the 16-unit source label slots at both screen edges.
        for label, local_y in (
            ("E", ARGON_HEM_ICON_SIZE * 0.5 * geometry.scale),
            ("L", (ARGON_HEM_ICON_SIZE + ARGON_HEM_ICON_BAR_GAP
                   + ARGON_HEM_BAR_LENGTH + ARGON_HEM_ICON_BAR_GAP
                   + ARGON_HEM_ICON_SIZE * 0.5) * geometry.scale),
        ):
            texture, width, height = fr._cached_text(
                label, 18, (235, 235, 245, 235),
            )
            x, y = point(instance, geometry.cross_centre, local_y)
            fr._draw_external_texture(
                texture,
                x=int(round(x - width * 0.5)),
                y=int(round(rc.height - y - height * 0.5)),
                w=width, h=height, alpha=0.92,
            )

        if s.hit_error_ema_ms is not None:
            axis = argon_hit_error_axis_y(
                geometry, s.hit_error_ema_ms, max_window,
            )
            points = (
                point(instance, geometry.scale, axis - 4.0 * geometry.scale),
                point(instance, 7.0 * geometry.scale, axis),
                point(instance, geometry.scale, axis + 4.0 * geometry.scale),
            )
            _draw_capsule_segment(
                ctx, points[0], points[1], geometry.chevron_stroke,
                (1, 1, 1, 0.95),
            )
            _draw_capsule_segment(
                ctx, points[1], points[2], geometry.chevron_stroke,
                (1, 1, 1, 0.95),
            )


def _argon_unstable_rate(ctx) -> None:
    """R3D standalone UR readout at the shared neutral 720-space origin."""
    value = max(0.0, float(getattr(ctx.scene, "unstable_rate", 0.0)))
    text = f"UR: {value:.2f}"
    texture, width, height = ctx.fr._cached_text(
        text, 27, (255, 255, 255, 235),
    )
    scale = ctx.height / ARGON_REFERENCE_HEIGHT
    left = 575.0 * scale
    top = 682.0 * scale
    ctx.fr._draw_external_texture(
        texture,
        x=int(round(left)),
        y=int(round(ctx.height - top - height)),
        w=width, h=height, alpha=0.92,
    )


def fail_overlay(*, element, skin, assets, variables, ctx) -> None:
    s = ctx.scene
    if s.hp <= 0.001 and s.results_opacity <= 0:
        ctx.fr._draw_fail_overlay()


def hp_bar(*, element, skin, assets, variables, ctx) -> None:
    """Health bar at the TOP — legacy scorebar when the skin ships it,
    else the Argon source-shaped path."""
    if not ctx.options.show_hp_bar:
        return
    # Argon default: the health line is drawn in _draw_argon_hud (after the
    # wedge, so it isn't covered). hp_bar runs before hud in RENDER_ORDER.
    if is_argon_default(ctx, 0):
        return
    hp = max(0.0, min(1.0, getattr(ctx.scene, "hp", 1.0)))
    if ctx.atlas.global_source("scorebar_bg") == "user":
        _legacy_scorebar(ctx, hp)
    else:
        _argon_health(ctx, hp)


def _legacy_scorebar(ctx, hp: float) -> None:
    """lazer LegacyHealthDisplay. Top-left, sprites scaled osu-px→render
    (s = height/480). scorebar-bg at (0,0); the colour fill sits at the
    old/new style offset and is clipped to HP×width (empties from the
    right); the marker rides the fill's right edge. New style = the skin
    ships scorebar-marker."""
    atlas = ctx.atlas
    # Legacy HUD textures render at native px × (height/768) — measured:
    # Vio scorebar-bg 40px → 59px at 1125h (×1.475).
    s = ctx.height / 768.0
    new_style = atlas.global_source("scorebar_marker") == "user"

    bg_w, bg_h = atlas.global_native_size("scorebar_bg")
    _draw_tl(ctx, "scorebar_bg", 0, 0, bg_w * s, bg_h * s, (1, 1, 1, 1), direct=True)

    col_w, col_h = atlas.global_native_size("scorebar_colour")
    if col_w <= 0:
        return
    # Fill offset in lazer's 768-space (osu value × 1.6): new (12,12.48),
    # old (4.8,16).
    off_x, off_y = (12.0, 12.48) if new_style else (4.8, 16.0)
    fill_w_full = col_w * s
    fill_h = col_h * s
    fill_w = hp * fill_w_full        # clip-approx: solid bars squish ≈ clip
    fr_tint = _scorebar_fill_colour(hp) if new_style else (1.0, 1.0, 1.0)
    _draw_tl(ctx, "scorebar_colour", off_x * s, off_y * s, fill_w, fill_h,
             (*fr_tint, 1.0), direct=True)

    # Marker at the right edge of the fill, centred (new) / top edge (old).
    right_x = off_x * s + fill_w
    if new_style:
        mk = "scorebar_marker"
        mw, mh = atlas.global_native_size(mk)
        mcx = right_x
        mcy = off_y * s + fill_h / 2.0
        _draw_tl(ctx, mk, mcx - mw * s / 2.0, mcy - mh * s / 2.0,
                 mw * s, mh * s, (*fr_tint, 1.0))
    else:
        # Old style: ki / kidanger / kidanger2 by HP, centred on the edge.
        mk = ("scorebar_kidanger2" if hp < 0.2
              else "scorebar_kidanger" if hp < 0.5 else "scorebar_ki")
        if atlas.global_source(mk) == "user":
            mw, mh = atlas.global_native_size(mk)
            _draw_tl(ctx, mk, right_x - mw * s / 2.0, off_y * s - mh * s / 2.0,
                     mw * s, mh * s, (1, 1, 1, 1))


def _argon_health(ctx, hp: float) -> None:
    """Argon procedural health: a rounded white capsule at (50,20), W=300,
    BarHeight=30 (lazer ArgonSkin coords, scaled by height/768), filling
    0→HP left→right over a dim track, cyan trailing glow."""
    s = ctx.height / 768.0
    x = 50 * s
    top = 20 * s
    w = 300 * s
    bh = max(3, int(30 * s))
    # Dim track (full width) then the white fill 0→HP.
    _draw_tl(ctx, "column_bg", x, top, w, bh, (0.10, 0.11, 0.13, 0.85))
    fill_w = hp * w
    # Trailing cyan glow just past the fill head.
    glow_w = max(2, int(6 * s))
    _draw_tl(ctx, "column_bg", x + fill_w, top, glow_w, bh,
             (0.49, 0.84, 0.99, 0.5))
    _draw_tl(ctx, "column_bg", x, top, fill_w, bh, (1, 1, 1, 0.95))


def banner(*, element, skin, assets, variables, ctx) -> None:
    return  # R3D song-title banner; lazer has no gameplay title banner


def _draw_fallback_hud(ctx) -> None:
    """Score + accuracy for skins WITHOUT a score font. lazer would use the
    default skin's font here; we render a clean PIL readout (no 4K pill, no
    zero-padding) top-right, plus the hexagon mod icons. Mirrors the skin-font
    layout so it sits in the same place."""
    fr = ctx.fr
    s = ctx.scene
    rc = fr.rc
    if not ctx.options.show_score:
        _draw_mod_icons(ctx, rc.width - max(8, int(rc.width * 0.012)),
                        max(8, int(rc.height * 0.03)))
        return
    if s.results_opacity > 0:
        disp_score, disp_acc = s.score, s.accuracy
    else:
        disp_score = s.score_smoothed if s.score_smoothed > 0 else s.score
        disp_acc = s.accuracy_smoothed
    right_pad = max(8, int(rc.width * 0.012))
    top_pad = max(8, int(rc.height * 0.02))
    right_x = rc.width - right_pad

    score_h = max(20, int(rc.height * 0.058))
    stex, sw, sh = fr._cached_text(f"{int(disp_score):d}", score_h, (255, 255, 255, 255))
    fr._draw_external_texture(stex, x=right_x - sw, y=rc.height - top_pad - sh,
                              w=sw, h=sh, alpha=1.0)
    acc_h = max(14, int(rc.height * 0.036))
    atex, aw, ah = fr._cached_text(f"{disp_acc:.2f}%", acc_h, (235, 235, 245, 255))
    acc_y = rc.height - top_pad - sh - ah - max(2, int(rc.height * 0.008))
    fr._draw_external_texture(atex, x=right_x - aw, y=acc_y, w=aw, h=ah, alpha=0.95)
    # lazer hexagon mod icons under the readout (key-count excluded).
    mods_top = (rc.height - acc_y) + max(6, int(rc.height * 0.012))
    _draw_mod_icons(ctx, right_x, mods_top)


def _argon_number(ctx, text, *, x, center_y, glyph_h, align, alpha=1.0,
                  tint=(1.0, 1.0, 1.0)):
    """Draw `text` in the argon-counter font with the dim wireframe template
    behind it (lazer's ArgonCounterTextComponent: live digits over the
    'wireframes' backing)."""
    ctx.draw_number(text, x=x, center_y=center_y, glyph_h=glyph_h,
                    overlap_px=ARGON_OVERLAP, align=align,
                    alpha=ARGON_WIRE_ALPHA, font="argon", wireframe=True,
                    tint=(0.7, 0.75, 0.85))
    ctx.draw_number(text, x=x, center_y=center_y, glyph_h=glyph_h,
                    overlap_px=ARGON_OVERLAP, align=align, alpha=alpha,
                    font="argon", tint=tint)


def _draw_argon_hud(ctx) -> None:
    """lazer's Argon default HUD: score in the top-left wedge, accuracy
    top-right (with an 'ACCURACY' label + pp under it), all in the
    argon-counter font. Layout comes from ArgonSkin's 1280x720 gameplay
    surface and scales by output height/720. Combo remains in notes.py."""
    fr = ctx.fr
    s = ctx.scene
    rc = fr.rc
    geometry = argon_hud_geometry(rc.width, rc.height)
    scale = geometry.scale
    visibility = hud_component_visibility(ctx.options)

    if s.results_opacity > 0:
        disp_score, disp_acc = s.score, s.accuracy
    else:
        disp_score = s.score_smoothed if s.score_smoothed > 0 else s.score
        disp_acc = s.accuracy_smoothed

    # ── Score: top-left wedge banner, number right-aligned inside it ──
    # The two source wedge pieces remain decorative HUD background even when
    # the independently-controlled health or score component is hidden.
    _draw_argon_wedges(ctx)

    if visibility.score:
        # argon-counter textures are 240px at glyph scale 0.125 = 30 source px.
        score_right, score_top = geometry.score_anchor
        score_height = 30.0 * scale
        score_cy_gl = rc.height - score_top - score_height * 0.5
        _argon_number(
            ctx, f"{int(disp_score):d}", x=score_right,
            center_y=score_cy_gl, glyph_h=score_height, align="right",
        )
    if visibility.health:
        _draw_argon_health_display(ctx, geometry, getattr(s, "hp", 1.0))

    # ── Accuracy: top-right, with the small 'ACCURACY' label above it ──
    acc_height = 30.0 * scale
    acc_right, acc_top = geometry.accuracy_anchor
    ltex, lw, lh = fr._cached_text("ACCURACY", 18,
                                   (200, 205, 220, 235))
    fr._draw_external_texture(ltex, x=int(acc_right - lw),
                              y=int(rc.height - acc_top - lh), w=lw, h=lh,
                              alpha=0.9)
    # Source label is 12px tall and the glyph row begins at local y=12.
    acc_cy_gl = rc.height - acc_top - 12.0 * scale - acc_height * 0.5
    _argon_number(ctx, f"{disp_acc:.2f}%", x=acc_right, center_y=acc_cy_gl,
                  glyph_h=acc_height, align="right")

    # ── PP: under the accuracy line ('PP' label + value), Argon-amber ──
    if visibility.performance_points and s.max_pp > 0:
        pp_right, pp_top = geometry.pp_anchor
        pp_height = 30.0 * ARGON_PP_SCALE * scale
        pl, plw, plh = fr._cached_text("PP", 14,
                                       (200, 205, 220, 235))
        fr._draw_external_texture(pl, x=int(pp_right - plw),
                                  y=int(rc.height - pp_top - plh), w=plw, h=plh,
                                  alpha=0.85)
        pp_cy_gl = (
            rc.height - pp_top - 12.0 * ARGON_PP_SCALE * scale
            - pp_height * 0.5
        )
        _argon_number(ctx, f"{int(s.pp)}", x=pp_right, center_y=pp_cy_gl,
                      glyph_h=pp_height, align="right", tint=(1.0, 0.86, 0.55))

    # Active mods (hexagon icons) under the accuracy/pp block, top-right.
    _draw_mod_icons(ctx, acc_right, 110.0 * scale, ui_scale=scale)

def hud(*, element, skin, assets, variables, ctx) -> None:
    """Score + accuracy readout, top-right.

    When the user skin ships the score font (`score-0..9`, etc.) we compose
    the digits from those glyphs — lazer's `LegacyScoreCounter` /
    `LegacyAccuracyCounter`, both `LegacyFont.Score`, anchored top-right,
    overlap = skin.ini `[Fonts] ScoreOverlap` (default 0). When NO user skin
    is selected (Argon default), use lazer's Argon HUD (argon-counter font,
    score wedge top-left). Otherwise fall back to the clean PIL readout."""
    # hud_opacity 0: skip the whole score/accuracy/pp/mods/wedges/health HUD
    # element. Accuracy is intentionally independent of show_score, so this
    # top-level gate remains the complete gameplay-only overlay switch.
    if ctx.options.hud_opacity <= 0.0:
        return
    fr = ctx.fr
    s = ctx.scene
    if is_argon_default(ctx, 0) and ctx.has_argon_font():
        _draw_argon_hud(ctx)
        return
    if not ctx.has_score_font():
        # Skin ships no score font → lazer falls through to the default
        # skin's font. We don't have an Argon glyph font bundled, so render a
        # clean PIL readout (no R3D "4K" pill, no 8-digit zero-padding),
        # top-right, with the lazer hexagon mod icons — NOT fr._draw_hud
        # (that's the R3D viewer chrome).
        _draw_fallback_hud(ctx)
        return

    rc = fr.rc
    # Legacy HUD scale = height/768 (same as the scorebar). Score counter
    # scale 0.96, accuracy 0.6×0.96 (lazer LegacyScore/AccuracyCounter).
    hud_scale = rc.height / 768.0
    _gw, score_nh = ctx.atlas.global_native_size("score_0")
    score_nh = score_nh or 70
    right_pad = max(6, int(10 * hud_scale))      # LegacyScoreCounter Margin H=10
    accuracy_top_margin = max(4, int(9 * hud_scale))
    overlap = ctx.skin_ini.score_overlap if ctx.skin_ini is not None else 0

    # Smoothed during gameplay (counter rolls up), authoritative on results.
    if s.results_opacity > 0:
        display_score, display_acc = s.score, s.accuracy
    else:
        display_score = s.score_smoothed if s.score_smoothed > 0 else s.score
        display_acc = s.accuracy_smoothed

    right_x = rc.width - right_pad
    # `mods_top` is a FROM-TOP y (what _draw_mod_icons/_draw_tl expect).
    mods_top = accuracy_top_margin

    if ctx.options.show_score:
        score_h = score_nh * hud_scale * 0.96
        # LegacyScoreCounter has no vertical margin. The 9-unit margin belongs
        # to LegacyAccuracyCounter below it.
        score_cy = rc.height - score_h / 2.0
        ctx.draw_number(
            f"{int(display_score):d}", x=right_x, center_y=score_cy,
            glyph_h=score_h, overlap_px=overlap, align="right",
        )
        # Accuracy: scale 0.6×0.96, Margin H=17 (7px more indented than score).
        acc_h = score_nh * hud_scale * 0.576
        acc_right_x = rc.width - max(10, int(17 * hud_scale))
        acc_cy = (score_cy - score_h / 2.0) - accuracy_top_margin - acc_h / 2.0
        ctx.draw_number(
            f"{display_acc:.2f}%", x=acc_right_x, center_y=acc_cy,
            glyph_h=acc_h, overlap_px=overlap, align="right", alpha=0.95,
        )
        # Below the accuracy line: from-top = height − (accuracy bottom GL).
        mods_top = (rc.height - (acc_cy - acc_h / 2.0)) + max(6, int(rc.height * 0.012))

    # Active mods as lazer-style hexagon icons, top-right under the readout.
    _draw_mod_icons(ctx, right_x, mods_top)

    if ctx.options.show_pp_counter and s.max_pp > 0:
        # Argon-style PP readout, right-aligned under the accuracy line.
        pp_tex, pw, ph = fr._cached_text(f"{s.pp:.0f}pp", 44, (255, 220, 140, 255))
        fr._draw_external_texture(
            pp_tex, x=rc.width - pw - right_pad,
            y=ay - ph - 14, w=pw, h=ph, alpha=0.95,
        )


def key_counter(*, element, skin, assets, variables, ctx) -> None:
    """lazer's KeyCounterDisplay (Argon style), bottom-right: one cell per
    column — a pill that lights on press, a cyan 'B{n}' trigger label, and the
    cumulative press count. Counts come from scene.key_press_counts (rising
    edges up to t). Scaled by height/1080 (lazer HUD is absolute px @1080p)."""
    if not getattr(ctx.options, "show_key_counter", True):
        return
    fr = ctx.fr
    s = ctx.scene
    counts = getattr(s, "key_press_counts", ()) or ()
    if not counts:
        return
    rc = fr.rc
    A = rc.height / 1080.0
    k = len(counts)

    pill_w = 76 * A
    pill_h = max(3, int(7 * A))
    pitch = 92 * A
    right_edge = rc.width - 40 * A
    total_w = (k - 1) * pitch + pill_w
    left = right_edge - total_w
    # Bottom-right (from-top y of the pill row), block runs pill→label→count.
    pill_top = rc.height - 150 * A
    label_top = pill_top + 14 * A
    count_top = pill_top + 36 * A

    # NOTE: _cached_text already scales `size` by height/1080, so pass
    # 1080-reference sizes here (multiplying by A would double-scale).
    label_h = 18
    count_h = 30

    for c in range(k):
        cx = left + pill_w / 2.0 + c * pitch
        held = c < len(s.keys_held) and s.keys_held[c]
        # Pill — a thin solid bar, dim grey at rest, bright white when held.
        ptint = (1.0, 1.0, 1.0, 0.95) if held else (0.55, 0.58, 0.64, 0.8)
        _draw_tl(ctx, "column_bg", cx - pill_w / 2.0, pill_top,
                 pill_w, pill_h, ptint)
        # Cyan trigger label.
        ltex, lw, lh = fr._cached_text(f"B{c + 1}", label_h, (120, 205, 240, 255))
        fr._draw_external_texture(ltex, x=int(cx - lw / 2.0),
                                  y=int(rc.height - label_top - lh),
                                  w=lw, h=lh, alpha=0.95)
        # White press count.
        ctex, cw, ch = fr._cached_text(str(counts[c]), count_h, (255, 255, 255, 255))
        fr._draw_external_texture(ctex, x=int(cx - cw / 2.0),
                                  y=int(rc.height - count_top - ch),
                                  w=cw, h=ch, alpha=1.0)


def top_chrome(*, element, skin, assets, variables, ctx) -> None:
    return  # R3D title chrome; not in lazer


def ur_summary(*, element, skin, assets, variables, ctx) -> None:
    # R3D's standalone website-controlled UR counter is separate from the
    # source Argon main HUD and never includes average-offset text.
    if (
        hud_component_visibility(ctx.options).unstable_rate
        and is_argon_default(ctx, 0)
    ):
        _argon_unstable_rate(ctx)


def results_overlay(*, element, skin, assets, variables, ctx) -> None:
    s = ctx.scene
    if s.results_opacity > 0 and ctx.options.show_result_screen:
        # Pass the live gameplay ctx so the results numbers render in the
        # argon-counter font (draw_number/_argon_number need the FrameContext).
        ctx.fr._draw_results_overlay(s, ctx)


def watermark(*, element, skin, assets, variables, ctx) -> None:
    if ctx.options.watermark_text:
        ctx.fr._draw_watermark(ctx.options.watermark_text)
