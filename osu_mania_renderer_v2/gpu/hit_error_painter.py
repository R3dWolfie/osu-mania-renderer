"""One cached GPU painter for TheAussie's horizontal Mania hit-error meter.

Analytic rounded strokes and a continuous six-pixel tip gradient avoid
per-frame texture generation, stepped approximations and doubled chevron
joint alpha. Icon bytes remain the canonical bundled PNGs.
"""

from __future__ import annotations

from pathlib import Path

import moderngl
import numpy as np
from PIL import Image

from .hit_error_meter import (
    BLUE,
    CENTRE_DARK_BLUE,
    CENTRE_MARKER_SIZE,
    CHEVRON_STROKE_WIDTH,
    COLUMN_SIZE,
    EDGE_FADE_SIZE,
    HIT_RESULT_COLOURS,
    ICON_SIZE,
    MARKER_MAX_CONCURRENT,
    TICK_THICKNESS,
    hit_error_meter_geometry,
    hit_error_offset_position,
    hit_error_tick_state,
    hit_window_bands,
)

_VERTEX = """
#version 330
in vec2 in_pos;
uniform vec2 screen;
uniform vec4 rect;
out vec2 point;
void main() {
    point = rect.xy + in_pos * rect.zw;
    gl_Position = vec4(point.x / screen.x * 2.0 - 1.0,
                       1.0 - point.y / screen.y * 2.0, 0.0, 1.0);
}
"""

_FRAGMENT = """
#version 330
in vec2 point;
out vec4 frag;
uniform vec4 colour;
uniform vec4 rect;
uniform vec2 a;
uniform vec2 b;
uniform vec2 c;
uniform float radius;
uniform float fade;
uniform int mode;
float segmentDistance(vec2 p, vec2 start, vec2 end) {
    vec2 delta = end - start;
    float t = clamp(dot(p - start, delta) / max(dot(delta, delta), 0.00001), 0.0, 1.0);
    return length(p - start - delta * t);
}
void main() {
    float alpha = 1.0;
    if (mode == 1) {
        // No texture sampling/slices: exact linear fade in render pixels.
        if (fade > 0.0)
            alpha = clamp(min(point.x - rect.x, rect.x + rect.z - point.x) / fade, 0.0, 1.0);
    } else {
        float distance = segmentDistance(point, a, b);
        if (mode == 2) distance = min(distance, segmentDistance(point, b, c));
        alpha = 1.0 - smoothstep(radius - 0.5, radius + 0.5, distance);
    }
    frag = vec4(colour.rgb, colour.a * alpha);
}
"""


class HitErrorMeterPainter:
    def __init__(self, renderer):
        self.fr = renderer
        gl = renderer.rc.ctx
        self.program = gl.program(vertex_shader=_VERTEX, fragment_shader=_FRAGMENT)
        self.buffer = gl.buffer(
            np.array(
                [(0, 0), (1, 0), (1, 1), (0, 0), (1, 1), (0, 1)],
                dtype="f4",
            ).tobytes()
        )
        self.vao = gl.simple_vertex_array(self.program, self.buffer, "in_pos")
        self.program["screen"].value = (renderer.rc.width, renderer.rc.height)
        self.icons = {}
        root = Path(__file__).resolve().parents[1] / "assets" / "hud" / "hit-error-meter"
        for name in ("hare", "tortoise"):
            with Image.open(root / f"{name}.png") as source:
                image = source.convert("RGBA")
            texture = gl.texture(image.size, 4, image.tobytes())
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            texture.repeat_x = texture.repeat_y = False
            self.icons[name] = texture

    def _shape(self, rect, colour, *, mode=0, a=(0, 0), b=(0, 0), c=(0, 0), radius=0, fade=0):
        program = self.program
        for name, value in (
            ("rect", rect),
            ("colour", colour),
            ("mode", mode),
            ("a", a),
            ("b", b),
            ("c", c),
            ("radius", radius),
            ("fade", fade),
        ):
            program[name].value = value
        self.vao.render(moderngl.TRIANGLES)

    def _stroke(self, points, thickness, colour):
        radius = thickness / 2
        xs, ys = zip(*points, strict=True)
        margin = radius + 1
        rect = (
            min(xs) - margin,
            min(ys) - margin,
            max(xs) - min(xs) + margin * 2,
            max(ys) - min(ys) + margin * 2,
        )
        self._shape(
            rect,
            colour,
            a=points[0],
            b=points[1],
            c=points[-1],
            radius=radius,
            mode=2 if len(points) == 3 else 0,
        )

    def draw(self, scene):
        bands = hit_window_bands(scene.hit_error_windows)
        if not bands:
            return
        fr = self.fr
        fr._flush_sprite_batch()
        gl = fr.rc.ctx
        gl.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        geometry = hit_error_meter_geometry(fr.rc.width, fr.rc.height)
        scale = geometry.scale

        def point(local):
            return geometry.component_x + local[0], geometry.component_top + local[1]

        for band in reversed(bands):
            x, y, width, height = geometry.band_rect(band.relative_length)
            self._shape(
                (geometry.component_x + x, geometry.component_top + y, width, height),
                (*band.colour, 1),
                mode=1,
                fade=EDGE_FADE_SIZE * scale if band is bands[-1] else 0,
            )

        centre = point((geometry.axis_centre, geometry.cross_centre))
        self._stroke((centre, centre), CENTRE_MARKER_SIZE * scale, (*BLUE, 1))
        # Batch lifetime/cap data comes from the absolute gameplay clock.
        gl.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
        try:
            events = [
                event for event in scene.hit_error_events if event.judgment in HIT_RESULT_COLOURS
            ][-MARKER_MAX_CONCURRENT:]
            for event in events:
                state = hit_error_tick_state(event.age_ms)
                if state.alpha <= 0 or state.width_fraction <= 0:
                    continue
                position = hit_error_offset_position(event.offset_ms, bands[-1].window_ms)
                points = tuple(
                    point(p) for p in geometry.marker_segment(position, state.width_fraction)
                )
                self._stroke(
                    points,
                    TICK_THICKNESS * scale,
                    (*HIT_RESULT_COLOURS[event.judgment], state.alpha),
                )
        finally:
            gl.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self._stroke((centre, centre), CENTRE_MARKER_SIZE / 2 * scale, (*CENTRE_DARK_BLUE, 1))

        for name, local_centre in (
            ("hare", geometry.early_icon_centre),
            ("tortoise", geometry.late_icon_centre),
        ):
            texture = self.icons[name]
            fit = min(ICON_SIZE * scale / texture.width, COLUMN_SIZE * scale / texture.height)
            width, height = texture.width * fit, texture.height * fit
            x, top = point(local_centre)
            fr._draw_external_texture(
                texture,
                x=x - width / 2,
                y=fr.rc.height - top - height / 2,
                w=width,
                h=height,
                alpha=1,
            )

        position = scene.hit_error_chevron_position
        if position is not None:
            self._stroke(
                tuple(point(p) for p in geometry.chevron_points(position)),
                CHEVRON_STROKE_WIDTH * scale,
                (1, 1, 1, 0.95),
            )
