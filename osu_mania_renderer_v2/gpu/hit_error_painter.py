"""One cached GPU painter for TheAussie's horizontal Mania hit-error meter.

Analytic rounded strokes and a continuous six-pixel tip gradient avoid
per-frame texture generation, stepped approximations and doubled chevron
joint alpha. Icon bytes remain the canonical bundled PNGs.
"""

from __future__ import annotations

import os
import sys
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

# ---- every shape of a blend run in ONE draw (R3D_MANIA_GROUPED) --------------
# The meter is about 22 shapes a frame, and each used to be 8 uniform writes
# and its own draw call: ~176 uniform writes and ~22 draws a frame from Python.
# Grouped, a shape is one row in an instance buffer and each run of shapes that
# share a blend mode is one instanced draw (4 a frame: bands + centre, the
# additive ticks, the dark centre, the chevron). Instances are rasterised in
# order, so the blending is the same.
#
# The picture is meant to be the same to the byte. To keep it so, nothing is
# recomputed: the vertex shader's position expression is the stock one, and the
# fragment shader IS the stock source, with each uniform turned into a global
# that main() fills from a `flat` per-instance value (no interpolation).
#
# On by default on macOS, where it was measured and compared; off elsewhere
# until it has been checked there. R3D_MANIA_GROUPED=1 / =0 forces it.
_SHAPE_FLOATS = 17            # rect 4, colour 4, a 2, b 2, c 2, radius, fade, mode
_SHAPE_CAP = 96               # shapes per run before a flush (a frame has ~22)

_VERTEX_GROUPED = """
#version 330
in vec2 in_pos;
in vec4 i_rect;
in vec4 i_colour;
in vec2 i_a;
in vec2 i_b;
in vec2 i_c;
in vec3 i_radius_fade_mode;
uniform vec2 screen;
out vec2 point;
flat out vec4 v_rect;
flat out vec4 v_colour;
flat out vec2 v_a;
flat out vec2 v_b;
flat out vec2 v_c;
flat out float v_radius;
flat out float v_fade;
flat out int v_mode;
void main() {
    point = i_rect.xy + in_pos * i_rect.zw;
    gl_Position = vec4(point.x / screen.x * 2.0 - 1.0,
                       1.0 - point.y / screen.y * 2.0, 0.0, 1.0);
    v_rect = i_rect;
    v_colour = i_colour;
    v_a = i_a;
    v_b = i_b;
    v_c = i_c;
    v_radius = i_radius_fade_mode.x;
    v_fade = i_radius_fade_mode.y;
    v_mode = int(i_radius_fade_mode.z + 0.5);
}
"""

_GROUPED_NAMES = (("vec4", "colour"), ("vec4", "rect"), ("vec2", "a"), ("vec2", "b"),
                  ("vec2", "c"), ("float", "radius"), ("float", "fade"), ("int", "mode"))


def _grouped_fragment() -> str:
    """The stock fragment shader with its uniforms fed per instance: every
    `uniform T name;` becomes a `flat in T v_name;` plus a plain global
    `T name;` that main() fills first, so the body below is the stock text,
    word for word. (Not #defines: `a` would also rewrite `colour.a`.)"""
    src = _FRAGMENT
    for gl_type, name in _GROUPED_NAMES:
        decl = f"uniform {gl_type} {name};\n"
        if src.count(decl) != 1:
            raise RuntimeError(f"hit error meter: uniform {name} not found once")
        src = src.replace(decl, f"flat in {gl_type} v_{name};\n{gl_type} {name};\n")
    entry = "void main() {\n"
    if src.count(entry) != 1:
        raise RuntimeError("hit error meter: main() not found once")
    fill = "".join(f"    {name} = v_{name};\n" for _t, name in _GROUPED_NAMES)
    return src.replace(entry, entry + fill)


def grouped_draws() -> bool:
    v = os.environ.get("R3D_MANIA_GROUPED")
    if v is None:
        return sys.platform == "darwin"
    return v.strip().lower() not in ("", "0", "false", "no", "off")


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
        self._grouped = grouped_draws()
        self._queued = 0
        if self._grouped:
            self._grouped_program = gl.program(
                vertex_shader=_VERTEX_GROUPED, fragment_shader=_grouped_fragment())
            self._grouped_program["screen"].value = (renderer.rc.width, renderer.rc.height)
            self._shape_rows = np.zeros((_SHAPE_CAP, _SHAPE_FLOATS), dtype="f4")
            self._shape_buffer = gl.buffer(reserve=self._shape_rows.nbytes, dynamic=True)
            self._grouped_vao = gl.vertex_array(
                self._grouped_program,
                [
                    (self.buffer, "2f", "in_pos"),
                    (self._shape_buffer, "4f 4f 2f 2f 2f 3f /i",
                     "i_rect", "i_colour", "i_a", "i_b", "i_c", "i_radius_fade_mode"),
                ],
            )
        self.icons = {}
        root = Path(__file__).resolve().parents[1] / "assets" / "hud" / "hit-error-meter"
        for name in ("hare", "tortoise"):
            with Image.open(root / f"{name}.png") as source:
                image = source.convert("RGBA")
            texture = gl.texture(image.size, 4, image.tobytes())
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            texture.repeat_x = texture.repeat_y = False
            self.icons[name] = texture

    def _flush_shapes(self) -> None:
        """Draw the queued shapes in one instanced call. Called before anything
        that must land between two shapes: a blend-mode change, an icon, the
        end of the meter. Does nothing on the stock path."""
        n = self._queued
        if n == 0:
            return
        from .renderer import _stream_write
        _stream_write(self._shape_buffer, self._shape_rows[:n])
        self._grouped_vao.render(moderngl.TRIANGLES, vertices=6, instances=n)
        self._queued = 0

    def _shape(self, rect, colour, *, mode=0, a=(0, 0), b=(0, 0), c=(0, 0), radius=0, fade=0):
        if self._grouped:
            if self._queued >= _SHAPE_CAP:
                self._flush_shapes()
            self._shape_rows[self._queued] = (*rect, *colour, *a, *b, *c, radius, fade, mode)
            self._queued += 1
            return
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
        self._flush_shapes()
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
            self._flush_shapes()
            gl.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self._stroke((centre, centre), CENTRE_MARKER_SIZE / 2 * scale, (*CENTRE_DARK_BLUE, 1))
        self._flush_shapes()

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
        self._flush_shapes()
