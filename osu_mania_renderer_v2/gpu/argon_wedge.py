"""Procedural source-faithful Argon HUD wedge primitive."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import moderngl

ARGON_WEDGE_REFERENCE_HEIGHT = 720.0
ARGON_WEDGE_SIZE = (380.0, 72.0)
ARGON_WEDGE_POSITIONS = ((-50.0, 15.0), (-46.0, 20.0))
ARGON_WEDGE_CORNER_RADIUS = 10.0
ARGON_WEDGE_SHEAR_X = 0.8
ARGON_WEDGE_ACCENT = (0x66 / 255, 0xCC / 255, 1.0)
ARGON_WEDGE_TOP_ALPHA = 0.0
ARGON_WEDGE_BOTTOM_ALPHA = 0.25


@dataclass(frozen=True)
class ArgonWedgeGeometry:
    """Source constants plus their output-space scale."""

    scale: float
    source_size: tuple[float, float]
    output_size: tuple[float, float]
    source_positions: tuple[tuple[float, float], ...]
    output_positions: tuple[tuple[float, float], ...]
    corner_radius: float
    output_corner_radius: float
    shear_x: float
    accent: tuple[float, float, float]
    top_alpha: float
    bottom_alpha: float


def argon_wedge_geometry(render_height: int) -> ArgonWedgeGeometry:
    """Resolve source wedge constants for an output height."""
    scale = render_height / ARGON_WEDGE_REFERENCE_HEIGHT
    return ArgonWedgeGeometry(
        scale=scale,
        source_size=ARGON_WEDGE_SIZE,
        output_size=tuple(value * scale for value in ARGON_WEDGE_SIZE),
        source_positions=ARGON_WEDGE_POSITIONS,
        output_positions=tuple(
            (x * scale, y * scale) for x, y in ARGON_WEDGE_POSITIONS
        ),
        corner_radius=ARGON_WEDGE_CORNER_RADIUS,
        output_corner_radius=ARGON_WEDGE_CORNER_RADIUS * scale,
        shear_x=ARGON_WEDGE_SHEAR_X,
        accent=ARGON_WEDGE_ACCENT,
        top_alpha=ARGON_WEDGE_TOP_ALPHA,
        bottom_alpha=ARGON_WEDGE_BOTTOM_ALPHA,
    )


def argon_wedge_transform_point(
    position: tuple[float, float],
    local_point: tuple[float, float],
    scale: float,
) -> tuple[float, float]:
    """Apply the framework's source shear, translation, then output scale."""
    local_x, local_y = local_point
    return (
        (position[0] + local_x - ARGON_WEDGE_SHEAR_X * local_y) * scale,
        (position[1] + local_y) * scale,
    )


_VERTEX_SHADER = """#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_local;
uniform vec2 u_position;
uniform vec2 u_size;
uniform vec2 u_viewport;
uniform float u_scale;
uniform float u_shear_x;
void main() {
    v_local = vec2(in_pos.x * u_size.x, (1.0 - in_uv.y) * u_size.y);
    vec2 source = vec2(
        u_position.x + v_local.x - u_shear_x * v_local.y,
        u_position.y + v_local.y
    );
    vec2 pixel = vec2(source.x * u_scale, u_viewport.y - source.y * u_scale);
    vec2 ndc = pixel / u_viewport * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
"""


_FRAGMENT_SHADER = """#version 330
in vec2 v_local;
out vec4 frag_color;
uniform vec2 u_size;
uniform float u_corner_radius;
uniform vec3 u_accent;
uniform float u_top_alpha;
uniform float u_bottom_alpha;
void main() {
    vec2 half_size = u_size * 0.5;
    vec2 q = abs(v_local - half_size) - (half_size - vec2(u_corner_radius));
    float distance_to_edge = length(max(q, vec2(0.0)))
        + min(max(q.x, q.y), 0.0) - u_corner_radius;
    float antialias = max(fwidth(distance_to_edge), 0.001);
    float mask = 1.0 - smoothstep(-antialias, antialias, distance_to_edge);
    float gradient = mix(
        u_top_alpha,
        u_bottom_alpha,
        clamp(v_local.y / u_size.y, 0.0, 1.0)
    );
    float alpha = mask * gradient;
    frag_color = vec4(u_accent * alpha, alpha);
}
"""


class ArgonWedgeRenderer:
    """Cached program/VAO using the renderer's existing immutable quad VBO."""

    def __init__(self, frame_renderer: Any) -> None:
        gl = frame_renderer.rc.ctx
        self.program = gl.program(
            vertex_shader=_VERTEX_SHADER,
            fragment_shader=_FRAGMENT_SHADER,
        )
        self.vao = gl.simple_vertex_array(
            self.program,
            frame_renderer._unit_quad,
            "in_pos",
            "in_uv",
        )

    def draw(
        self,
        position: tuple[float, float],
        geometry: ArgonWedgeGeometry,
        viewport: tuple[int, int],
    ) -> None:
        program = self.program
        program["u_position"].value = position
        program["u_size"].value = geometry.source_size
        program["u_viewport"].value = viewport
        program["u_scale"].value = geometry.scale
        program["u_shear_x"].value = geometry.shear_x
        program["u_corner_radius"].value = geometry.corner_radius
        program["u_accent"].value = geometry.accent
        program["u_top_alpha"].value = geometry.top_alpha
        program["u_bottom_alpha"].value = geometry.bottom_alpha
        self.vao.render(moderngl.TRIANGLES)


def argon_wedge_renderer(frame_renderer: Any) -> ArgonWedgeRenderer:
    """Return renderer-owned cached wedge resources."""
    renderer = getattr(frame_renderer, "_argon_wedge_renderer", None)
    if renderer is None:
        renderer = ArgonWedgeRenderer(frame_renderer)
        frame_renderer._argon_wedge_renderer = renderer
    return renderer
