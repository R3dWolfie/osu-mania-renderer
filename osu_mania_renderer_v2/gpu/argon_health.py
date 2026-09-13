"""Source-faithful Argon health-path geometry and ModernGL renderer."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import moderngl


@dataclass(frozen=True)
class ArgonHealthPathGeometry:
    """CPU representation of the path phases evaluated by the GLSL shader."""

    size: tuple[float, float]
    radius: float
    start: tuple[float, float]
    top_arc_start: tuple[float, float]
    top_arc_end: tuple[float, float]
    slash_end: tuple[float, float]
    bottom_arc_end: tuple[float, float]
    end: tuple[float, float]
    top_arc_centre: tuple[float, float]
    bottom_arc_centre: tuple[float, float]
    arc_radius: float
    arc_angle: float
    top_line_end: float
    top_arc_end_progress: float
    slash_end_progress: float
    bottom_arc_end_progress: float


def argon_health_path_geometry(
    width: float = 300.0,
    height: float = 50.0,
    radius: float = 10.0,
) -> ArgonHealthPathGeometry:
    """Resolve the exact phase boundaries from ``getBarTexturePosition``."""
    p1 = (min(radius, width * 0.5), min(radius, height * 0.5))
    p4 = (
        max(width - radius, width * 0.5),
        max(height - radius, height * 0.5),
    )
    top_width = max(width - radius - 70.0, p1[0]) - p1[0]
    bottom_width = p4[0] - max(width - radius - 40.0, p1[0])
    if top_width < bottom_width:
        top_width = bottom_width = (top_width + bottom_width) * 0.5

    p2 = (p1[0] + top_width, p1[1])
    p3 = (p4[0] - bottom_width, p4[1])
    slash_length = math.dist(p2, p3)
    round_offset = min(top_width, slash_length * 0.5, 10.0)
    slash_fraction = round_offset / slash_length if slash_length > 0 else 0.0
    arc1_start = (p2[0] - round_offset, p2[1])
    arc1_end = (
        p2[0] + (p3[0] - p2[0]) * slash_fraction,
        p2[1] + (p3[1] - p2[1]) * slash_fraction,
    )
    arc2_start = (
        p2[0] + (p3[0] - p2[0]) * (1.0 - slash_fraction),
        p2[1] + (p3[1] - p2[1]) * (1.0 - slash_fraction),
    )
    arc2_end = (p3[0] + round_offset, p3[1])

    slash_angle = math.atan2(p2[1] - p3[1], p2[0] - p3[0]) - math.pi / 2
    if slash_angle < 0:
        slash_angle += math.tau
    slash_angle = math.tau - math.pi / 2 - slash_angle
    arc_angle = (math.pi / 2 - slash_angle * 0.5) * 2.0
    arc_radius = round_offset * math.tan(slash_angle * 0.5)
    arc_length = arc_angle * arc_radius
    first_line_length = arc1_start[0] - p1[0]
    slash_segment_length = math.dist(arc1_end, arc2_start)
    last_line_length = p4[0] - arc2_end[0]
    total = (
        first_line_length
        + slash_segment_length
        + last_line_length
        + arc_length * 2.0
    )
    if total <= 0:
        raise ValueError("Argon health path has no drawable length")

    top_line_end = first_line_length / total
    top_arc_end = (first_line_length + arc_length) / total
    slash_end = (first_line_length + arc_length + slash_segment_length) / total
    bottom_arc_end = (
        first_line_length + slash_segment_length + arc_length * 2.0
    ) / total
    return ArgonHealthPathGeometry(
        size=(width, height),
        radius=radius,
        start=p1,
        top_arc_start=arc1_start,
        top_arc_end=arc1_end,
        slash_end=arc2_start,
        bottom_arc_end=arc2_end,
        end=p4,
        top_arc_centre=(p2[0] - round_offset, p2[1] + arc_radius),
        bottom_arc_centre=(p3[0] + round_offset, p3[1] - arc_radius),
        arc_radius=arc_radius,
        arc_angle=arc_angle,
        top_line_end=top_line_end,
        top_arc_end_progress=top_arc_end,
        slash_end_progress=slash_end,
        bottom_arc_end_progress=bottom_arc_end,
    )


def argon_health_path_point(
    progress: float,
    geometry: ArgonHealthPathGeometry | None = None,
) -> tuple[float, float]:
    """Return the source centreline endpoint for a clamped progress value."""
    path = geometry or argon_health_path_geometry()
    progress = max(0.0, min(1.0, float(progress)))

    def lerp(start, end, amount):
        return (
            start[0] + (end[0] - start[0]) * amount,
            start[1] + (end[1] - start[1]) * amount,
        )

    if progress <= path.top_line_end:
        amount = progress / path.top_line_end if path.top_line_end else 0.0
        return lerp(path.start, path.top_arc_start, amount)
    if progress <= path.top_arc_end_progress:
        span = path.top_arc_end_progress - path.top_line_end
        amount = (progress - path.top_line_end) / span if span else 0.0
        angle = -math.pi / 2 + path.arc_angle * amount
        return (
            path.top_arc_centre[0] + math.cos(angle) * path.arc_radius,
            path.top_arc_centre[1] + math.sin(angle) * path.arc_radius,
        )
    if progress <= path.slash_end_progress:
        span = path.slash_end_progress - path.top_arc_end_progress
        amount = (
            (progress - path.top_arc_end_progress) / span if span else 0.0
        )
        return lerp(path.top_arc_end, path.slash_end, amount)
    if progress <= path.bottom_arc_end_progress:
        span = path.bottom_arc_end_progress - path.slash_end_progress
        amount = (progress - path.slash_end_progress) / span if span else 0.0
        angle = math.pi / 2 + path.arc_angle * (1.0 - amount)
        return (
            path.bottom_arc_centre[0] + math.cos(angle) * path.arc_radius,
            path.bottom_arc_centre[1] + math.sin(angle) * path.arc_radius,
        )
    span = 1.0 - path.bottom_arc_end_progress
    amount = (
        (progress - path.bottom_arc_end_progress) / span if span else 0.0
    )
    return lerp(path.bottom_arc_end, path.end, amount)


_VERTEX_SHADER = """#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
uniform vec4 u_rect;
uniform vec2 u_viewport;
void main() {
    vec2 pixel = u_rect.xy + in_pos * u_rect.zw;
    vec2 ndc = pixel / u_viewport * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = vec2(in_uv.x, 1.0 - in_uv.y);
}
"""


_PATH_UTILS_GLSL = """
#define ARGON_PI 3.1415926536
#define ARGON_HALF_PI 1.57079632679
#define ARGON_TWO_PI 6.28318530718

float dstToLine(vec2 start, vec2 end, vec2 pixelPos) {
    float lineLength = distance(end, start);
    if (lineLength < 0.001) return distance(pixelPos, start);
    vec2 a = (end - start) / lineLength;
    vec2 closest = clamp(dot(a, pixelPos - start), 0.0, lineLength) * a + start;
    return distance(closest, pixelPos);
}

float dstToArc(vec2 arcPos, float radius, float angle, vec2 pixelPos, float rotation) {
    pixelPos -= arcPos;
    float pixelAngle = atan(-pixelPos.y, -pixelPos.x) - ARGON_HALF_PI + rotation;
    if (pixelAngle < 0.0) pixelAngle += ARGON_TWO_PI;
    vec2 cs = vec2(cos(pixelAngle - ARGON_HALF_PI), sin(pixelAngle - ARGON_HALF_PI));
    pixelPos = cs * vec2(distance(pixelPos, vec2(0.0)));
    pixelPos.x = abs(pixelPos.x);
    if (angle == 0.0) return distance(pixelPos, vec2(0.0, radius));
    vec2 sc = vec2(sin(angle * 0.5), cos(angle * 0.5));
    return (sc.y * pixelPos.x > sc.x * pixelPos.y)
        ? length(pixelPos - sc * radius) : abs(length(pixelPos) - radius);
}

float dstToTopLine(vec2 range, vec2 p1, vec2 arcStart, float arcProgress, vec2 pos) {
    if (range.x > arcProgress) return 1000.0;
    if (arcProgress < 0.001) return distance(pos, arcStart);
    range.y = min(range.y, arcProgress);
    return dstToLine(
        mix(p1, arcStart, range.x / arcProgress),
        mix(p1, arcStart, range.y / arcProgress),
        pos
    );
}

float dstToTopArc(
    vec2 range, float radius, float ps, float pe, vec2 centre,
    float slash, float full, vec2 pos
) {
    if (range.x > pe || range.y < ps) return 1000.0;
    if (pe - ps < 0.001) return distance(pos, vec2(centre.x, centre.y - radius));
    range.x = max(range.x, ps);
    range.y = min(range.y, pe);
    float angle = full * (range.y - range.x) / (pe - ps);
    float offset = full * (range.x - ps) / (pe - ps);
    return dstToArc(
        centre, radius, angle, pos,
        ARGON_HALF_PI + (slash + full - angle) * 0.5 - offset
    );
}

float dstToSlash(vec2 range, vec2 p1, vec2 p2, float ps, float pe, vec2 pos) {
    if (range.x > pe || range.y < ps) return 1000.0;
    range.x = max(range.x, ps);
    range.y = min(range.y, pe);
    return dstToLine(
        mix(p1, p2, (range.x - ps) / (pe - ps)),
        mix(p1, p2, (range.y - ps) / (pe - ps)),
        pos
    );
}

float dstToBottomArc(
    vec2 range, float radius, float ps, float pe, vec2 centre,
    float slash, float full, vec2 pos
) {
    if (range.x > pe || range.y < ps) return 1000.0;
    if (pe - ps < 0.001) return distance(pos, vec2(centre.x, centre.y + radius));
    range.x = max(range.x, ps);
    range.y = min(range.y, pe);
    float angle = full * (range.y - range.x) / (pe - ps);
    float offset = full * (range.x - ps) / (pe - ps);
    return dstToArc(
        centre, radius, angle, pos,
        ARGON_PI + ARGON_HALF_PI + (slash - full + angle) * 0.5 + offset
    );
}

float dstToBottomLine(
    vec2 range, vec2 arcEnd, vec2 p4, float progress, vec2 pos
) {
    if (range.y < progress) return 1000.0;
    if (1.0 - progress < 0.001) return distance(pos, arcEnd);
    range.x = max(range.x, progress);
    return dstToLine(
        mix(arcEnd, p4, (range.x - progress) / (1.0 - progress)),
        mix(arcEnd, p4, (range.y - progress) / (1.0 - progress)),
        pos
    );
}

float getBarTexturePosition(vec2 size, vec2 range, float radius, vec2 pos) {
    vec2 p1 = vec2(min(radius, size.x * 0.5), min(radius, size.y * 0.5));
    vec2 p4 = vec2(
        max(size.x - radius, size.x * 0.5),
        max(size.y - radius, size.y * 0.5)
    );
    if (p4.y == p1.y) {
        return dstToLine(
            vec2(p1.x + range.x * (p4.x - p1.x), p1.y),
            vec2(p1.x + range.y * (p4.x - p1.x), p1.y),
            pos
        );
    }
    float topWidth = max(size.x - radius - 70.0, p1.x) - p1.x;
    float bottomWidth = p4.x - max(size.x - radius - 40.0, p1.x);
    if (topWidth < bottomWidth) {
        float width = (topWidth + bottomWidth) * 0.5;
        topWidth = width;
        bottomWidth = width;
    }
    vec2 p2 = vec2(p1.x + topWidth, p1.y);
    vec2 p3 = vec2(p4.x - bottomWidth, p4.y);
    float slashLength = distance(p2, p3);
    float roundOffset = min(min(topWidth, slashLength * 0.5), 10.0);
    vec2 arc1Start = vec2(p2.x - roundOffset, p2.y);
    vec2 arc1End = mix(p2, p3, roundOffset / slashLength);
    vec2 arc2Start = mix(p2, p3, 1.0 - roundOffset / slashLength);
    vec2 arc2End = vec2(p3.x + roundOffset, p3.y);
    float slashAngle = atan(p2.y - p3.y, p2.x - p3.x) - ARGON_HALF_PI;
    if (slashAngle < 0.0) slashAngle += ARGON_TWO_PI;
    slashAngle = ARGON_TWO_PI - ARGON_HALF_PI - slashAngle;
    float arcAngle = (ARGON_HALF_PI - slashAngle * 0.5) * 2.0;
    float arcRadius = roundOffset * tan(slashAngle * 0.5);
    float arcLength = arcAngle * arcRadius;
    float l1 = arc1Start.x - p1.x;
    float l2 = distance(arc1End, arc2Start);
    float l3 = p4.x - arc2End.x;
    float total = l1 + l2 + l3 + arcLength * 2.0;
    float p1s = l1 / total;
    float p1e = (l1 + arcLength) / total;
    float p2s = (l1 + arcLength + l2) / total;
    float p2e = (l1 + l2 + arcLength * 2.0) / total;
    vec2 c1 = vec2(p2.x - roundOffset, p2.y + arcRadius);
    vec2 c2 = vec2(p3.x + roundOffset, p3.y - arcRadius);
    float result = dstToTopLine(range, p1, arc1Start, p1s, pos);
    result = min(
        result,
        dstToTopArc(range, arcRadius, p1s, p1e, c1, slashAngle, arcAngle, pos)
    );
    result = min(
        result,
        dstToSlash(range, arc1End, arc2Start, p1e, p2s, pos)
    );
    result = min(
        result,
        dstToBottomArc(range, arcRadius, p2s, p2e, c2, slashAngle, arcAngle, pos)
    );
    return min(result, dstToBottomLine(range, arc2End, p4, p2e, pos));
}
"""


_BAR_FRAGMENT_SHADER = """#version 330
in vec2 v_uv;
out vec4 frag_color;
uniform vec2 u_size;
uniform vec2 u_progress_range;
uniform float u_path_radius;
uniform float u_glow_portion;
uniform vec4 u_bar_colour;
uniform vec4 u_glow_colour;
uniform vec4 u_gradient_left;
uniform vec4 u_gradient_right;
""" + _PATH_UTILS_GLSL + """
vec4 glowAt(float pos, float portion) {
    float amount = 1.0 - (pos - u_path_radius + portion) / portion;
    amount *= amount;
    amount *= amount;
    amount *= amount;
    return vec4(u_glow_colour.rgb, u_glow_colour.a * amount);
}

vec4 barColourAt(float pos) {
    float portion = u_path_radius * u_glow_portion;
    pos = clamp(pos, 0.0, u_path_radius);
    if (pos < u_path_radius - portion - 1.0) return u_bar_colour;
    if (pos < u_path_radius - portion) {
        return mix(
            u_glow_colour,
            u_bar_colour,
            u_path_radius - portion - pos
        );
    }
    return glowAt(pos, portion);
}

void main() {
    vec2 absolutePos = u_size * v_uv;
    float pathPosition = getBarTexturePosition(
        u_size, u_progress_range, u_path_radius, absolutePos
    );
    vec4 gradient = mix(u_gradient_left, u_gradient_right, v_uv.x);
    frag_color = barColourAt(pathPosition) * gradient;
    frag_color.rgb *= frag_color.a;
}
"""


_BACKGROUND_FRAGMENT_SHADER = """#version 330
in vec2 v_uv;
out vec4 frag_color;
uniform vec2 u_size;
""" + _PATH_UTILS_GLSL + """
vec4 backgroundAt(float pos, float radius) {
    float relative = clamp(pos / radius, 0.0, 1.5);
    return mix(
        vec4(0.0, 0.0, 0.0, 0.2),
        vec4(1.0, 1.0, 1.0, 0.8),
        relative / 1.5
    );
}

vec4 backgroundColour(float pos, float radius) {
    if (pos > radius - 1.0) {
        return mix(
            vec4(1.0),
            vec4(1.0, 1.0, 1.0, 0.0),
            pos - (radius - 1.0)
        );
    }
    if (pos > radius - 2.0) {
        return mix(
            backgroundAt(pos, radius),
            vec4(1.0),
            pos - (radius - 2.0)
        );
    }
    return backgroundAt(pos, radius);
}

void main() {
    const float radius = 10.0;
    float pathPosition = getBarTexturePosition(
        u_size, vec2(0.0, 1.0), radius, u_size * v_uv
    );
    frag_color = backgroundColour(pathPosition, radius);
    frag_color.rgb *= frag_color.a;
}
"""


class ArgonHealthPathRenderer:
    """One cached VBO/VAO/program set for all three health layers."""

    def __init__(self, frame_renderer: Any) -> None:
        gl = frame_renderer.rc.ctx
        self.background_program = gl.program(
            vertex_shader=_VERTEX_SHADER,
            fragment_shader=_BACKGROUND_FRAGMENT_SHADER,
        )
        self.bar_program = gl.program(
            vertex_shader=_VERTEX_SHADER,
            fragment_shader=_BAR_FRAGMENT_SHADER,
        )
        self.background_vao = gl.simple_vertex_array(
            self.background_program,
            frame_renderer._unit_quad,
            "in_pos",
            "in_uv",
        )
        self.bar_vao = gl.simple_vertex_array(
            self.bar_program,
            frame_renderer._unit_quad,
            "in_pos",
            "in_uv",
        )

    @staticmethod
    def _set_common_uniforms(
        program,
        rect: tuple[float, float, float, float],
        viewport: tuple[int, int],
        native_size: tuple[float, float],
    ) -> None:
        left, top, width, height = rect
        viewport_width, viewport_height = viewport
        program["u_rect"].value = (
            left,
            viewport_height - top - height,
            width,
            height,
        )
        program["u_viewport"].value = viewport
        program["u_size"].value = native_size

    def draw_background(
        self,
        rect: tuple[float, float, float, float],
        viewport: tuple[int, int],
        native_size: tuple[float, float],
    ) -> None:
        self._set_common_uniforms(
            self.background_program, rect, viewport, native_size,
        )
        self.background_vao.render(moderngl.TRIANGLES)

    def draw_bar(
        self,
        rect: tuple[float, float, float, float],
        viewport: tuple[int, int],
        native_size: tuple[float, float],
        *,
        progress: float,
        radius: float,
        glow_portion: float,
        bar_colour: tuple[float, float, float, float],
        glow_colour: tuple[float, float, float, float],
        gradient_left: tuple[float, float, float, float],
        gradient_right: tuple[float, float, float, float],
    ) -> None:
        program = self.bar_program
        self._set_common_uniforms(program, rect, viewport, native_size)
        program["u_progress_range"].value = (0.0, progress)
        program["u_path_radius"].value = radius
        program["u_glow_portion"].value = glow_portion
        program["u_bar_colour"].value = bar_colour
        program["u_glow_colour"].value = glow_colour
        program["u_gradient_left"].value = gradient_left
        program["u_gradient_right"].value = gradient_right
        self.bar_vao.render(moderngl.TRIANGLES)


def argon_health_path_renderer(frame_renderer: Any) -> ArgonHealthPathRenderer:
    """Return the renderer-owned cached Argon path resources."""
    renderer = getattr(frame_renderer, "_argon_health_path_renderer", None)
    if renderer is None:
        renderer = ArgonHealthPathRenderer(frame_renderer)
        frame_renderer._argon_health_path_renderer = renderer
    return renderer
