"""Load GLSL shader programs from bundled asset files."""
from __future__ import annotations

from pathlib import Path

import moderngl

SHADERS_DIR = Path(__file__).resolve().parent.parent / "assets" / "shaders"


def load_programs(ctx: moderngl.Context) -> dict[str, moderngl.Program]:
    # Original sprite shader kept around for `_draw_external_texture` —
    # ad-hoc single-textured draws (text glyphs, the background bitmap)
    # that aren't atlas-based and don't need instancing.
    sprite = ctx.program(
        vertex_shader=(SHADERS_DIR / "sprite.vert").read_text(),
        fragment_shader=(SHADERS_DIR / "sprite.frag").read_text(),
    )
    # Instanced variant: one VBO of 4 unit-quad corners + a per-instance
    # stream of (x, y, w, h, atlas_idx, r, g, b, a). Collapses ~30 draw
    # calls per frame into a single `glDrawArraysInstanced`.
    sprite_instanced = ctx.program(
        vertex_shader=(SHADERS_DIR / "sprite_instanced.vert").read_text(),
        fragment_shader=(SHADERS_DIR / "sprite.frag").read_text(),
    )
    # The flashlight pass uses a passthrough vertex shader (full-screen quad).
    flash_vert = """#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
}
"""
    flashlight = ctx.program(
        vertex_shader=flash_vert,
        fragment_shader=(SHADERS_DIR / "flashlight.frag").read_text(),
    )
    return {
        "sprite": sprite,
        "sprite_instanced": sprite_instanced,
        "flashlight": flashlight,
    }
