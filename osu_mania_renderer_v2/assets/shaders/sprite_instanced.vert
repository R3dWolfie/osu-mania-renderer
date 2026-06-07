#version 330

// Per-vertex: one of the 4 unit-quad corners (0,0), (1,0), (0,1), (1,1).
// Drawn as TRIANGLE_STRIP so 4 vertices = one quad. Same 4-vertex VBO
// reused for every sprite this frame.
in vec2 in_corner;

// Per-instance: clip-space rectangle and tint, packed into one stream.
// Layout matches the 9 floats `_draw_sprite` writes into the instance
// buffer slice: x, y, w, h, atlas_idx, r, g, b, a.
in vec4 in_rect;       // (x_clip, y_clip, w_clip, h_clip)
in float in_atlas;
in vec4 in_color;

out vec2 v_uv;
flat out int v_atlas_index;
out vec4 v_color;

void main() {
    vec2 pos = in_rect.xy + in_corner * in_rect.zw;
    gl_Position = vec4(pos, 0.0, 1.0);
    // Flip V so the atlas's row-0-at-top convention matches our drawing.
    v_uv = vec2(in_corner.x, 1.0 - in_corner.y);
    v_atlas_index = int(in_atlas);
    v_color = in_color;
}
