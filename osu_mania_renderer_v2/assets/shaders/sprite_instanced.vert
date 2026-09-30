#version 330

// Per-vertex: one of the 4 unit-quad corners (0,0), (1,0), (0,1), (1,1).
// Drawn as TRIANGLE_STRIP so 4 vertices = one quad. Same 4-vertex VBO
// reused for every sprite this frame.
in vec2 in_corner;

// Per-instance: clip-space rectangle, tint, and source-Y crop, packed into
// one stream. Layout matches the 11 floats `_draw_sprite_idx` writes.
in vec4 in_rect;       // (x_clip, y_clip, w_clip, h_clip)
in float in_atlas;
in vec4 in_color;
// Source fractions measured bottom-to-top. Regular sprites use (0, 1);
// partial repeated hold-body tiles provide cropped bounds.
in vec2 in_v_bounds;

out vec2 v_uv;
flat out int v_atlas_index;
out vec4 v_color;

void main() {
    vec2 pos = in_rect.xy + in_corner * in_rect.zw;
    gl_Position = vec4(pos, 0.0, 1.0);
    // Flip V so the atlas's row-0-at-top convention matches our drawing.
    float source_y = mix(in_v_bounds.x, in_v_bounds.y, in_corner.y);
    v_uv = vec2(in_corner.x, 1.0 - source_y);
    v_atlas_index = int(in_atlas);
    v_color = in_color;
}
