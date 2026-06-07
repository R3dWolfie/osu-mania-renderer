#version 330

uniform mat3 u_projection;

in vec2 in_pos;
in vec2 in_uv;
in float in_atlas_index;
in vec4 in_color;

out vec2 v_uv;
flat out int v_atlas_index;
out vec4 v_color;

void main() {
    vec3 p = u_projection * vec3(in_pos, 1.0);
    gl_Position = vec4(p.xy, 0.0, 1.0);
    v_uv = in_uv;
    v_atlas_index = int(in_atlas_index);
    v_color = in_color;
}
