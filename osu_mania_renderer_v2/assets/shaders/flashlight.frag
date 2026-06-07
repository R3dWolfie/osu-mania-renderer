#version 330

uniform sampler2D u_scene;
uniform vec2 u_center;     // pixel coords of receptor center
uniform float u_radius;    // pixel radius of the lit area

in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec4 scene = texture(u_scene, v_uv);
    float dist = distance(gl_FragCoord.xy, u_center);
    float mask = 1.0 - smoothstep(u_radius * 0.6, u_radius, dist);
    frag_color = vec4(scene.rgb * mask, 1.0);
}
