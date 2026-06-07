#version 330

uniform sampler2DArray u_atlas;
uniform float u_hd;     // 0 = off, 1 = on
uniform float u_fi;     // 0 = off, 1 = on
uniform float u_hd_top; // playfield-top in NDC for fade math
uniform float u_hd_bot; // playfield-bot in NDC

in vec2 v_uv;
flat in int v_atlas_index;
in vec4 v_color;

out vec4 frag_color;

void main() {
    vec4 sample_color = texture(u_atlas, vec3(v_uv, float(v_atlas_index)));
    vec4 result = sample_color * v_color;

    // Hidden / Fade In alpha shaping based on screen Y.
    float y_frac = clamp(
        (gl_FragCoord.y - u_hd_bot) / max(u_hd_top - u_hd_bot, 1e-3),
        0.0, 1.0
    );
    if (u_hd > 0.5) {
        // Fade out near the receptor (low y).
        float a = smoothstep(0.0, 0.4, y_frac);
        result.a *= a;
    }
    if (u_fi > 0.5) {
        // Fade in from the top.
        float a = smoothstep(1.0, 0.6, y_frac);
        result.a *= a;
    }

    frag_color = result;
}
