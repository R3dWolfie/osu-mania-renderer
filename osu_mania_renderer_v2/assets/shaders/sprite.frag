#version 330

uniform sampler2DArray u_atlas;
uniform float u_hd;        // 0 = off, 1 = on (Hidden)
uniform float u_fi;        // 0 = off, 1 = on (Fade In)
uniform float u_hd_recep;  // receptor Y (px, bottom-origin) — Hidden anchor
uniform float u_pf_top;    // playfield top Y (px) — Fade In anchor
uniform float u_cov_fill;  // fully-hidden fill height (px)
uniform float u_cov_grad;  // gradient fade height (px)

in vec2 v_uv;
flat in int v_atlas_index;
in vec4 v_color;

out vec4 frag_color;

void main() {
    vec4 sample_color = texture(u_atlas, vec3(v_uv, float(v_atlas_index)));
    vec4 result = sample_color * v_color;

    // osu!lazer mania Hidden / Fade In: a cover anchored at one end of the
    // playfield. `u_cov_fill` px are fully hidden; the next `u_cov_grad` px
    // fade hidden->visible (the gradient leading edge). Coverage scales with
    // combo (computed per-frame on the CPU side).
    if (u_hd > 0.5) {
        // Hidden: cover grows up from the receptor — notes vanish before
        // you hit them.
        float d = gl_FragCoord.y - u_hd_recep;
        result.a *= smoothstep(u_cov_fill, u_cov_fill + u_cov_grad, d);
    }
    if (u_fi > 0.5) {
        // Fade In: cover grows down from the top — notes appear late.
        float d = u_pf_top - gl_FragCoord.y;
        result.a *= smoothstep(u_cov_fill, u_cov_fill + u_cov_grad, d);
    }

    frag_color = result;
}
