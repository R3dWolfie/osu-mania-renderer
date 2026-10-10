"""RGB -> yuv420p on the GPU, so ffmpeg has nothing left to convert.

Stock, the engine reads back 3 bytes a pixel and ffmpeg turns every frame into
yuv420p on the CPU (`scale=in_range=full:out_range=limited,format=yuv420p`,
which with the `-colorspace bt709` the encoder is given means: BT.709 matrix,
limited range). With ``R3D_MANIA_GPU_YUV`` two small passes on the GPU do that
conversion, the engine reads back 1.5 bytes a pixel, and ffmpeg is handed
frames it only has to flip.

THE CONVERSION IS FFMPEG'S OWN, SAMPLE FOR SAMPLE. The arithmetic below is the
std engine's proven twin of libswscale's rgb24 -> yuv420p (osu_std_renderer
render/gl.py), with the integers swscale uses for BT.709. They were recovered
from ffmpeg's own output: each is the only exact match in a search around the
textbook value, and the result differs from ffmpeg 8.1 on 0 of 16.7 million
samples (noise at three sizes, one-pixel stripes, hard colour edges, real
frames of this engine). So the encoder is handed the same bytes either way
and the file is the same file.

That holds for an ffmpeg whose swscale uses this arithmetic, which is not
every build (the std engine found ffmpeg 6.1.1 on x86-64 one level apart on
about 9% of noise samples). So it is checked, not assumed: unless forced, the
conversion is used only where ``ffmpeg_matches_twin`` says THIS machine's
ffmpeg agrees with it.

Frames stay bottom-up, as the engine has always handed them over, and ffmpeg
still does the ``vflip``: converting and then flipping gives the same samples
as flipping and then converting, because the chroma filter and its edge clamp
are symmetric. The input must be declared ``-color_range tv -colorspace bt709``;
without that ffmpeg converts the already converted frames once more.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

log = logging.getLogger("osu_mania_renderer_v2.gpu.yuv")

# libswscale's integers for BT.709, limited range (15-bit fixed point).
Y_COEF = (5983, 20127, 2032)
U_COEF = (-3298, -11094, 14392)
V_COEF = (14392, -13073, -1320)
# The vertical chroma filter: eight rows, weights summing to 8192.
CHROMA_TAPS = (-116, -344, 984, 3572, 3572, 984, -344, -116)

# What the stock command does to a frame, as a filter chain that needs no
# encoder options to mean the same thing. The probe compares against this.
FFMPEG_REFERENCE_VF = ("scale=in_range=full:out_range=limited:"
                       "out_color_matrix=bt709,format=yuv420p")
# How the converted frames must be declared to ffmpeg (before `-i`).
FFMPEG_INPUT_ARGS = ("-color_range", "tv", "-colorspace", "bt709")

_MIN_ROWS = 12      # below this swscale shortens its chroma filter


def rgb_to_yuv420p(rgb) -> np.ndarray:
    """CPU twin of the GPU conversion, and of ffmpeg's: the ORACLE.

    `rgb` is (h, w, 3) uint8. Returns a flat uint8 yuv420p buffer (Y | U | V)
    in the same row order as the input. int64 throughout: the shifts must be
    arithmetic on a signed type."""
    h, w = rgb.shape[:2]
    r = rgb[..., 0].astype(np.int64)
    g = rgb[..., 1].astype(np.int64)
    b = rgb[..., 2].astype(np.int64)
    y = ((((Y_COEF[0] * r + Y_COEF[1] * g + Y_COEF[2] * b) + (0x801 << 8)) >> 9) + 32) >> 6
    r2, g2, b2 = (c[:, 0::2] + c[:, 1::2] for c in (r, g, b))
    j2 = np.arange(h // 2) * 2

    def chroma(c):
        row = ((c[0] * r2 + c[1] * g2 + c[2] * b2) + (0x4001 << 9)) >> 10
        acc = np.zeros((h // 2, w // 2), np.int64)
        for k, tap in enumerate(CHROMA_TAPS):
            acc += tap * row[np.clip(j2 - 3 + k, 0, h - 1)]
        return np.clip((acc + (1 << 18)) >> 19, 0, 255)

    u, v = chroma(U_COEF), chroma(V_COEF)
    out = np.empty(w * h * 3 // 2, np.uint8)
    out[:w * h] = y.ravel()
    out[w * h:w * h + u.size] = u.ravel()
    out[w * h + u.size:] = v.ravel()
    return out


def black_frame(width: int, height: int) -> bytes:
    """What an all-zero RGB frame converts to: Y 16, U and V 128. The frame
    reader leads the video with these while its pipeline warms up; plain
    zeros would be a green frame."""
    n = width * height
    return bytes([16]) * n + bytes([128]) * (n // 2)


def size_ok(width: int, height: int) -> bool:
    return not (width & 1) and not (height & 1) and height >= _MIN_ROWS


_probe_memo: dict = {}


def ffmpeg_matches_twin(ffmpeg_prefix=("ffmpeg",)) -> bool:
    """Does the ffmpeg this render will use convert a frame to exactly the
    bytes `rgb_to_yuv420p` gives? One small noise frame through it, once per
    ffmpeg binary: the answer is cached beside the other r3d caches, keyed on
    the binary's path, size and modification time. Any trouble counts as no."""
    prefix = tuple(ffmpeg_prefix)
    if prefix in _probe_memo:
        return _probe_memo[prefix]
    ok = False
    try:
        mark = None
        exe = shutil.which(prefix[0]) if len(prefix) == 1 else None
        if exe:
            real = os.path.realpath(exe)
            st = os.stat(real)
            key = hashlib.sha1(
                f"{real}|{st.st_size}|{st.st_mtime_ns}|mania-bt709-v1".encode()
            ).hexdigest()[:16]
            base = (os.path.expanduser("~/Library/Caches/r3d") if sys.platform == "darwin"
                    else os.path.join(tempfile.gettempdir(), "r3d-cache"))
            mark = os.path.join(base, f"mania-gpu-yuv-probe-{key}")
        cached = ""
        if mark:
            try:
                with open(mark) as fh:
                    cached = fh.read().strip()
            except OSError:
                cached = ""
        if cached in ("1", "0"):
            ok = cached == "1"
        else:
            w, h = 128, 96
            rgb = np.random.default_rng(20261010).integers(
                0, 256, (h, w, 3), dtype=np.uint8)
            p = subprocess.run(
                [*prefix, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{w}x{h}", "-i", "pipe:0", "-vf", FFMPEG_REFERENCE_VF,
                 "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"],
                input=rgb.tobytes(), capture_output=True, timeout=15)
            ok = p.returncode == 0 and p.stdout == rgb_to_yuv420p(rgb).tobytes()
            if mark:
                try:
                    os.makedirs(os.path.dirname(mark), exist_ok=True)
                    with open(mark, "w") as fh:
                        fh.write("1" if ok else "0")
                except OSError:
                    pass
    except Exception:  # noqa: BLE001 - a probe must never fail a render
        ok = False
    _probe_memo[prefix] = ok
    return ok


def wanted(width: int, height: int, encoder: str, ffmpeg_prefix=("ffmpeg",)) -> bool:
    """Should this render convert on the GPU? Decided when the ffmpeg command
    is planned, because the command's input format depends on it.

      R3D_MANIA_GPU_YUV unset : on macOS (where it was built, measured and
                                compared) if this machine's ffmpeg agrees with
                                the conversion; off anywhere else.
      R3D_MANIA_GPU_YUV=1     : on, whatever the platform or the ffmpeg.
      R3D_MANIA_GPU_YUV=0     : off.

    Never with VAAPI (its frames go through `format=nv12,hwupload`, a path
    that has not been compared), and never for a size the conversion does not
    cover (odd, or under 12 rows)."""
    v = os.environ.get("R3D_MANIA_GPU_YUV")
    forced = v is not None and v.strip().lower() not in ("", "0", "false", "no", "off")
    if v is not None and not forced:
        return False
    if encoder == "h264_vaapi" or not size_ok(width, height):
        return False
    if forced:
        return True
    if sys.platform != "darwin":
        return False
    return ffmpeg_matches_twin(ffmpeg_prefix)


class YuvConverter:
    """The two conversion passes and their targets, for one frame size.

    INTEGER math throughout, the same as `rgb_to_yuv420p`. Texture samples
    come back as normalised floats, so `int(v*255.0 + 0.5)` recovers the byte.
    Every sum stays inside int32 (the largest, the chroma accumulator, is
    under 1.6e8)."""

    def __init__(self, ctx, width: int, height: int) -> None:
        import moderngl
        if not size_ok(width, height):
            raise ValueError(f"GPU yuv needs even dimensions and at least "
                             f"{_MIN_ROWS} rows, got {width}x{height}")
        self._mgl = moderngl
        self.ctx = ctx
        self.width, self.height = width, height
        ry, gy, by = Y_COEF
        ru, gu, bu = U_COEF
        rv, gv, bv = V_COEF
        vert = ("#version 330\nin vec2 in_pos;\n"
                "void main(){ gl_Position = vec4(in_pos,0.0,1.0); }")
        frag_y = f"""#version 330
        uniform sampler2D scene;
        out float outY;
        void main() {{
            vec3 c = texelFetch(scene, ivec2(gl_FragCoord.xy), 0).rgb;
            int r = int(c.r*255.0+0.5), g = int(c.g*255.0+0.5), b = int(c.b*255.0+0.5);
            outY = float(((((({ry}*r + {gy}*g + {by}*b) + {0x801 << 8}) >> 9) + 32) >> 6)) / 255.0;
        }}"""
        # U and V share the gather (two pixels across, eight rows down), so
        # one pass with two attachments does both.
        frag_uv = f"""#version 330
        uniform sampler2D scene;
        layout(location=0) out float outU;
        layout(location=1) out float outV;
        const int TAP[8] = int[8]({", ".join(str(t) for t in CHROMA_TAPS)});
        void main() {{
            ivec2 p = ivec2(gl_FragCoord.xy);
            int x = p.x * 2;
            int ymax = textureSize(scene, 0).y - 1;
            int su = 0, sv = 0;
            for (int k = 0; k < 8; ++k) {{
                int y = clamp(p.y * 2 - 3 + k, 0, ymax);
                ivec3 s = ivec3(texelFetch(scene, ivec2(x, y), 0).rgb * 255.0 + 0.5)
                        + ivec3(texelFetch(scene, ivec2(x + 1, y), 0).rgb * 255.0 + 0.5);
                su += TAP[k] * ((({ru}*s.r + {gu}*s.g + {bu}*s.b) + {0x4001 << 9}) >> 10);
                sv += TAP[k] * ((({rv}*s.r + {gv}*s.g + {bv}*s.b) + {0x4001 << 9}) >> 10);
            }}
            outU = float(clamp((su + {1 << 18}) >> 19, 0, 255)) / 255.0;
            outV = float(clamp((sv + {1 << 18}) >> 19, 0, 255)) / 255.0;
        }}"""
        self._quad = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], "f4").tobytes())
        self._prog_y = ctx.program(vertex_shader=vert, fragment_shader=frag_y)
        self._prog_uv = ctx.program(vertex_shader=vert, fragment_shader=frag_uv)
        self._vao_y = ctx.vertex_array(self._prog_y, [(self._quad, "2f4", "in_pos")])
        self._vao_uv = ctx.vertex_array(self._prog_uv, [(self._quad, "2f4", "in_pos")])
        self._tex_y = ctx.texture((width, height), 1, dtype="f1")
        self._tex_u = ctx.texture((width // 2, height // 2), 1, dtype="f1")
        self._tex_v = ctx.texture((width // 2, height // 2), 1, dtype="f1")
        self.fbo_y = ctx.framebuffer(color_attachments=[self._tex_y])
        self.fbo_uv = ctx.framebuffer(color_attachments=[self._tex_u, self._tex_v])
        self.frame_size = width * height * 3 // 2
        self._prog_y["scene"] = 0
        self._prog_uv["scene"] = 0

    def run(self, scene_tex) -> None:
        """Both passes over `scene_tex`. Blending is off for them (they write
        computed values, not composites) and back on afterwards, which is how
        the frame renderer starts every frame anyway. The caller binds its
        own framebuffer again."""
        ctx, mgl = self.ctx, self._mgl
        ctx.disable(mgl.BLEND)
        scene_tex.use(location=0)
        self.fbo_y.use()
        self._vao_y.render(mgl.TRIANGLES)
        self.fbo_uv.use()
        self._vao_uv.render(mgl.TRIANGLES)
        ctx.enable(mgl.BLEND)

    def read_into(self, buffer, offset: int = 0) -> None:
        """The three planes into one buffer (a pixel buffer object or a
        bytearray) at their yuv420p offsets, so whoever writes the frame out
        pushes one contiguous block and need not know it is planar."""
        ysz = self.width * self.height
        csz = (self.width // 2) * (self.height // 2)
        self.fbo_y.read_into(buffer, components=1, alignment=1, write_offset=offset)
        self.fbo_uv.read_into(buffer, components=1, alignment=1, attachment=0,
                              write_offset=offset + ysz)
        self.fbo_uv.read_into(buffer, components=1, alignment=1, attachment=1,
                              write_offset=offset + ysz + csz)

    def read_bytes(self) -> bytearray:
        out = bytearray(self.frame_size)
        self.read_into(out)
        return out


class CpuTwinReader:
    """The way out when the GPU passes cannot be built on this machine after
    the ffmpeg command has already been started for yuv420p frames: read RGB
    the stock way and convert each frame with the CPU twin. Slow (about 16 ms
    a 720p frame) and the same bytes, so the render is late instead of lost.

    Wraps a stock `FrameReader`; `read()` and `drain()` as there, except that
    what comes out is always plain bytes."""

    def __init__(self, reader, width: int, height: int) -> None:
        self._reader = reader
        self._w, self._h = width, height
        self.frame_size = width * height * 3 // 2

    def _convert(self, buf) -> bytes:
        rgb = np.frombuffer(buf, np.uint8).reshape(self._h, self._w, 3)
        return rgb_to_yuv420p(rgb).tobytes()

    def read(self) -> bytes:
        frame = self._reader.read()
        mv = getattr(frame, "mv", None)
        if mv is None:
            return self._convert(frame)
        try:
            return self._convert(mv)
        finally:
            frame.done.set()        # the lease is ours to release: nobody else holds it

    def drain(self) -> list:
        return [self._convert(f) for f in self._reader.drain()]

