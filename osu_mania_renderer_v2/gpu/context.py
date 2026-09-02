"""Headless ModernGL context backed by EGL (no window manager required).

By default we let moderngl pick the EGL display, which on multi-GPU
NVIDIA boxes routes to the primary device. When `R3D_EGL_DEVICE_INDEX`
is set (e.g. `=1` for the second NVIDIA GPU), we bypass moderngl's
default device pick and bind the GL context to that specific EGL
device via `eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, …)`.

Why the env var route doesn't suffice:
   `CUDA_VISIBLE_DEVICES` controls CUDA visibility (affects NVENC) but
   not EGL device selection. `EGL_VISIBLE_DEVICES` is honoured by
   some NVIDIA drivers for compute workloads but not for the EGL
   device list returned by `eglQueryDevicesEXT`. The platform-device
   extension is the only reliable cross-driver path.
"""
from __future__ import annotations

import ctypes
import logging
import os
import sys

import moderngl

from osu_mania_renderer_v2.errors import GpuUnavailableError

log = logging.getLogger("osu_mania_renderer_v2")


# ─── EGL constants / fn signatures (only used when device-pinning) ────────

_EGL_PLATFORM_DEVICE_EXT = 0x313F
_EGL_OPENGL_API = 0x30A2
_EGL_PBUFFER_BIT = 0x0001
_EGL_SURFACE_TYPE = 0x3033
_EGL_RENDERABLE_TYPE = 0x3040
_EGL_OPENGL_BIT = 0x0008
_EGL_NONE = 0x3038
_EGL_WIDTH = 0x3057
_EGL_HEIGHT = 0x3056
_EGL_CONTEXT_MAJOR_VERSION = 0x3098
_EGL_CONTEXT_MINOR_VERSION = 0x30FB


def _create_pinned_egl_context(device_index: int) -> moderngl.Context:
    """Open an EGL display + context bound to the requested EGL device.

    Returns a moderngl.Context wrapping the new context. The context is
    immediately made current; subsequent moderngl calls operate on it.
    """
    egl = ctypes.CDLL("libEGL.so.1")
    egl.eglGetProcAddress.restype = ctypes.c_void_p
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]

    query_addr = egl.eglGetProcAddress(b"eglQueryDevicesEXT")
    display_addr = egl.eglGetProcAddress(b"eglGetPlatformDisplayEXT")
    if not query_addr or not display_addr:
        raise GpuUnavailableError(
            "EGL_EXT_device_query / EGL_EXT_platform_device not available "
            "in libEGL — cannot pin to a specific GPU"
        )

    QueryDevices = ctypes.CFUNCTYPE(
        ctypes.c_uint, ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int),
    )(query_addr)
    GetPlatformDisplay = ctypes.CFUNCTYPE(
        ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    )(display_addr)

    # Enumerate devices.
    num = ctypes.c_int(0)
    QueryDevices(0, None, ctypes.byref(num))
    if num.value <= device_index:
        raise GpuUnavailableError(
            f"R3D_EGL_DEVICE_INDEX={device_index} but only {num.value} "
            "EGL devices are visible"
        )
    devices = (ctypes.c_void_p * num.value)()
    QueryDevices(num.value, devices, ctypes.byref(num))

    display = GetPlatformDisplay(_EGL_PLATFORM_DEVICE_EXT, devices[device_index], None)
    if not display:
        raise GpuUnavailableError(
            f"eglGetPlatformDisplayEXT returned NULL for device[{device_index}]"
        )

    egl.eglInitialize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                                  ctypes.POINTER(ctypes.c_int)]
    egl.eglInitialize.restype = ctypes.c_uint
    egl.eglBindAPI.argtypes = [ctypes.c_uint]
    egl.eglChooseConfig.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                                    ctypes.POINTER(ctypes.c_void_p),
                                    ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    egl.eglCreatePbufferSurface.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                            ctypes.POINTER(ctypes.c_int)]
    egl.eglCreatePbufferSurface.restype = ctypes.c_void_p
    egl.eglCreateContext.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    egl.eglCreateContext.restype = ctypes.c_void_p
    egl.eglMakeCurrent.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_void_p, ctypes.c_void_p]

    major = ctypes.c_int(0)
    minor = ctypes.c_int(0)
    if not egl.eglInitialize(display, ctypes.byref(major), ctypes.byref(minor)):
        raise GpuUnavailableError(
            f"eglInitialize failed on device[{device_index}]"
        )
    egl.eglBindAPI(_EGL_OPENGL_API)

    config_attribs = (ctypes.c_int * 7)(
        _EGL_SURFACE_TYPE, _EGL_PBUFFER_BIT,
        _EGL_RENDERABLE_TYPE, _EGL_OPENGL_BIT,
        _EGL_NONE, 0, 0,
    )
    config = ctypes.c_void_p(0)
    num_configs = ctypes.c_int(0)
    egl.eglChooseConfig(display, config_attribs,
                        ctypes.byref(config), 1, ctypes.byref(num_configs))
    if num_configs.value < 1 or not config.value:
        raise GpuUnavailableError(
            f"eglChooseConfig returned no configs on device[{device_index}]"
        )

    # A 1x1 pbuffer surface is enough — moderngl will allocate its own
    # framebuffer for actual rendering.
    pb_attribs = (ctypes.c_int * 5)(_EGL_WIDTH, 1, _EGL_HEIGHT, 1, _EGL_NONE)
    surface = egl.eglCreatePbufferSurface(display, config, pb_attribs)
    if not surface:
        raise GpuUnavailableError("eglCreatePbufferSurface returned NULL")

    ctx_attribs = (ctypes.c_int * 5)(
        _EGL_CONTEXT_MAJOR_VERSION, 3,
        _EGL_CONTEXT_MINOR_VERSION, 3,
        _EGL_NONE,
    )
    context = egl.eglCreateContext(display, config, None, ctx_attribs)
    if not context:
        raise GpuUnavailableError(
            f"eglCreateContext failed on device[{device_index}]"
        )

    if not egl.eglMakeCurrent(display, surface, surface, context):
        raise GpuUnavailableError("eglMakeCurrent failed")

    # Hand the already-current context off to moderngl. detect_framebuffer
    # picks up the bound surface.
    return moderngl.create_context(require=330, standalone=False)


class HeadlessGl:
    """Open a standalone GL context and allocate an offscreen FBO.

    Use as a context manager: closes the GL context on exit.

    GPU selection: set `R3D_EGL_DEVICE_INDEX` to the EGL device index
    (typically 0 = primary NVIDIA GPU, 1 = second NVIDIA GPU). Leave
    unset for default behaviour.
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._ctx: moderngl.Context | None = None
        self._fbo: moderngl.Framebuffer | None = None
        self._color = None
        self._depth = None

    def __enter__(self) -> HeadlessGl:
        device_index_env = os.environ.get("R3D_EGL_DEVICE_INDEX")
        try:
            if sys.platform == "win32":
                # Windows: glcontext ships ONLY the WGL module -- there is no
                # EGL device and 'wgl' is NOT a valid backend name. Passing NO
                # backend= lets moderngl.default_backend() select WGL. No EGL
                # device pinning (R3D_EGL_DEVICE_INDEX / MODERNGL_BACKEND have
                # no WGL equivalent; Windows uses the primary adapter).
                self._ctx = moderngl.create_standalone_context(require=330)
            elif device_index_env is not None:
                self._ctx = _create_pinned_egl_context(int(device_index_env))
            else:
                self._ctx = moderngl.create_standalone_context(
                    backend=os.environ.get("MODERNGL_BACKEND", "egl"),
                    require=330,
                )
        except GpuUnavailableError:
            raise
        except Exception as e:
            raise GpuUnavailableError(f"could not create GL context: {e}") from e
        self._color = self._ctx.texture((self.width, self.height), 4)
        self._depth = self._ctx.depth_renderbuffer((self.width, self.height))
        self._fbo = self._ctx.framebuffer(
            color_attachments=[self._color],
            depth_attachment=self._depth,
        )
        self._fbo.use()
        renderer = self._ctx.info.get("GL_RENDERER", "?")
        log.info(
            "gl_context_ready",
            extra={
                "renderer": renderer,
                "egl_device_index": device_index_env,
            },
        )
        return self

    @property
    def ctx(self) -> moderngl.Context:
        if self._ctx is None:
            raise GpuUnavailableError("context is closed")
        return self._ctx

    @property
    def fbo(self) -> moderngl.Framebuffer:
        if self._fbo is None:
            raise GpuUnavailableError("fbo not allocated")
        return self._fbo

    def close(self) -> None:
        for obj in (self._fbo, self._color, self._depth, self._ctx):
            if obj is not None:
                try:
                    obj.release()
                except Exception:  # noqa: BLE001
                    pass
        self._ctx = None
        self._fbo = None
        self._color = None
        self._depth = None

    def __exit__(self, *exc) -> None:
        self.close()
