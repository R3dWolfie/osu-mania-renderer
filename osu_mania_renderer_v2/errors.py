"""Exception hierarchy. Everything raised by this package is a RendererError."""
from __future__ import annotations


class RendererError(Exception):
    """Base class for everything the renderer raises."""


class BeatmapParseError(RendererError):
    """Raised when the .osu file is malformed or missing required fields."""


class ReplayParseError(RendererError):
    """Raised when the .osr file is malformed."""


class NotAManiaError(RendererError):
    """Raised when the beatmap or replay is not osu!mania (mode != 3)."""

    def __init__(self, mode: int) -> None:
        self.mode = mode
        super().__init__(f"Expected mania (mode=3), got mode={mode}")


class MissingAudioError(RendererError):
    """Raised only when RenderOptions.audio_required=True and the audio file is absent."""


class GpuUnavailableError(RendererError):
    """Raised when ModernGL cannot create a GL context."""


class EncoderError(RendererError):
    """Raised when ffmpeg exits non-zero or the output MP4 is missing/empty."""


class RenderTimeoutError(RendererError):
    """Raised when a render exceeds RenderOptions.timeout_seconds."""
