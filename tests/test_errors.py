from osu_mania_renderer_v2.errors import (
    BeatmapParseError,
    EncoderError,
    GpuUnavailableError,
    MissingAudioError,
    NotAManiaError,
    RendererError,
    RenderTimeoutError,
    ReplayParseError,
)


def test_all_inherit_from_renderer_error():
    for cls in [
        BeatmapParseError, ReplayParseError, NotAManiaError, MissingAudioError,
        GpuUnavailableError, EncoderError, RenderTimeoutError,
    ]:
        assert issubclass(cls, RendererError)


def test_not_a_mania_error_carries_mode():
    e = NotAManiaError(mode=0)
    assert e.mode == 0
    assert "0" in str(e)
