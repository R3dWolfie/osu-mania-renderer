from osu_mania_renderer_v2.models import (
    HoldNote,
    KeyEvent,
    Note,
    RenderOptions,
    VisualMods,
)


def test_note_is_frozen():
    n = Note(column=2, time_ms=1000)
    import dataclasses
    assert dataclasses.is_dataclass(n)
    import pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        n.column = 3


def test_hold_note_has_end_time():
    h = HoldNote(column=0, time_ms=500, end_time_ms=1500)
    assert h.duration_ms == 1000


def test_render_options_defaults():
    o = RenderOptions(resolution=(1920, 1080), fps=60)
    assert o.encoder == "auto"
    assert o.encoder_device is None
    assert o.timeout_seconds == 600
    assert o.audio_required is False


def test_visual_mods_flags():
    v = VisualMods(hidden=True, flashlight=False, fade_in=False)
    assert v.hidden
    assert not v.flashlight


def test_key_event():
    e = KeyEvent(time_ms=1234, keys_held=0b0101)
    assert e.is_held(0)
    assert not e.is_held(1)
    assert e.is_held(2)
