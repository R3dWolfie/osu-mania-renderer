"""Argon judgment presentation dispatch regression coverage."""
from types import SimpleNamespace

from osu_mania_renderer_v2.gpu.renderer import FrameRenderer


def test_argon_combo_and_judgment_uses_shared_presentation():
    fr = object.__new__(FrameRenderer)
    fr._is_argon_default = lambda: True
    calls = []
    fr._draw_argon_combo_and_judgment = (
        lambda scene, draw_combo=True: calls.append((scene, draw_combo))
    )
    scene = SimpleNamespace()

    FrameRenderer._draw_combo_and_judgment(fr, scene, draw_combo=False)

    assert calls == [(scene, False)]
