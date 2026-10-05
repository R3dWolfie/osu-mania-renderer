"""R3D mode policy precedes stable fallback semantics in both compositors."""
import os
from types import SimpleNamespace

import pytest
from PIL import Image

from osu_mania_renderer_v2.beatmap.skin_ini import parse_skin_ini
from osu_mania_renderer_v2.gpu.atlas import SpriteAtlas
from osu_mania_renderer_v2.gpu.context import HeadlessGl
from osu_mania_renderer_v2.gpu.renderer import FrameRenderer
from osu_mania_renderer_v2.wiki_elements._common import is_argon_default
from osu_mania_renderer_v2.wiki_elements.context import FrameContext


@pytest.mark.skipif(os.getenv("RUN_SLOW") != "1", reason="opt-in real GL")
@pytest.mark.parametrize("case,expected", [
    ("nonmatching-7K", True),
    ("matching-4K", False),
    ("user-current-note", False),
    ("beatmap-current-key", False),
    ("classic-only", True),
    ("standard-hud-only", True),
])
def test_current_render_gate_and_wiki_agree(tmp_path, case, expected):
    skin, beatmap = tmp_path / "skin", tmp_path / "map"
    skin.mkdir()
    beatmap.mkdir()
    key_count = 4 if case == "matching-4K" else 7
    (skin / "skin.ini").write_text(f"[General]\nVersion:2.7\n[Mania]\nKeys:{key_count}\n")
    if case == "classic-only":
        (skin / "skin.ini").unlink()
    if case == "user-current-note":
        Image.new("RGBA", (8, 8), "red").save(skin / "mania-note1.png")
    if case == "beatmap-current-key":
        Image.new("RGBA", (8, 8), "red").save(beatmap / "mania-key1.png")
    if case == "standard-hud-only":
        for name in ("scorebar-bg", "scorebar-colour", "score-0", "hit300"):
            Image.new("RGBA", (8, 8), "red").save(skin / f"{name}.png")
    ini = parse_skin_ini(skin)
    section = ini.mania_for_keycount(4) if ini else None
    with HeadlessGl(320, 240) as gl:
        atlas = SpriteAtlas.load(gl.ctx, key_count=4, skin_dir=skin,
                                 beatmap_dir=beatmap, mania_section=section)
        fr = object.__new__(FrameRenderer)
        fr.rc = SimpleNamespace(key_count=4)
        fr.skin_ini, fr.mania_section, fr.atlas = ini, section, atlas
        assert fr._is_argon_default() is expected
        ctx = FrameContext(fr, None, gl.ctx, gl.fbo, 320, 240, 4)
        assert is_argon_default(ctx, 0) is expected
        assert is_argon_default(ctx, 3) is expected
        if expected:
            assert atlas.column_source("note_tap", 0) == "classic"
