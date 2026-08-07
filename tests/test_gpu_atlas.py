import os

import pytest

from osu_mania_renderer_v2.gpu.atlas import (
    DEFAULT_LAYOUT,
    PER_COLUMN_KINDS,
    SpriteAtlas,
    column_variant,
    default_column_kind,
)
from osu_mania_renderer_v2.gpu.context import HeadlessGl


@pytest.mark.slow
def test_load_default_atlas():
    if os.environ.get("RUN_SLOW") != "1":
        pytest.skip("RUN_SLOW=1 to run GL smoke tests")
    with HeadlessGl(width=64, height=64) as gl:
        atlas = SpriteAtlas.load(gl.ctx, key_count=4)
        # Global slots — looked up by name.
        for name in (
            "hit_light",
            "judgment_geki", "judgment_300", "judgment_katu",
            "judgment_100", "judgment_50", "judgment_miss",
            "column_bg", "note_circle",
        ):
            assert isinstance(atlas.index_of(name), int)
        # Per-column slots — looked up by (kind, column).
        for kind in PER_COLUMN_KINDS:
            for c in range(4):
                idx = atlas.column_slot_index(kind, c)
                assert isinstance(idx, int)
                # No collision with global slots.
                assert idx >= 0
        # Texture array is bound and accessible.
        assert atlas.texture_array is not None


def test_index_of_missing_raises():
    a = SpriteAtlas(key_count=4)
    a._global_indices = {"hit_light": 0}
    with pytest.raises(KeyError):
        a.index_of("nonexistent")


def test_column_slot_index_out_of_range():
    a = SpriteAtlas(key_count=4)
    with pytest.raises(IndexError):
        a.column_slot_index("note_tap", 99)
    with pytest.raises(KeyError):
        a.column_slot_index("not_a_kind", 0)


def test_default_layout_matches_wiki():
    # Osu! wiki Skinning/osu!mania default layout, verified against the
    # published table.
    assert DEFAULT_LAYOUT[4] == ("1", "2", "2", "1")
    assert DEFAULT_LAYOUT[5] == ("1", "2", "S", "2", "1")
    assert DEFAULT_LAYOUT[6] == ("1", "2", "1", "1", "2", "1")
    assert DEFAULT_LAYOUT[7] == ("1", "2", "1", "S", "1", "2", "1")
    assert DEFAULT_LAYOUT[8] == ("1", "2", "1", "2", "2", "1", "2", "1")
    assert DEFAULT_LAYOUT[9] == ("1", "2", "1", "2", "S", "2", "1", "2", "1")


def test_column_variant_routes_through_wiki_layout():
    # 6K col 2 was previously misclassified as 'inner' by the synthetic
    # column_variant rule; should now be 'outer' (matches wiki).
    assert column_variant(2, 6) == "outer"
    # 7K col 3 = S = center.
    assert column_variant(3, 7) == "center"
    # 4K col 0/3 = outer, col 1/2 = inner.
    assert column_variant(0, 4) == "outer"
    assert column_variant(1, 4) == "inner"
    assert column_variant(2, 4) == "inner"
    assert column_variant(3, 4) == "outer"


def test_default_column_kind_unusual_keycounts():
    # 11K isn't in the wiki table — fallback should give edges outer,
    # centre S on odd.
    assert default_column_kind(0, 11) == "1"
    assert default_column_kind(10, 11) == "1"
    assert default_column_kind(5, 11) == "S"


def test_resolve_column_beatmap_tier_wins(tmp_path):
    """BEATMAP > SKIN > FALLBACK. A file in beatmap_dir shadows the
    same-named file in skin_dir."""
    from PIL import Image
    skin_dir = tmp_path / "skin"
    skin_dir.mkdir()
    bm_dir = tmp_path / "map"
    bm_dir.mkdir()
    Image.new("RGBA", (32, 32), (255, 0, 0, 255)).save(skin_dir / "mania-note1.png")
    Image.new("RGBA", (32, 32), (0, 255, 0, 255)).save(bm_dir / "mania-note1.png")
    frames, src = SpriteAtlas._resolve_column(
        kind="note_tap", col=0, key_count=4,
        skin_dir=skin_dir, beatmap_dir=bm_dir, section=None,
    )
    assert src == "beatmap"
    assert len(frames) == 1
    assert frames[0].getpixel((0, 0)) == (0, 255, 0, 255)


def test_resolve_column_falls_back_to_skin_then_bundle(tmp_path):
    """No beatmap file → skin wins. No skin file either → bundle."""
    from PIL import Image
    skin_dir = tmp_path / "skin"
    skin_dir.mkdir()
    bm_dir = tmp_path / "map"
    bm_dir.mkdir()
    Image.new("RGBA", (32, 32), (255, 0, 0, 255)).save(skin_dir / "mania-note1.png")
    frames, src = SpriteAtlas._resolve_column(
        kind="note_tap", col=0, key_count=4,
        skin_dir=skin_dir, beatmap_dir=bm_dir, section=None,
    )
    assert src == "user"
    assert len(frames) == 1

    # Empty beatmap + empty skin → bundle.
    frames, src = SpriteAtlas._resolve_column(
        kind="note_tap", col=0, key_count=4,
        skin_dir=tmp_path / "nonexistent", beatmap_dir=None, section=None,
    )
    assert src == "bundle"
    assert len(frames) == 1


def test_try_animation_frames(tmp_path):
    """Frame discovery picks up contiguous `<base>-N.png` and stops at
    the first missing index."""
    from PIL import Image

    from osu_mania_renderer_v2.gpu.atlas import _try_animation_frames
    skin = tmp_path / "skin"
    skin.mkdir()
    Image.new("RGBA", (16, 16), (255, 0, 0, 255)).save(skin / "mania-hit300-0.png")
    Image.new("RGBA", (16, 16), (0, 255, 0, 255)).save(skin / "mania-hit300-1.png")
    Image.new("RGBA", (16, 16), (0, 0, 255, 255)).save(skin / "mania-hit300-2.png")
    # Gap at -3 → animation stops at 3 frames.
    Image.new("RGBA", (16, 16), (1, 1, 1, 255)).save(skin / "mania-hit300-7.png")
    frames = _try_animation_frames(skin, ("mania-hit300.png",))
    assert len(frames) == 3
    assert frames[0].getpixel((0, 0)) == (255, 0, 0, 255)
    assert frames[1].getpixel((0, 0)) == (0, 255, 0, 255)
    assert frames[2].getpixel((0, 0)) == (0, 0, 255, 255)


def test_try_animation_frames_no_match(tmp_path):
    """No `-0.png` → empty list (caller falls through to single-frame)."""
    from PIL import Image

    from osu_mania_renderer_v2.gpu.atlas import _try_animation_frames
    skin = tmp_path / "skin"
    skin.mkdir()
    # Only the base file, no -N.png frames.
    Image.new("RGBA", (16, 16), (255, 0, 0, 255)).save(skin / "mania-hit300.png")
    assert _try_animation_frames(skin, ("mania-hit300.png",)) == []


def test_resolve_global_animated_skin_hit_burst(tmp_path):
    """An animated hit-burst at skin tier returns the frame list."""
    from PIL import Image
    skin = tmp_path / "skin"
    skin.mkdir()
    Image.new("RGBA", (16, 16), (1, 0, 0, 255)).save(skin / "mania-hit300-0.png")
    Image.new("RGBA", (16, 16), (2, 0, 0, 255)).save(skin / "mania-hit300-1.png")
    Image.new("RGBA", (16, 16), (3, 0, 0, 255)).save(skin / "mania-hit300-2.png")
    frames, src = SpriteAtlas._resolve_global(
        "judgment_300", skin_dir=skin, beatmap_dir=None, section=None,
    )
    assert src == "user"
    assert len(frames) == 3
    assert frames[0].getpixel((0, 0)) == (1, 0, 0, 255)


def test_resolve_global_lighting_n_layered_fallback(tmp_path):
    """lighting_n tries mania-lightingN → lightingN → lighting."""
    from PIL import Image
    skin = tmp_path / "skin"
    skin.mkdir()
    # Only the legacy bare-named `lighting.png` exists.
    Image.new("RGBA", (16, 16), (5, 5, 5, 255)).save(skin / "lighting.png")
    frames, src = SpriteAtlas._resolve_global(
        "lighting_n", skin_dir=skin, beatmap_dir=None, section=None,
    )
    assert src == "user"
    assert len(frames) == 1
    assert frames[0].getpixel((0, 0)) == (5, 5, 5, 255)


def test_resolve_column_animated_skin(tmp_path):
    """Per-column note_tap animation discovery via `<base>-N.png`."""
    from PIL import Image
    skin = tmp_path / "skin"
    skin.mkdir()
    Image.new("RGBA", (16, 16), (1, 0, 0, 255)).save(skin / "mania-note1-0.png")
    Image.new("RGBA", (16, 16), (2, 0, 0, 255)).save(skin / "mania-note1-1.png")
    Image.new("RGBA", (16, 16), (3, 0, 0, 255)).save(skin / "mania-note1-2.png")
    frames, src = SpriteAtlas._resolve_column(
        kind="note_tap", col=0, key_count=4,
        skin_dir=skin, beatmap_dir=None, section=None,
    )
    assert src == "user"
    assert len(frames) == 3
    assert frames[0].getpixel((0, 0)) == (1, 0, 0, 255)


def test_resolve_column_animated_per_column_independence(tmp_path):
    """Two different columns of the same skin can have different animation
    frame counts (one animated, one static)."""
    from PIL import Image
    skin = tmp_path / "skin"
    skin.mkdir()
    # Col 0 (outer = 1): animated.
    for i in range(4):
        Image.new("RGBA", (16, 16), (i, 0, 0, 255)).save(
            skin / f"mania-note1-{i}.png"
        )
    # Col 1 (inner = 2): static only.
    Image.new("RGBA", (16, 16), (0, 99, 0, 255)).save(skin / "mania-note2.png")
    # Default 4K layout: cols 0,3 are "1", cols 1,2 are "2".
    frames0, src0 = SpriteAtlas._resolve_column(
        kind="note_tap", col=0, key_count=4,
        skin_dir=skin, beatmap_dir=None, section=None,
    )
    frames1, src1 = SpriteAtlas._resolve_column(
        kind="note_tap", col=1, key_count=4,
        skin_dir=skin, beatmap_dir=None, section=None,
    )
    assert src0 == "user" and len(frames0) == 4
    assert src1 == "user" and len(frames1) == 1


def test_resolve_column_tail_auto_flip(tmp_path):
    """When the skin only ships the head sprite (no T variant), the tail
    resolver auto-flips the head to mimic stable's behaviour."""
    from PIL import Image
    skin = tmp_path / "skin"
    skin.mkdir()
    # Half-white-top, half-black-bottom — easy to detect the flip.
    src = Image.new("RGBA", (4, 4), (0, 0, 0, 255))
    for x in range(4):
        for y in range(2):
            src.putpixel((x, y), (255, 255, 255, 255))
    src.save(skin / "mania-note1H.png")
    frames, _ = SpriteAtlas._resolve_column(
        kind="note_hold_tail", col=0, key_count=4,
        skin_dir=skin, beatmap_dir=None, section=None,
    )
    assert len(frames) == 1
    # After Y-flip, the top half should be black (was bottom).
    assert frames[0].getpixel((0, 0)) == (0, 0, 0, 255)
    assert frames[0].getpixel((0, 3)) == (255, 255, 255, 255)


def test_resolve_column_tail_flipped_for_downscroll(tmp_path):
    """The hold tail is flipped vertically for a downward-scrolling stage —
    lazer's LegacyHoldNoteTailPiece inverts the scroll direction, so even an
    explicit `mania-noteNT.png` is mirrored (e.g. Night05's flat-top tail
    becomes a rounded-top cap)."""
    from PIL import Image
    skin = tmp_path / "skin"
    skin.mkdir()
    src = Image.new("RGBA", (4, 4), (0, 0, 0, 255))
    for x in range(4):
        for y in range(2):
            src.putpixel((x, y), (255, 255, 255, 255))  # top half white
    src.save(skin / "mania-note1T.png")
    frames, _ = SpriteAtlas._resolve_column(
        kind="note_hold_tail", col=0, key_count=4,
        skin_dir=skin, beatmap_dir=None, section=None,
    )
    assert len(frames) == 1
    # After the vertical flip the white half is now at the BOTTOM.
    assert frames[0].getpixel((0, 0)) == (0, 0, 0, 255)
    assert frames[0].getpixel((0, 3)) == (255, 255, 255, 255)


def test_resolve_column_receptors_not_animated(tmp_path):
    """Per the wiki, key sprites do not support animation. Even if the
    skin provides `mania-key1-0.png` etc., the resolver falls through
    to the static `mania-key1.png` (or bundled)."""
    from PIL import Image
    skin = tmp_path / "skin"
    skin.mkdir()
    Image.new("RGBA", (16, 16), (1, 0, 0, 255)).save(skin / "mania-key1-0.png")
    Image.new("RGBA", (16, 16), (2, 0, 0, 255)).save(skin / "mania-key1-1.png")
    Image.new("RGBA", (16, 16), (9, 9, 9, 255)).save(skin / "mania-key1.png")
    frames, src = SpriteAtlas._resolve_column(
        kind="receptor_off", col=0, key_count=4,
        skin_dir=skin, beatmap_dir=None, section=None,
    )
    # Receptor resolver should ignore the -0/-1 stack and pick the
    # static file.
    assert src == "user"
    assert len(frames) == 1
    assert frames[0].getpixel((0, 0)) == (9, 9, 9, 255)


def test_beatmap_animated_overrides_skin_static(tmp_path):
    """Beatmap-tier animated stack wins over skin-tier static for the
    same slot. All frames must come from the same tier (danser's rule)."""
    from PIL import Image
    skin = tmp_path / "skin"
    skin.mkdir()
    bm = tmp_path / "map"
    bm.mkdir()
    # Skin has only the static base; beatmap ships the animated stack.
    Image.new("RGBA", (16, 16), (1, 1, 1, 255)).save(skin / "mania-hit50.png")
    Image.new("RGBA", (16, 16), (2, 2, 2, 255)).save(bm / "mania-hit50-0.png")
    Image.new("RGBA", (16, 16), (3, 3, 3, 255)).save(bm / "mania-hit50-1.png")
    frames, src = SpriteAtlas._resolve_global(
        "judgment_50", skin_dir=skin, beatmap_dir=bm, section=None,
    )
    assert src == "beatmap"
    assert len(frames) == 2
    assert frames[0].getpixel((0, 0)) == (2, 2, 2, 255)


def test_resolve_column_per_column_override_wins_over_beatmap(tmp_path):
    """Skin's explicit NoteImage{N} override has the highest priority
    even over a beatmap-tier conventional file. Author named the file
    on purpose — honour it."""
    from PIL import Image

    from osu_mania_renderer_v2.beatmap.skin_ini import ManiaSection
    skin_dir = tmp_path / "skin"
    skin_dir.mkdir()
    bm_dir = tmp_path / "map"
    bm_dir.mkdir()
    Image.new("RGBA", (32, 32), (255, 0, 0, 255)).save(skin_dir / "my-note.png")
    Image.new("RGBA", (32, 32), (0, 255, 0, 255)).save(bm_dir / "mania-note1.png")
    section = ManiaSection(keys=4, note_image={0: "my-note"})
    frames, src = SpriteAtlas._resolve_column(
        kind="note_tap", col=0, key_count=4,
        skin_dir=skin_dir, beatmap_dir=bm_dir, section=section,
    )
    assert src == "user"
    assert len(frames) == 1
    assert frames[0].getpixel((0, 0)) == (255, 0, 0, 255)
