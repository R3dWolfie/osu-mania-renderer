"""Three things the frame loop worked out again every frame and now keeps.

  * number layout (wiki_elements/context.py): a per-font, per-size table of
    each character's layout instead of ~8 lookups per character per draw;
  * the replay's mod icons (beatmap/mods.py legacy_mod_icons);
  * the throw of each Argon ring-explosion piece (wiki_elements/notes.py).

What must hold: every draw call and every returned width is exactly what the
per-frame code produced, and the kept values cannot outlive what they were
worked out from.
"""
from __future__ import annotations

import math
import random
from types import SimpleNamespace

import pytest

from osu_mania_renderer_v2.beatmap import mods as M
from osu_mania_renderer_v2.wiki_elements import notes as N
from osu_mania_renderer_v2.wiki_elements.context import FrameContext


class _Atlas:
    """Native sizes and sources for three number fonts. The combo font has
    digits only, so its other glyphs fall back to the score font."""

    def __init__(self, seed=0):
        r = random.Random(seed)
        self.size, self.source, self.index = {}, {}, {}
        names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "comma", "dot", "percent", "x"]
        for font in ("score", "combo", "argon"):
            for i, n in enumerate(names):
                if font == "combo" and not n.isdigit():
                    continue
                if font == "argon" and n == "comma":
                    continue
                slot = f"{font}_{n}"
                self.size[slot] = (r.choice([52, 120, 240, 37.5]), r.choice([300, 96, 41.0]))
                self.source[slot] = "user" if font != "argon" else "bundle"
                self.index[slot] = len(self.index)
        for slot in ("argon_wireframes",):
            self.size[slot] = (240, 300)
            self.index[slot] = len(self.index)

    def global_native_size(self, slot):
        return self.size.get(slot, (0, 0))

    def global_source(self, slot):
        return self.source.get(slot, "missing")

    def index_of(self, slot):
        return self.index[slot]


class _Fr:
    def __init__(self, atlas):
        self.atlas, self.calls = atlas, []

    def _draw_direct(self, *a):
        self.calls.append(("direct",) + a)

    def _draw_sprite_idx(self, *a):
        self.calls.append(("sprite",) + a)


def _ctx(atlas):
    fr = _Fr(atlas)
    return FrameContext(fr=fr, skin=None, gl=None, fbo=None, width=1280, height=720, key_count=4), fr


def _reference(ctx, text, *, x, center_y, glyph_h, overlap_px=0, align="left", alpha=1.0,
               font="score", wireframe=False, tint=(1.0, 1.0, 1.0)):
    """The per-glyph code as it was before the table, kept here word for word."""
    self = ctx

    def number_width(text, glyph_h, overlap_px, font):
        digit_h = self.atlas.global_native_size(f"{font}_0")[1] or 1
        scale = glyph_h / digit_h
        fixed_digit_width = (
            self.atlas.global_native_size(f"{font}_5")[0]
            if font in ("score", "combo")
            else 0.0
        )
        total = 0.0
        prev_gap = 0.0
        for ch in text:
            slot = self._glyph_slot(ch, font)
            if slot is None:
                continue
            gw, _gh = self.atlas.global_native_size(slot)
            cell_width = (
                fixed_digit_width * scale
                if ch.isdigit() and fixed_digit_width > 0
                else gw * scale
            )
            total += cell_width - prev_gap
            prev_gap = self._eff_overlap(overlap_px, gw) * scale
        return total

    calls = []
    digit_h = self.atlas.global_native_size(f"{font}_0")[1] or 1
    scale = glyph_h / digit_h
    fixed_digit_width = (
        self.atlas.global_native_size(f"{font}_5")[0]
        if font in ("score", "combo")
        else 0.0
    )
    total_w = number_width(text, glyph_h, overlap_px, font)
    if align == "right":
        pen_x = x - total_w
    elif align == "center":
        pen_x = x - total_w / 2
    else:
        pen_x = x
    rgba = (tint[0], tint[1], tint[2], alpha)
    for ch in text:
        slot = self._glyph_slot(ch, font)
        if slot is None:
            continue
        gw, gh = self.atlas.global_native_size(slot)
        cell_width = (
            fixed_digit_width * scale
            if ch.isdigit() and fixed_digit_width > 0
            else gw * scale
        )
        draw_slot = slot
        if wireframe and font == "argon":
            draw_slot = "argon_dot" if ch == "." else "argon_wireframes"
        dw, dh = self.atlas.global_native_size(draw_slot)
        cx = pen_x + cell_width / 2
        if font in ("score", "combo") and not wireframe:
            draw_width = dw * scale
            draw_height = dh * scale
            calls.append(("direct", draw_slot,
                          int(round(cx - draw_width / 2)),
                          int(round(center_y - draw_height / 2)),
                          max(1, int(round(draw_width))),
                          max(1, int(round(draw_height))),
                          rgba))
        else:
            q = max(dw, dh) * scale
            idx = self.atlas.index_of(draw_slot)
            calls.append(("sprite", idx, int(round(cx - q / 2)), int(round(center_y - q / 2)),
                          int(round(q)), int(round(q)), rgba))
        pen_x += cell_width - self._eff_overlap(overlap_px, gw) * scale
    return total_w, calls


def test_numbers_are_laid_out_exactly_as_before():
    r = random.Random(7)
    alphabet = "0123456789,.%x a"
    n = 0
    for atlas_seed in range(3):
        ctx, fr = _ctx(_Atlas(atlas_seed))
        for _ in range(700):
            font = r.choice(["score", "combo", "argon"])
            kw = dict(
                x=r.uniform(-50, 2000), center_y=r.uniform(0, 1100),
                glyph_h=r.choice([18.0, 27.5, 36.0, 54.0, r.uniform(8, 90)]),
                overlap_px=r.choice([0, 2, 16, 40, 400]), align=r.choice(["left", "right", "center"]),
                alpha=r.random(), font=font, wireframe=(font == "argon" and r.random() < 0.5),
                tint=(r.random(), r.random(), r.random()))
            text = "".join(r.choice(alphabet) for _ in range(r.randint(0, 12)))
            want_w, want_calls = _reference(ctx, text, **kw)
            fr.calls.clear()
            got_w = ctx.draw_number(text, **kw)
            assert got_w == want_w and fr.calls == want_calls, (font, text, kw)
            assert ctx.number_width(text, kw["glyph_h"], kw["overlap_px"], font) == want_w
            n += len(want_calls)
    assert n > 5000          # the comparison saw real glyphs, of both kinds
    assert {c[0] for c in want_calls} <= {"direct", "sprite"}


def test_number_tables_do_not_pile_up_and_follow_the_atlas():
    ctx, fr = _ctx(_Atlas(0))
    for i in range(500):                      # a number whose size changes every frame
        ctx.draw_number("123", x=0, center_y=0, glyph_h=20.0 + i * 0.01, font="argon")
    assert len(ctx._number_tables[1]) <= 64
    first = ctx._number_tables
    other = _Atlas(5)
    fr.atlas = other
    fr.calls.clear()
    want_w, want_calls = _reference(ctx, "40.5%", x=10, center_y=20, glyph_h=30.0, font="score")
    assert ctx.draw_number("40.5%", x=10, center_y=20, glyph_h=30.0, font="score") == want_w
    assert fr.calls == want_calls
    assert ctx._number_tables is not first and ctx._number_tables[0] is other


def test_mod_icons_kept_are_the_mod_icons_worked_out():
    fresh = M.legacy_mod_icons.__wrapped__
    bits = [int(m) for m in M.Mod]
    seen = set()
    r = random.Random(3)
    for value in [0] + bits + [r.getrandbits(31) for _ in range(400)] + [
            int(M.Mod.NC | M.Mod.DT), int(M.Mod.PF | M.Mod.SD), int(M.Mod.NC | M.Mod.DT | M.Mod.PF | M.Mod.SD)]:
        want = fresh(value)
        assert M.legacy_mod_icons(value) == want
        assert M.legacy_mod_icons(value) is M.legacy_mod_icons(value)      # kept, not rebuilt
        seen.add(want)
    assert len(seen) > 20


def _explode(ctx, fr, judgment, age, t_ms):
    fr.calls.clear()
    ctx.scene = SimpleNamespace(t_ms=t_ms)
    N._argon_ring_explosion(ctx, SimpleNamespace(judgment=judgment, age_ms=age), 640.0, 360.0, (1.0, 0.5, 0.25))
    return list(fr.calls)


def test_ring_throws_kept_are_the_throws_drawn_before(monkeypatch):
    class _Rc:
        height = 720
        ctx = SimpleNamespace(blend_func=None)

    class _RingFr:
        rc = _Rc()

        def __init__(self):
            self.calls = []

        def _flush_sprite_batch(self):
            pass

        def _draw_external_texture(self, tex, **kw):
            self.calls.append((tex, tuple(sorted(kw.items()))))

    monkeypatch.setattr(N, "_cached_ring_tex", lambda fr, col, size, thick: (("ring", size, thick), 24, 24))
    fr = _RingFr()
    ctx = SimpleNamespace(fr=fr, scene=None)
    total = 0
    for judgment, (n_small, n_large, tmult) in N._ARGON_RING_SPEC.items():
        for press in (0, 1234, 98765):
            for age in (0.0, 16.0, 250.0, 599.0, 600.0, 900.0):
                N._RING_THROW.clear()
                cold = _explode(ctx, fr, judgment, age, press + age)
                warm = _explode(ctx, fr, judgment, age, press + age)
                shown = (max(0.0, 1.0 - age / 1000.0)) ** 5 > 0.003      # faded out by then: nothing drawn
                assert cold == warm and len(cold) == (n_small + n_large if shown else 0)
                # and both are what a generator seeded per piece per frame gives
                tf = 40.0 * 720 / 1080.0
                travel = 52.0 / 28.0 * tf * tmult
                radius_frac = 0.3 + 0.7 * (1.0 - (1.0 - min(age, 600) / 600.0) ** 5)
                seed_base = int((press + age) - age)
                sizes = [9.0 / 28.0 * tf] * n_small + [14.0 / 28.0 * tf] * n_large
                for i, (call, size) in enumerate(zip(cold, sizes)):
                    rng = random.Random((seed_base * 1000003) ^ (i * 2654435761) ^ (hash(judgment) & 0xFFFF))
                    direction = rng.uniform(0.0, 360.0)
                    distance = rng.uniform(travel / 2.0, travel)
                    cur = distance * radius_frac
                    kw = dict(call[1])
                    assert kw["x"] == int(640.0 + math.cos(direction) * cur - 24 / 2.0)
                    assert kw["y"] == int(360.0 + math.sin(direction) * cur - 24 / 2.0)
                total += len(cold)
    assert total > 100
    N._RING_THROW.clear()
    for k in range(5000):                      # the store is bounded
        _explode(ctx, fr, next(iter(N._ARGON_RING_SPEC)), 10.0, k * 1000 + 10.0)
    assert len(N._RING_THROW) <= 4096
