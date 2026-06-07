#!/usr/bin/env python3
"""Fast sanity check for skin.ini parsing + atlas resolution.

Run before deploying any change to `skin_ini.py`, `gpu/atlas.py`, or
`gpu/renderer.py`. Walks every skin.ini under the configured skin root
and parses it, then resolves a small set of atlas slots without
needing GL. Fails loudly on any AttributeError, KeyError, or other
parse-side regression.

Default skin root: `/var/mnt/ASUStor-Samsung/R3DManiaORDRBot/skins/`.
Override via `--skin-root <path>` or `R3D_SKIN_ROOT` env var.

Usage:
    python3 scripts/parse_smoke.py
    R3D_SKIN_ROOT=/some/path python3 scripts/parse_smoke.py

Exit status: 0 = all skins parse + resolve cleanly; 1 = any failure.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the renderer package importable when running from a checkout.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from osu_mania_renderer_v2.gpu.atlas import (  # noqa: E402
    PER_COLUMN_KINDS,
    SpriteAtlas,
    default_column_kind,
)
from osu_mania_renderer_v2.skin_ini import parse_skin_ini  # noqa: E402

DEFAULT_SKIN_ROOT = Path("/var/mnt/ASUStor-Samsung/R3DManiaORDRBot/skins")


def _smoke_one(skin_dir: Path) -> list[str]:
    """Run the parser + atlas resolver pipeline for one skin. Returns
    a list of failure messages (empty = success)."""
    failures: list[str] = []
    try:
        ini = parse_skin_ini(skin_dir)
    except Exception as e:
        return [f"parse_skin_ini raised {type(e).__name__}: {e}"]
    # Cross-section field access — same call path `FrameRenderer.__init__`
    # uses. Any AttributeError here is a parser-vs-dataclass drift bug.
    for keys in (4, 5, 6, 7, 8):
        section = ini.mania_for_keycount(keys)
        if section is None:
            continue
        try:
            _ = (
                section.colour, section.colour_light,
                section.note_image, section.note_image_h,
                section.note_image_l, section.note_image_t,
                section.key_image, section.key_image_d,
                section.stage_left, section.stage_bottom,
                section.hit_0, section.hit_300, section.hit_300g,
                section.column_start, section.column_width,
                section.column_spacing, section.column_line_width,
                section.hit_position, section.score_position,
                section.combo_position, section.special_style,
                section.keys_under_notes, section.upside_down,
                section.light_frame_per_second,
            )
        except AttributeError as e:
            failures.append(f"K={keys} section missing attr: {e}")
    # Resolver pipeline (no GL). Verifies _resolve_column +
    # _resolve_global don't crash on real skins.
    for keys in (4, 7):
        section = ini.mania_for_keycount(keys)
        for kind in PER_COLUMN_KINDS:
            for col in range(keys):
                try:
                    frames, src = SpriteAtlas._resolve_column(
                        kind=kind, col=col, key_count=keys,
                        skin_dir=skin_dir, beatmap_dir=None,
                        section=section,
                    )
                    assert isinstance(frames, list) and frames, (
                        f"empty frame list for ({kind}, {col})"
                    )
                except Exception as e:
                    failures.append(
                        f"K={keys} ({kind}, col={col}): "
                        f"{type(e).__name__}: {e}"
                    )
        # Spot-check a couple of global slots too.
        for slot in ("judgment_300", "stage_light", "lighting_n"):
            try:
                frames, src = SpriteAtlas._resolve_global(
                    slot, skin_dir=skin_dir, beatmap_dir=None,
                    section=section,
                )
                assert isinstance(frames, list) and frames
            except Exception as e:
                failures.append(
                    f"K={keys} global {slot!r}: {type(e).__name__}: {e}"
                )
    return failures


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--skin-root",
        default=os.environ.get("R3D_SKIN_ROOT") or str(DEFAULT_SKIN_ROOT),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="limit number of skins checked (0 = no limit). useful "
             "when iterating locally.",
    )
    args = p.parse_args(argv)

    root = Path(args.skin_root)
    if not root.is_dir():
        print(f"skin root not a directory: {root}", file=sys.stderr)
        return 2

    ini_paths = sorted(root.rglob("skin.ini"))
    if args.limit:
        ini_paths = ini_paths[: args.limit]
    if not ini_paths:
        print(f"no skin.ini files found under {root}", file=sys.stderr)
        return 2

    ok = 0
    failed = 0
    for ini_path in ini_paths:
        skin_dir = ini_path.parent
        failures = _smoke_one(skin_dir)
        if failures:
            failed += 1
            label = skin_dir.relative_to(root)
            print(f"FAIL  {label}")
            for line in failures:
                print(f"      {line}")
        else:
            ok += 1

    print(f"\nparse_smoke: ok={ok} failed={failed} of {len(ini_paths)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
