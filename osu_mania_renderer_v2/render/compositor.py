"""Wiki-driven osu! renderer.

Every visible pixel traces to one of: a user-skin asset, a default-skin
asset, or a wiki-documented default value. Nothing else is allowed —
unresolved variables raise RenderError with the full lookup trace.

PIPELINE — for each element in RENDER_ORDER:

  1. Resolve assets
     - look in user skin   for the declared basename(s)
     - else default skin
     - else SKIP the element (no draw call)

  2. Resolve variables
     - look in user skin.ini
     - else default skin.ini
     - else use the wiki default
     - else raise RenderError with everywhere we looked

  3. Dispatch to the element's render_fn with the resolved bag

After all elements have run, the canvas reflects the correct back-to-front
layering because RENDER_ORDER is the painter's-algorithm sequence.

Wiki references that should drive the populated registries:
  - https://osu.ppy.sh/wiki/en/Skinning/skin.ini
  - https://osu.ppy.sh/wiki/en/Skinning/osu%21mania
  - https://osu.ppy.sh/wiki/en/Client/File_formats/osk_%28file_format%29
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


from osu_mania_renderer_v2.skin.mania_skin import (  # skin provider + primitives
    MISSING,
    ManiaSkin,
    RenderError,
    SkinIni,
    SkinPair,
)


@dataclass(frozen=True)
class VariableSpec:
    """One variable a render function needs.

    `key`           — the skin.ini key (e.g. 'ColumnWidth')
    `section`       — section name in skin.ini; None = root; ignored when
                      mania_aware is True
    `mania_aware`   — read from the [Mania] block matching the active
                      key count, not from `section`
    `wiki_default`  — used when neither skin defines `key`; set to
                      MISSING to make absence a hard error
    """

    key: str
    section: str | None = None
    mania_aware: bool = False
    wiki_default: Any = MISSING


@dataclass(frozen=True)
class ElementSpec:
    """One renderable element.

    `asset_basenames` — sprite filenames as documented by the wiki. The
        renderer resolves each one; if ALL of them are missing the
        element is skipped entirely (no draw call). Pass an empty tuple
        for elements that draw without a sprite (e.g. solid HUD text,
        background dim).
    `variables`       — every variable the render_fn needs. The pipeline
        resolves each one before dispatch.
    `render_fn`       — the actual draw call. Receives the resolved
        bag (see _stub for the signature).
    """

    name: str
    asset_basenames: tuple[str, ...]
    variables: tuple[VariableSpec, ...]
    render_fn: Callable[..., None]


# ---- registries --------------------------------------------------------
#
# RENDER_ORDER is the painter's-algorithm draw order. The first entry is
# drawn first (back layer), each later entry covers earlier ones.
#
# ELEMENTS maps each name in RENDER_ORDER to its spec. Both are empty
# until the wiki is wired in — populate them via register_element() or
# directly from wiki_elements/ modules.

RENDER_ORDER: list[str] = []

ELEMENTS: dict[str, ElementSpec] = {}


def register_element(name: str, spec: ElementSpec) -> None:
    """Register one render element. Appends *name* to RENDER_ORDER and
    stores *spec* in ELEMENTS. Idempotent on re-registration with the
    same name — the existing entry is updated in-place."""
    ELEMENTS[name] = spec
    if name not in RENDER_ORDER:
        RENDER_ORDER.append(name)


def _stub(
    *,
    element: str,
    skin: SkinPair,
    assets: dict[str, Path | None],
    variables: dict[str, Any],
    ctx: Any,
) -> None:
    """Default render_fn until the wiki defines the real one. Prints what
    the pipeline resolved so you can verify the lookup chain end-to-end
    before any drawing exists."""
    asset_str = {k: (str(v) if v else None) for k, v in assets.items()}
    print(
        f"[wiki_renderer] {element}: assets={asset_str}, "
        f"variables={variables}",
    )


# ---- entry point -------------------------------------------------------


def _resolve_element(skin: "SkinPair", name: str, key_count: int):
    """Resolve one element's assets + variables once (cached for the run).
    Returns (spec, assets, variables) or (spec, None, None) if the element
    declares sprite assets and NONE resolve (wiki rule: drop the element).
    Sprite presence is the atlas's job for atlas-backed elements; this
    SkinPair-level resolution only matters for elements that declare
    `asset_basenames` and read `variables`."""
    spec = ELEMENTS.get(name)
    if spec is None:
        raise RenderError(
            element=name, variable="<spec>",
            searched=["wiki_renderer.ELEMENTS"],
            hint=f"add ElementSpec for '{name}' or remove it from RENDER_ORDER",
        )
    assets: dict[str, Path | None] = {
        bn: skin.resolve_asset(bn) for bn in spec.asset_basenames
    }
    if spec.asset_basenames and not any(assets.values()):
        return spec, None, None  # skip marker
    variables: dict[str, Any] = {}
    for vs in spec.variables:
        variables[vs.key] = skin.resolve_variable(
            element=name, key=vs.key, section=vs.section,
            mania_keys=key_count if vs.mania_aware else None,
            wiki_default=vs.wiki_default,
        )
    return spec, assets, variables


async def render(
    *,
    osr_path: Path,
    beatmap_dir: Path,
    output_path: Path,
    options,
    skin_dir: Path,
    default_skin_dir: Path,
    progress_callback=None,
    allow_converted: bool = False,
    convert_to_keys: int = 4,
) -> None:
    """Wiki-driven render. Reuses the proven gameplay/setup core
    (`build_render_plan` + `build_frame_state`) and the GPU engine
    (`FrameRenderer`/`FrameReader`/`FfmpegPipe`); the ONLY difference from
    `render_mania` is that each frame is composed by dispatching the
    RENDER_ORDER element registry through a `FrameContext` instead of one
    monolithic `FrameRenderer.draw()`."""
    # Heavy GPU deps imported lazily so `import osu_mania_renderer_v2.render.pipeline`
    # (to populate registries) stays cheap.
    import asyncio
    from osu_mania_renderer_v2.render.encode import FfmpegPipe
    from osu_mania_renderer_v2.errors import RenderTimeoutError
    from osu_mania_renderer_v2.gpu.context import HeadlessGl
    from osu_mania_renderer_v2.gpu.readback import FrameReader
    from osu_mania_renderer_v2.gpu.renderer import FrameRenderer, RenderContext
    from osu_mania_renderer_v2.render.render import build_frame_state, build_render_plan
    from osu_mania_renderer_v2.render.frame_context import FrameContext

    if not RENDER_ORDER:
        raise RenderError(
            element="<pipeline>", variable="RENDER_ORDER",
            searched=["wiki_renderer.RENDER_ORDER"],
            hint="import osu_mania_renderer_v2.render.pipeline to populate RENDER_ORDER",
        )

    skin = SkinPair(user_dir=skin_dir, default_dir=default_skin_dir)
    plan = await build_render_plan(
        osr_path=osr_path, beatmap_dir=beatmap_dir, output_path=output_path,
        options=options, skin_dir=skin_dir, allow_converted=allow_converted,
        convert_to_keys=convert_to_keys,
    )

    # Resolve each element's assets + variables ONCE (cached for the run).
    resolved = {name: _resolve_element(skin, name, plan.key_count)
                for name in RENDER_ORDER}

    pipe = FfmpegPipe(plan.ffmpeg_cmd, fifo_path=plan.fifo_path)
    await pipe.start()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + options.timeout_seconds
    try:
        with HeadlessGl(width=options.resolution[0], height=options.resolution[1]) as gl:
            rc = RenderContext(
                ctx=gl.ctx, fbo=gl.fbo,
                width=options.resolution[0], height=options.resolution[1],
                key_count=plan.key_count,
            )
            fr = FrameRenderer(
                rc, options, skin_dir=skin_dir, beatmap_dir=beatmap_dir,
                first_note_ms=plan.first_note_ms,
                # bg dim envelope inputs (dim.py): modded-time note starts +
                # break periods, and the scroll-speed-scaled approach window.
                note_starts=plan.note_times,
                breaks=getattr(plan.modded, "breaks", ()),
                approach_ms=plan.effective_approach_ms,
                # break overlay clock: real/video time -> map time
                rate=plan.audio_rate,
            )
            if plan.bg_path and plan.bg_path.exists():
                fr.set_background(plan.bg_path)
            fr.set_banner_text(plan.banner_text)
            # LAZER RESULTS SCREEN data (hud/lazer_results.py — the ported
            # osu!(lazer) ranking screen; the results_overlay element draws
            # it via fr._draw_results_overlay). Fail-soft: unset data keeps
            # the legacy argon card — loudly.
            try:
                from osu_mania_renderer_v2.hud.lazer_results import (
                    results_data_from_plan,
                )
                fr.set_results_data(results_data_from_plan(plan))
            except Exception:  # noqa: BLE001 — results data never kills a render
                import traceback
                print("[mania-renderer] !!! lazer results data plumbing "
                      "failed — legacy results card this render:")
                traceback.print_exc()
            reader = FrameReader(gl.ctx, gl.fbo, components=3)
            fctx = FrameContext(
                fr=fr, skin=skin, gl=gl.ctx, fbo=gl.fbo,
                width=options.resolution[0], height=options.resolution[1],
                key_count=plan.key_count,
            )

            score_smoothed = 0.0
            accuracy_smoothed = 100.0
            last_progress_t = 0.0
            for frame_n in range(plan.total_frames):
                if loop.time() > deadline:
                    raise RenderTimeoutError(
                        f"render exceeded {options.timeout_seconds}s"
                    )
                t_ms = int(frame_n * 1000 / options.fps)
                scene_full, score_smoothed, accuracy_smoothed = build_frame_state(
                    plan, t_ms, score_smoothed, accuracy_smoothed,
                )
                fctx.scene = scene_full
                fctx.t_ms = t_ms
                fctx.frame_n = frame_n
                fctx.begin_frame()
                for name in RENDER_ORDER:
                    spec, assets, variables = resolved[name]
                    if assets is None:  # skipped (no asset resolved anywhere)
                        continue
                    spec.render_fn(
                        element=name, skin=skin, assets=assets,
                        variables=variables, ctx=fctx,
                    )
                fctx.flush()
                frame = reader.read()
                await pipe.write_frame(frame)
                if progress_callback and (loop.time() - last_progress_t > 0.5):
                    await progress_callback(frame_n / plan.total_frames)
                    last_progress_t = loop.time()

            for tail_frame in reader.drain():
                await pipe.write_frame(tail_frame)
            if progress_callback:
                await progress_callback(1.0)
        await pipe.close(output_path)
    except BaseException:
        if pipe.proc and pipe.proc.returncode is None:
            try:
                pipe.proc.kill()
            except ProcessLookupError:
                pass
        raise
    finally:
        if plan.hitsound_wav is not None:
            try:
                plan.hitsound_wav.unlink(missing_ok=True)
            except OSError:
                pass


def _cli() -> None:
    """Prod/worker CLI: `python -m osu_mania_renderer_v2.wiki_renderer OSR
    BEATMAP_DIR -o OUT --skin-dir DIR [--default-skin DIR] [flags]`.

    Reuses cli.py's FULL parser + ``build_render_options`` so this entrypoint
    (what the bot/worker actually invokes) and the monolith CLI can never drift
    on which flags they honour. Previously this had its own ~20-flag parser +
    ``parse_known_args``, silently dropping ~22 bot-sent flags (--watermark,
    --scroll-speed, volumes, every visibility toggle, --show-pp, ...), so users'
    settings and the free-tier watermark were thrown away. Now every declared
    flag is honoured; genuinely-unknown flags are WARN-logged, not silently
    eaten.
    """
    import asyncio
    import logging
    import tempfile
    # Lazy import breaks the compositor <-> cli <-> __init__ module cycle.
    from osu_mania_renderer_v2.cli import _build_parser, build_render_options

    args, unknown = _build_parser().parse_known_args()
    if unknown:
        logging.getLogger("osu_mania_renderer_v2").warning(
            "wiki CLI ignored unknown flag(s) — declare them in cli.py "
            "_build_parser to honour them: %s", " ".join(unknown))

    default_skin = args.default_skin
    if default_skin is None:
        default_skin = Path(__file__).resolve().parent.parent / "assets" / "default_skin"
    if args.skin_dir is None:
        args.skin_dir = Path(tempfile.mkdtemp(prefix="argon-empty-"))

    import osu_mania_renderer_v2.render.pipeline  # noqa: F401 — populate registries
    options = build_render_options(args)

    async def _print_progress(fraction: float) -> None:
        # Same line shape the bot's _PROGRESS_RE parses from subprocess stdout.
        print(f"\rrendering… {fraction:.0%}", end="", flush=True)

    asyncio.run(render(
        osr_path=args.osr, beatmap_dir=args.beatmap_dir,
        output_path=args.output, options=options,
        skin_dir=args.skin_dir, default_skin_dir=default_skin,
        allow_converted=args.allow_converted,
        convert_to_keys=args.convert_to_keys,
        progress_callback=_print_progress,
    ))


if __name__ == "__main__":
    import importlib
    importlib.import_module(__spec__.name)._cli()
