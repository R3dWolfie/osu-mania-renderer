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


# Sentinel: a variable spec WITHOUT a wiki_default forces a RenderError
# when the skin doesn't supply it. Compare with `is MISSING` only.
MISSING: Any = object()


class RenderError(RuntimeError):
    """Raised when a required value cannot be resolved anywhere.

    Carries `element`, `variable`, and the ordered `searched` list so a
    log line is enough to diagnose without re-running the render."""

    def __init__(
        self,
        *,
        element: str,
        variable: str,
        searched: list[str],
        hint: str | None = None,
    ) -> None:
        self.element = element
        self.variable = variable
        self.searched = list(searched)
        self.hint = hint
        msg = (
            f"render error: cannot resolve '{variable}' for element "
            f"'{element}'. Looked in: " + " | ".join(searched)
        )
        if hint:
            msg += f" — {hint}"
        super().__init__(msg)


class SkinIni:
    """Tolerant skin.ini reader. Sections + 'Key: value' lines, with
    repeated [Mania] blocks bucketed by `Keys:` count so per-keymode
    overrides resolve cleanly."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._sections: dict[str, dict[str, str]] = {}
        self._mania_by_keys: dict[int, dict[str, str]] = {}
        if path is not None and path.is_file():
            self._parse(path.read_text(encoding="utf-8", errors="replace"))

    def _parse(self, text: str) -> None:
        section = "_root"
        cur = self._sections.setdefault(section, {})
        mania_blocks: list[dict[str, str]] = []
        for raw in text.splitlines():
            line = raw.split("//", 1)[0].strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                name = line[1:-1]
                if name == "Mania":
                    cur = {}
                    mania_blocks.append(cur)
                    section = name
                else:
                    section = name
                    cur = self._sections.setdefault(name, {})
                continue
            if ":" in line:
                k, v = line.split(":", 1)
                cur[k.strip()] = v.strip()
        for block in mania_blocks:
            try:
                k = int(block.get("Keys", "0"))
            except ValueError:
                continue
            if k > 0:
                self._mania_by_keys[k] = block

    def get(self, key: str, *, section: str | None = None) -> str | None:
        target = "_root" if section is None else section
        return self._sections.get(target, {}).get(key)

    def get_mania(self, keys: int, key: str) -> str | None:
        return self._mania_by_keys.get(keys, {}).get(key)


@dataclass
class SkinPair:
    """User skin + default skin, with skin.ini files pre-parsed.
    Every asset and variable lookup in the pipeline goes through this
    object so the fall-back order is consistent."""

    user_dir: Path
    default_dir: Path
    user_ini: SkinIni = field(init=False)
    default_ini: SkinIni = field(init=False)

    def __post_init__(self) -> None:
        self.user_ini = SkinIni(self.user_dir / "skin.ini")
        self.default_ini = SkinIni(self.default_dir / "skin.ini")

    def resolve_asset(
        self, basename: str, *, extensions: tuple[str, ...] = (".png", ".jpg"),
    ) -> Path | None:
        """user skin → default skin → None. Honors the @2x high-DPI suffix
        because the wiki treats `name@2x.png` as the preferred variant
        when present."""
        for root in (self.user_dir, self.default_dir):
            for stem in (f"{basename}@2x", basename):
                for ext in extensions:
                    p = root / f"{stem}{ext}"
                    if p.is_file():
                        return p
        return None

    def resolve_variable(
        self,
        *,
        element: str,
        key: str,
        section: str | None = None,
        mania_keys: int | None = None,
        wiki_default: Any = MISSING,
    ) -> Any:
        """user skin.ini → default skin.ini → wiki_default → RenderError.
        Pass `mania_keys=N` to read from the [Mania] block for that key
        count instead of a named section."""
        searched: list[str] = []
        for ini, label in (
            (self.user_ini, "user"),
            (self.default_ini, "default"),
        ):
            if mania_keys is not None:
                searched.append(f"{label}.ini [Mania Keys={mania_keys}].{key}")
                val = ini.get_mania(mania_keys, key)
            else:
                where = section or "root"
                searched.append(f"{label}.ini [{where}].{key}")
                val = ini.get(key, section=section)
            if val is not None:
                return val
        searched.append("wiki default")
        if wiki_default is not MISSING:
            return wiki_default
        raise RenderError(
            element=element, variable=key, searched=searched,
            hint="add a wiki_default to the VariableSpec if this is intended",
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
    # Heavy GPU deps imported lazily so `import osu_mania_renderer_v2.wiki_elements`
    # (to populate registries) stays cheap.
    import asyncio
    from osu_mania_renderer_v2.render.encode import FfmpegPipe
    from osu_mania_renderer_v2.errors import RenderTimeoutError
    from osu_mania_renderer_v2.gpu.context import HeadlessGl
    from osu_mania_renderer_v2.gpu.readback import FrameReader
    from osu_mania_renderer_v2.gpu.renderer import FrameRenderer, RenderContext
    from osu_mania_renderer_v2.render.render import build_frame_state, build_render_plan
    from osu_mania_renderer_v2.wiki_elements.context import FrameContext

    if not RENDER_ORDER:
        raise RenderError(
            element="<pipeline>", variable="RENDER_ORDER",
            searched=["wiki_renderer.RENDER_ORDER"],
            hint="import osu_mania_renderer_v2.wiki_elements to populate RENDER_ORDER",
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
    """Admin test CLI: `python -m osu_mania_renderer_v2.wiki_renderer OSR BEATMAP_DIR
    -o OUT --skin-dir DIR [--default-skin DIR] [legacy flags]`. Builds a
    RenderOptions from the flags and drives the async `render`."""
    import asyncio
    from osu_mania_renderer_v2.beatmap.models import RenderOptions

    p = argparse.ArgumentParser(description="wiki-driven renderer (admin test path)")
    p.add_argument("osr", type=Path, help=".osr replay file")
    p.add_argument("beatmap_dir", type=Path,
                   help="dir containing the matching .osu and assets")
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--skin-dir", type=Path, default=None,
                   help="user skin dir (extracted .osk)")
    p.add_argument("--default-skin", type=Path, default=None,
                   help="default skin dir; auto-detects assets/default_skin "
                        "relative to the package when omitted")
    p.add_argument("--resolution", default="1920x1080")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--encoder", default="auto")
    p.add_argument("--timeout", type=int, default=0)
    p.add_argument("--allow-converted", action="store_true")
    p.add_argument("--convert-to-keys", type=int, default=4)
    p.add_argument("--no-combo", action="store_true", help="hide combo counter")
    p.add_argument("--no-judgment", action="store_true",
                   help="hide hit-judgement text/sprite")
    p.add_argument("--no-key-counter", action="store_true",
                   help="hide bottom-right key counter")
    # Background dim (same names/mapping as cli.py — the bot's
    # mania_ordr/renderer.py has ALWAYS sent these four flags, but only
    # cli.py declared them, so this prod path silently dropped them via
    # parse_known_args and every render used the 0.70 background_dim
    # default). ints are 0-100 preset values → RenderOptions fractions;
    # --bg-dim is the legacy single float 0-1.
    p.add_argument("--bg-dim",          type=float, default=None,
                   help="background dim 0.0 - 1.0 (legacy single-value)")
    p.add_argument("--bg-dim-intro",    type=int, default=None,
                   help="background dim during intro 0-100 (overrides --bg-dim)")
    p.add_argument("--bg-dim-game",     type=int, default=None,
                   help="background dim during gameplay 0-100")
    p.add_argument("--bg-dim-breaks",   type=int, default=None,
                   help="background dim during breaks 0-100")
    # NB: must be declared HERE (not only in cli.py) — parse_known_args
    # silently drops unknown flags, so an undeclared --logo would no-op.
    p.add_argument("--logo", action="store_true",
                   help="show_logo: the R3D 'R' tile splash during the "
                        "intro, fading out as the first note spawns "
                        "(parity with std/catch)")
    p.add_argument("--featured-avatar-png", type=Path, default=None,
                   help="featured player's osu! avatar PNG → results-screen "
                        "header (parity with std). Absent ⇒ grey placeholder")
    # Exact official-PP override (parity with the taiko renderer). When the
    # caller supplies the authoritative passed pp, the results card + live
    # counter show it exactly instead of the rosu-pp estimate. Must be
    # declared HERE - parse_known_args silently drops undeclared flags.
    p.add_argument("--pp", type=float, default=None,
                   help="exact official PP for the results card / live "
                        "counter (overrides the rosu-pp estimate); "
                        "omit to use rosu")
    # --sr (parity with taiko/std/catch). Must be declared HERE too -
    # parse_known_args silently drops undeclared flags. mania draws no SR
    # on-card yet, so this is accepted + stored but currently inert.
    p.add_argument("--sr", type=float, default=None,
                   help="exact official star rating (parity with taiko/std/"
                        "catch); accepted + stored, no visible effect yet")
    args, _unknown = p.parse_known_args()

    default_skin = args.default_skin
    if default_skin is None:
        default_skin = Path(__file__).resolve().parent / "assets" / "default_skin"

    if args.skin_dir is None:
        import tempfile
        args.skin_dir = Path(tempfile.mkdtemp(prefix="argon-empty-"))

    import osu_mania_renderer_v2.wiki_elements  # noqa: F401 — populate registries
    from osu_mania_renderer_v2.beatmap.models import RenderOptions as _RO  # noqa: F811
    w, h = (int(x) for x in args.resolution.lower().split("x"))
    # Stage-aware bg dim (cli.py's exact clamping/mapping): omitted flags
    # keep the RenderOptions defaults (background_dim 0.70 fallback).
    dim_kwargs: dict = {}
    if args.bg_dim is not None:
        dim_kwargs["background_dim"] = max(0.0, min(1.0, args.bg_dim))
    if args.bg_dim_intro is not None:
        dim_kwargs["bg_dim_intro"] = max(0, min(100, args.bg_dim_intro)) / 100.0
    if args.bg_dim_game is not None:
        dim_kwargs["bg_dim_game"] = max(0, min(100, args.bg_dim_game)) / 100.0
    if args.bg_dim_breaks is not None:
        dim_kwargs["bg_dim_breaks"] = max(0, min(100, args.bg_dim_breaks)) / 100.0
    options = RenderOptions(
        resolution=(w, h), fps=args.fps, encoder=args.encoder,
        timeout_seconds=(args.timeout or 600),
        show_combo=not args.no_combo,
        show_judgment=not args.no_judgment,
        show_key_counter=not args.no_key_counter,
        show_logo=args.logo,
        featured_avatar_png=(str(args.featured_avatar_png)
                             if args.featured_avatar_png else None),
        pp_override=args.pp,
        sr_override=args.sr,
        **dim_kwargs,
    )
    async def _print_progress(fraction: float) -> None:
        # Same line shape the bot's _PROGRESS_RE parses from the subprocess
        # stdout (mania_ordr/renderer.py). Without this the wiki path emitted
        # no progress and the UI sat at 0% for the entire render.
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
