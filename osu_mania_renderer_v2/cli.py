"""CLI: osu-renderer in.osr beatmap_dir/ -o out.mp4"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from osu_mania_renderer_v2 import RenderOptions, render_mania


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="osu-renderer",
        description="Render an osu! .osr replay to MP4.",
    )
    p.add_argument("osr", type=Path, help=".osr replay file")
    p.add_argument(
        "beatmap_dir", type=Path,
        help="directory containing the .osu, audio.mp3, and background",
    )
    p.add_argument(
        "-o", "--output", type=Path, default=Path("out.mp4"),
        help="output MP4 path (default: out.mp4)",
    )
    p.add_argument(
        "--resolution", default="1920x1080",
        help="WxH, e.g. 1280x720",
    )
    p.add_argument("--fps", type=int, default=60)
    p.add_argument(
        "--encoder", default="auto",
        choices=["auto", "h264_vaapi", "h264_nvenc", "libx264"],
    )
    p.add_argument(
        "--encoder-device", default=None,
        help="VAAPI device (e.g. /dev/dri/renderD128)",
    )
    p.add_argument(
        "--timeout", type=int, default=600,
        help="render timeout in seconds",
    )
    p.add_argument(
        "--skin-dir", type=Path, default=None,
        help="path to an extracted .osk directory (overrides bundled sprites)",
    )
    # Settings-page surface. Each flag maps 1:1 to a RenderOptions field;
    # `store_true` flips a default-on toggle off. Defaults match
    # RenderOptions' own defaults so omitting a flag preserves prior behaviour.
    p.add_argument("--bg-dim",          type=float, default=None,
                   help="background dim 0.0 - 1.0 (legacy single-value)")
    p.add_argument("--bg-dim-intro",    type=int, default=None,
                   help="background dim during intro 0-100 (overrides --bg-dim)")
    p.add_argument("--bg-dim-game",     type=int, default=None,
                   help="background dim during gameplay 0-100")
    p.add_argument("--bg-dim-breaks",   type=int, default=None,
                   help="background dim during breaks 0-100")
    p.add_argument("--bg-blur",         type=int, default=None,
                   help="background blur 0-10")
    p.add_argument("--scroll-speed",    type=int, default=None,
                   help="osu!mania scroll-speed 1-40 (lazer default 17, higher = faster)")
    p.add_argument("--music-volume",    type=int, default=None,
                   help="music volume 0-100 (default: 100)")
    p.add_argument("--hitsound-volume", type=int, default=None,
                   help="hitsound volume 0-100 (default: 100)")
    p.add_argument("--combo-break-threshold", type=int, default=None,
                   help="combo at which 'break' SFX plays on miss (default: 20)")
    p.add_argument("--audio-fade-out-ms", type=int, default=None,
                   help="end-of-song audio fade duration (default: 600)")
    p.add_argument("--no-hp-bar",       action="store_true", help="hide HP bar")
    p.add_argument("--no-hit-error",    action="store_true", help="hide hit-error meter")
    p.add_argument("--no-ur",           action="store_true", help="hide unstable rate")
    p.add_argument("--no-progress",     action="store_true", help="hide progress bar")
    p.add_argument("--no-combo-pop",    action="store_true", help="disable combo pop animation")
    p.add_argument("--no-miss-shake",   action="store_true", help="disable miss-shake")
    p.add_argument("--no-loudnorm",     action="store_true", help="skip ffmpeg loudnorm pass")
    p.add_argument("--no-combo-break",  action="store_true", help="skip combo-break SFX")
    p.add_argument("--no-score",        action="store_true", help="hide score readout")
    p.add_argument("--no-grade",        action="store_true", help="hide grade letter")
    p.add_argument("--no-key-overlay",  action="store_true", help="hide receptor key flash")
    p.add_argument("--no-key-counter",  action="store_true", help="hide bottom-right key-press counter")
    p.add_argument("--no-result-screen",action="store_true", help="cut the results card")
    p.add_argument("--show-pp",         action="store_true", help="show live PP counter")
    p.add_argument("--pp",              type=float, default=None,
                   help="exact official PP for the results card / live "
                        "counter (overrides the rosu-pp estimate); "
                        "omit to use rosu")
    p.add_argument("--hide-judgement-line", action="store_true",
                   help="hide the horizontal line at the receptor")
    p.add_argument("--no-skip-intro",   action="store_true",
                   help="don't skip the audio lead-in silence")
    p.add_argument("--no-replay-hitsounds", action="store_true",
                   help="don't dub per-note hitsounds (song only)")
    p.add_argument("--nightcore-hitsounds", action="store_true",
                   help="layer NC-style claps/finishes on each measure")
    p.add_argument("--skin-hitsounds", action="store_true",
                   help="prefer hitsounds from the user .osk over bundled")
    p.add_argument("--no-beatmap-hitsounds", action="store_true",
                   help="ignore the beatmap's custom hitsounds (skin/default only)")
    p.add_argument("--no-miss-hitsound", action="store_true",
                   help="silence the combobreak (miss) hitsound")
    p.add_argument("--skin-combo-colors", default=None,
                   choices=["beatmap", "skin"],
                   help="combo-color source (beatmap or skin)")
    p.add_argument("--logo", action="store_true",
                   help="show the R3D 'R' tile splash during the intro, "
                        "fading out as the first note spawns "
                        "(parity with std/catch)")
    p.add_argument("--featured-avatar-png", type=Path, default=None,
                   help="featured player's osu! avatar PNG → results-screen "
                        "header (parity with std). Absent ⇒ grey placeholder")
    p.add_argument("--watermark",       default=None,
                   help="text shown bottom-right (default: empty)")
    p.add_argument("--allow-converted", action="store_true",
                   help="render standard/taiko/ctb beatmaps by converting "
                        "to mania (rough reproduction of in-game converter)")
    p.add_argument("--convert-to-keys", type=int, default=4,
                   choices=[4, 5, 6, 7, 8, 9, 10],
                   help="target keycount when converting (default 4)")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    w, h = (int(x) for x in args.resolution.lower().split("x"))
    # Build RenderOptions kwargs, only overriding fields where a flag was
    # given. RenderOptions defaults stay authoritative.
    opts_kwargs: dict = {
        "resolution":      (w, h),
        "fps":             args.fps,
        "encoder":         args.encoder,
        "encoder_device":  args.encoder_device,
        "timeout_seconds": args.timeout,
    }
    if args.bg_dim is not None:
        opts_kwargs["background_dim"] = max(0.0, min(1.0, args.bg_dim))
    if args.bg_dim_intro is not None:
        opts_kwargs["bg_dim_intro"] = max(0, min(100, args.bg_dim_intro)) / 100.0
    if args.bg_dim_game is not None:
        opts_kwargs["bg_dim_game"]  = max(0, min(100, args.bg_dim_game)) / 100.0
    if args.bg_dim_breaks is not None:
        opts_kwargs["bg_dim_breaks"] = max(0, min(100, args.bg_dim_breaks)) / 100.0
    if args.bg_blur is not None:
        opts_kwargs["bg_blur"] = max(0, min(10, args.bg_blur))
    if args.scroll_speed is not None:
        opts_kwargs["scroll_speed"] = max(1, min(40, args.scroll_speed))
    if args.music_volume is not None:
        opts_kwargs["music_volume"] = max(0, min(100, args.music_volume)) / 100.0
    if args.hitsound_volume is not None:
        opts_kwargs["hitsound_volume"] = max(0, min(100, args.hitsound_volume)) / 100.0
    if args.combo_break_threshold is not None:
        opts_kwargs["combo_break_threshold"] = max(0, args.combo_break_threshold)
    if args.audio_fade_out_ms is not None:
        opts_kwargs["audio_fade_out_ms"] = max(0, args.audio_fade_out_ms)
    if args.no_hp_bar:      opts_kwargs["show_hp_bar"]         = False
    if args.no_hit_error:   opts_kwargs["show_hit_error_popup"] = False
    if args.no_ur:          opts_kwargs["show_ur_bar"]         = False
    if args.no_progress:    opts_kwargs["show_progress_bar"]   = False
    if args.no_combo_pop:   opts_kwargs["show_combo_pop"]      = False
    if args.no_miss_shake:  opts_kwargs["show_miss_shake"]     = False
    if args.no_loudnorm:    opts_kwargs["normalize_loudness"]  = False
    if args.no_combo_break: opts_kwargs["combo_break_sound"]   = False
    if args.no_score:           opts_kwargs["show_score"]        = False
    if args.no_grade:           opts_kwargs["show_grade"]        = False
    if args.no_key_overlay:     opts_kwargs["show_key_overlay"]  = False
    if args.no_key_counter:     opts_kwargs["show_key_counter"]  = False
    if args.no_result_screen:   opts_kwargs["show_result_screen"]= False
    if args.show_pp:            opts_kwargs["show_pp_counter"]   = True
    if args.pp is not None:     opts_kwargs["pp_override"]       = args.pp
    if args.hide_judgement_line:opts_kwargs["hide_judgement_line"] = True
    if args.no_skip_intro:      opts_kwargs["skip_intro"]        = False
    if args.no_replay_hitsounds: opts_kwargs["use_replay_hitsounds"] = False
    if args.nightcore_hitsounds: opts_kwargs["nightcore_hitsounds"]  = True
    if args.skin_hitsounds:      opts_kwargs["use_skin_hitsounds"]   = True
    if args.no_beatmap_hitsounds: opts_kwargs["beatmap_hitsounds"]  = False
    if args.no_miss_hitsound:     opts_kwargs["miss_hitsound"]      = False
    if args.skin_combo_colors:   opts_kwargs["skin_combo_colors"]    = args.skin_combo_colors
    if args.logo:                opts_kwargs["show_logo"]            = True
    if args.featured_avatar_png is not None:
        opts_kwargs["featured_avatar_png"] = str(args.featured_avatar_png)
    if args.watermark is not None:
        opts_kwargs["watermark_text"] = args.watermark[:64]
    options = RenderOptions(**opts_kwargs)

    async def _run() -> None:
        await render_mania(
            osr_path=args.osr,
            beatmap_dir=args.beatmap_dir,
            output_path=args.output,
            options=options,
            progress_callback=_print_progress,
            skin_dir=args.skin_dir,
            allow_converted=args.allow_converted,
            convert_to_keys=args.convert_to_keys,
        )

    try:
        asyncio.run(_run())
    except Exception as e:  # noqa: BLE001
        # Full traceback to stderr so the bot's log captures it. The bot's
        # error message to the user stays terse ("Render failed: see log")
        # but we get actionable detail in the log when something blows up.
        import traceback
        print(f"error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


async def _print_progress(fraction: float) -> None:
    print(f"\rrendering… {fraction:.0%}", end="", flush=True)


if __name__ == "__main__":
    sys.exit(main())
