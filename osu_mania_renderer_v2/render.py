"""The public render_mania orchestrator: parse → mod → render → encode.

The per-render setup (parse/mods/judgments/timelines/encoder/ffmpeg) and the
per-frame gameplay-state computation are extracted into `build_render_plan`
and `build_frame_state` so an alternative draw path (the wiki-driven renderer)
can reuse identical gameplay semantics — the only thing that differs between
paths is HOW each frame is drawn.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import time as _t
from bisect import bisect_right as _br
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from osu_mania_renderer_v2.beatmap import build_sv_distance_table, parse_beatmap
from osu_mania_renderer_v2.encode import FfmpegPipe, build_ffmpeg_cmd, probe_encoder
from osu_mania_renderer_v2.errors import (
    BeatmapParseError,
    MissingAudioError,
    RenderTimeoutError,
)
from osu_mania_renderer_v2.gpu.context import HeadlessGl
from osu_mania_renderer_v2.gpu.readback import FrameReader
from osu_mania_renderer_v2.gpu.renderer import FrameRenderer, RenderContext
from osu_mania_renderer_v2.judgments import compute_judgments, reconcile_to_counts
from osu_mania_renderer_v2.models import HoldNote, RenderOptions
from osu_mania_renderer_v2.mods import apply_mods, mod_acronyms
from osu_mania_renderer_v2.hitsounds import build_hitsound_track
from osu_mania_renderer_v2.pp import compute_pp
from osu_mania_renderer_v2.replay import parse_replay
from osu_mania_renderer_v2.scene import JudgmentPopup, snapshot

log = logging.getLogger("osu_mania_renderer_v2")

APPROACH_MS = 600
# Scroll-speed scale matches osu!mania's integer 1–40 (lazer default 17,
# higher = faster). The baseline picks the value that reproduces the
# pre-settings behaviour (APPROACH_MS = 600 ms) — anything higher shrinks
# the approach window proportionally, anything lower widens it.
SCROLL_SPEED_BASELINE = 17
RESULTS_DURATION_MS = 6000   # how long the post-game results card shows
RESULTS_GAP_MS = 800         # quiet gap between last note and results fade-in
START_FADE_MS = 1600         # opening fade-in from black at song start
END_FADE_MS = 600            # gameplay → results transition fade
HIT_LIGHT_DURATION_MS = 320  # how long a receptor flash lingers
COMBO_POP_DURATION_MS = 180  # how long the combo number stays scaled up

# osu!mania HP deltas per judgment. The osu! stable HP drain formula is
# complex; these values approximate the visible behaviour — geki/300 fully
# heal, misses drain hard, 100/50 light drain.
_HP_DELTA: dict[str, float] = {
    "geki": +0.04,
    "300":  +0.025,
    "katu": +0.005,
    "100":  -0.02,
    "50":   -0.04,
    "miss": -0.08,
}

# osu!mania v1 base score weights — used to interpolate the on-screen score
# proportionally to hit quality (not just hit count) so each judgment moves
# the counter by a believable amount.
_HIT_SCORE_WEIGHT: dict[str, int] = {
    "geki": 320, "300": 300, "katu": 200,
    "100": 100, "50": 50, "miss": 0,
}

# Default skin location on the host (Night05 lives here, including a
# combobreak.wav). The renderer treats it as a fallback for samples
# missing in the beatmap dir.
_DEFAULT_SKIN_DIRS = (
    Path("/var/mnt/Synology-Reddie/Mania ORDR Bot/skins/_default/_default-source"),
    Path("/var/mnt/Synology-Reddie/Mania ORDR Bot/skins/_default-source"),
    Path("/var/mnt/ASUStor-Samsung/R3DManiaORDRBot/skins/_default/_default-source"),
    Path("/var/mnt/ASUStor-Samsung/R3DManiaORDRBot/skins/_default"),
)


@dataclass
class RenderPlan:
    """Everything computed once per render, before the per-frame loop.

    Shared by `render_mania` (legacy draw) and the wiki-driven renderer so
    both compute identical gameplay numbers. Holds pure data only — no GL
    context, ffmpeg pipe, or other live resource (those are owned by the
    loop function so their lifecycle stays in one try/finally)."""

    # inputs / context
    options: RenderOptions
    skin_dir: Path | None
    beatmap_dir: Path
    output_path: Path
    replay: Any
    modded: Any
    key_count: int
    # gameplay-state inputs (consumed by build_frame_state)
    effective_approach_ms: int
    visual_mods: Any
    judged_hits: dict[tuple[int, int], int]
    sv_for_note: dict[tuple[int, int], float]
    timing_points: tuple
    sv_table: tuple[float, ...]
    note_times: tuple[int, ...]
    max_hold_dur_ms: int
    judgment_events: tuple
    judgment_timeline: list
    total_quality: int
    kiai_ranges: list[tuple[int, int]]
    per_column_ur: tuple[float, ...]
    miss_break_times: list[int]
    press_iters: list[list[int]]
    acronyms: tuple[str, ...]
    player_pp: float
    max_pp: float
    gameplay_end_ms: int
    results_start_ms: int
    total_video_ms: int
    total_frames: int
    # loop-setup data
    encoder: Any
    encoder_device: str | None
    audio_path: Path | None
    audio_rate: float
    hitsound_wav: Path | None
    effective_lead_in_ms: int
    ffmpeg_cmd: Any
    fifo_path: Path | None
    bg_path: Path | None
    first_note_ms: int
    banner_text: str


async def build_render_plan(
    *,
    osr_path: Path,
    beatmap_dir: Path,
    output_path: Path,
    options: RenderOptions,
    skin_dir: Path | None = None,
    allow_converted: bool = False,
    convert_to_keys: int = 4,
) -> RenderPlan:
    """Parse + mod + judge + build all per-render data and the ffmpeg command.
    Pure of live resources (no GL/pipe); side effects limited to probing the
    encoder and building the optional hitsound WAV."""
    replay = parse_replay(osr_path)
    osu_file = _find_osu(beatmap_dir, replay.beatmap_md5)
    beatmap = parse_beatmap(
        osu_file,
        allow_converted=allow_converted,
        convert_to_keys=convert_to_keys,
        # The converter uses the player's key-press events to recover the
        # original ManiaBeatmapConverter's column assignments — see
        # converter.py for the matching algorithm.
        replay_key_events=replay.key_events,
    )
    # Scroll-speed override → on-screen approach window. None keeps baseline.
    if options.scroll_speed:
        effective_approach_ms = int(APPROACH_MS * SCROLL_SPEED_BASELINE / options.scroll_speed)
    else:
        effective_approach_ms = APPROACH_MS
    mod_res = apply_mods(beatmap, replay)
    for w in mod_res.warnings:
        log.warning(w)
    modded = mod_res.beatmap

    judgments = compute_judgments(
        modded.notes, replay.key_events, modded.key_count,
        overall_difficulty=getattr(modded, "overall_difficulty", None),
    )
    # Re-label the simulated judgments to the .osr's authoritative tallies so the
    # live accuracy/counts are correct frame-by-frame and need no end-of-song
    # patch (see reconcile_to_counts).
    judgments = reconcile_to_counts(
        judgments, replay.count_geki, replay.count_300, replay.count_katu,
        replay.count_100, replay.count_50, replay.count_miss,
    )
    total_quality = max(
        1, sum(_HIT_SCORE_WEIGHT[j.judgment] for j in judgments.events),
    )
    # Per-note "consumed" timestamps: notes the player actually tried to hit.
    # `judged_hits` keyed on (column, scheduled note time) → press timestamp.
    # Only NON-miss notes appear here, so true misses fall through to the
    # "slip past" code path in `scene.snapshot` and stay visible.
    judged_hits: dict[tuple[int, int], int] = {}
    for j in judgments.events:
        if j.judgment != "miss" and j.hit_offset_ms is not None:
            press_t = int(j.time_ms + j.hit_offset_ms)
            judged_hits[(j.column, j.time_ms)] = press_t

    # Pre-compute kiai-time ranges from the beatmap (bit-0 flag in a timing
    # point's `effects`), as (start_ms, end_ms) windows in chronological order.
    kiai_ranges: list[tuple[int, int]] = _extract_kiai_ranges(osu_file)

    # Per-column UR — computed ONCE at end-of-game from the final judgment
    # timeline so the results card can show it.
    per_column_ur = _per_column_ur(judgments.events, modded.key_count)

    # SV positioning via cumulative-distance integration. `sv_for_note` kept
    # as an empty dict so legacy code paths don't NoneType-crash.
    tps = modded.timing_points
    sv_table = build_sv_distance_table(tps)
    sv_for_note: dict[tuple[int, int], float] = {}
    # Mod pill labels for the HUD.
    acronyms = mod_acronyms(replay.mods, modded.key_count)

    # PP for this play + max possible PP (SS-FC). Both 0 if rosu-pp absent.
    player_pp, max_pp = compute_pp(osu_file, replay)
    log.info("pp", extra={"player": player_pp, "max": max_pp})

    # Pre-sort judgments by the PRESS time so the combo/score/accuracy
    # timeline reflects the player's actual hand, not the scheduled times.
    judgment_timeline = sorted(
        (
            (
                int(j.time_ms + (j.hit_offset_ms or 0)),  # effective time
                j,
            )
            for j in judgments.events
        ),
        key=lambda pair: pair[0],
    )
    log.info(
        "judgments_done",
        extra={
            "geki": judgments.count_geki, "300": judgments.count_300,
            "katu": judgments.count_katu, "100": judgments.count_100,
            "50": judgments.count_50, "miss": judgments.count_miss,
            "max_combo": judgments.max_combo,
        },
    )

    # Auto-pick the standard VAAPI render node.
    encoder_device = options.encoder_device
    if encoder_device is None and Path("/dev/dri/renderD128").exists():
        encoder_device = "/dev/dri/renderD128"
    encoder = await probe_encoder(options.encoder, encoder_device)
    audio_path: Path | None = None
    if modded.audio_filename:
        cand = beatmap_dir / modded.audio_filename
        if cand.exists():
            audio_path = cand
        elif options.audio_required:
            raise MissingAudioError(f"audio file not found: {cand}")
        else:
            log.warning("audio_missing", extra={"expected": str(cand)})

    # End-of-song layout: brief silent gap → results card.
    gameplay_end_ms = modded.total_duration_ms
    results_start_ms = gameplay_end_ms + RESULTS_GAP_MS
    total_video_ms = results_start_ms + RESULTS_DURATION_MS
    total_frames = math.ceil(total_video_ms / 1000 * options.fps)

    # Hitsound track — a temp WAV pre-mixed with one sample at every non-miss
    # judgment's press time. Failure → song only.
    hitsound_wav: Path | None = None
    if audio_path is not None and (options.use_replay_hitsounds or options.nightcore_hitsounds):
        skin_dirs: list[Path] = []
        if options.use_skin_hitsounds and skin_dir is not None and skin_dir.is_dir():
            skin_dirs.append(skin_dir)
        skin_dirs.extend(p for p in _DEFAULT_SKIN_DIRS if p.is_dir())
        try:
            hitsound_wav = build_hitsound_track(
                judgments_events=judgments.events if options.use_replay_hitsounds else (),
                beatmap=modded,
                beatmap_dir=beatmap_dir,
                output_wav=output_path.with_suffix(".hits.wav"),
                duration_ms=total_video_ms,
                audio_rate=mod_res.audio_rate,
                skin_dirs=tuple(skin_dirs),
                nightcore=options.nightcore_hitsounds,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("hitsound_build_failed", extra={"err": str(e)})
            hitsound_wav = None

    # Host-ffmpeg FIFO path (toolbox). Otherwise plain stdin.
    fifo_path: Path | None = None
    if Path("/run/host/etc/os-release").exists() and Path("/usr/bin/flatpak-spawn").exists():
        fifo_path = Path(f"/tmp/mania-frames-{os.getpid()}.fifo")
        if fifo_path.exists():
            try:
                fifo_path.unlink()
            except OSError:
                pass

    # skip_intro=True collapses the silent lead-in to zero.
    effective_lead_in_ms = 0 if options.skip_intro else modded.audio_lead_in_ms

    cmd = build_ffmpeg_cmd(
        encoder=encoder,
        encoder_device=encoder_device,
        resolution=options.resolution,
        fps=options.fps,
        audio_path=audio_path,
        audio_rate=mod_res.audio_rate,
        audio_lead_in_ms=effective_lead_in_ms,
        video_bitrate=options.video_bitrate,
        audio_bitrate=options.audio_bitrate,
        output_path=output_path,
        total_duration_ms=total_video_ms,
        hitsound_path=hitsound_wav,
        frames_fifo_path=fifo_path,
        music_volume=options.music_volume,
        hitsound_volume=options.hitsound_volume,
    )

    bg_filename = modded.background_filename
    bg_path = (beatmap_dir / bg_filename) if bg_filename else None
    first_note_ms = min((n.time_ms for n in modded.notes), default=0)
    banner_text = (
        f"{modded.artist} - {modded.title} [{modded.difficulty}]   "
        f"{replay.player_name}"
    )

    # scene.snapshot perf hints (precomputed once).
    note_times_tuple = tuple(n.time_ms for n in modded.notes)
    max_hold_dur_ms_val = 0
    for _n in modded.notes:
        if isinstance(_n, HoldNote):
            _dur = _n.end_time_ms - _n.time_ms
            if _dur > max_hold_dur_ms_val:
                max_hold_dur_ms_val = _dur

    miss_break_times = _compute_miss_break_times(judgments.events, threshold=20)
    press_iters = _rising_edges_per_col(replay.key_events, modded.key_count)

    return RenderPlan(
        options=options, skin_dir=skin_dir, beatmap_dir=beatmap_dir,
        output_path=output_path, replay=replay, modded=modded,
        key_count=modded.key_count,
        effective_approach_ms=effective_approach_ms,
        visual_mods=mod_res.visual_mods, judged_hits=judged_hits,
        sv_for_note=sv_for_note, timing_points=tps, sv_table=sv_table,
        note_times=note_times_tuple, max_hold_dur_ms=max_hold_dur_ms_val,
        judgment_events=judgments.events, judgment_timeline=judgment_timeline,
        total_quality=total_quality, kiai_ranges=kiai_ranges,
        per_column_ur=per_column_ur, miss_break_times=miss_break_times,
        press_iters=press_iters, acronyms=acronyms,
        player_pp=player_pp, max_pp=max_pp,
        gameplay_end_ms=gameplay_end_ms, results_start_ms=results_start_ms,
        total_video_ms=total_video_ms, total_frames=total_frames,
        encoder=encoder, encoder_device=encoder_device,
        audio_path=audio_path, audio_rate=mod_res.audio_rate,
        hitsound_wav=hitsound_wav, effective_lead_in_ms=effective_lead_in_ms,
        ffmpeg_cmd=cmd, fifo_path=fifo_path, bg_path=bg_path,
        first_note_ms=first_note_ms, banner_text=banner_text,
    )


def build_frame_state(
    plan: RenderPlan,
    t_ms: int,
    score_smoothed: float,
    accuracy_smoothed: float,
):
    """Compute the full per-frame SceneState. Pure given (plan, t_ms,
    smoothing carriers). Returns (scene_full, score_smoothed, accuracy_smoothed)
    so the caller threads the two single-pole low-pass accumulators forward."""
    replay = plan.replay
    key_count = plan.key_count
    gameplay_end_ms = plan.gameplay_end_ms
    results_start_ms = plan.results_start_ms

    scene = snapshot(
        notes=plan.modded.notes,
        key_events=replay.key_events,
        t_ms=t_ms,
        key_count=key_count,
        approach_ms=plan.effective_approach_ms,
        visual_mods=plan.visual_mods,
        consumed_times=plan.judged_hits,
        sv_for_note=plan.sv_for_note,
        timing_points=plan.timing_points,
        sv_table=plan.sv_table,
        note_times=plan.note_times,
        max_hold_dur_ms=plan.max_hold_dur_ms,
    )
    # Active judgments: any whose time ∈ [t-600ms, t].
    active = tuple(
        JudgmentPopup(
            column=j.column, judgment=j.judgment,
            age_ms=t_ms - j.time_ms,
        )
        for j in plan.judgment_events
        if 0 <= t_ms - j.time_ms < 600
    )
    # One pass over the press-time-sorted timeline: counts → live accuracy +
    # quality-weighted score + running combo + UR/avg offset + HP +
    # last-hit-per-column + offset-per-column.
    running = {"geki": 0, "300": 0, "katu": 0, "100": 0, "50": 0, "miss": 0}
    quality_so_far = 0
    offsets_so_far: list[float] = []
    combo_at_t = 0
    last_combo_change_t = 0
    hp = 1.0
    last_hit_per_col: list[tuple[int, str, float]] = [
        (-99999, "", 0.0) for _ in range(key_count)
    ]
    for eff_t, j in plan.judgment_timeline:
        if eff_t > t_ms:
            break
        running[j.judgment] += 1
        quality_so_far += _HIT_SCORE_WEIGHT[j.judgment]
        if j.judgment == "miss":
            combo_at_t = 0
        else:
            combo_at_t += 1
        last_combo_change_t = eff_t
        hp = max(0.0, min(1.0, hp + _HP_DELTA.get(j.judgment, 0)))
        if j.judgment != "miss" and 0 <= j.column < key_count:
            last_hit_per_col[j.column] = (
                eff_t, j.judgment, j.hit_offset_ms or 0.0,
            )
        if j.hit_offset_ms is not None:
            offsets_so_far.append(j.hit_offset_ms)
    score_so_far = int(replay.score * quality_so_far / plan.total_quality)
    pp_live = (plan.player_pp * quality_so_far / plan.total_quality
               if plan.total_quality > 0 else 0.0)
    if offsets_so_far:
        avg_offset = sum(offsets_so_far) / len(offsets_so_far)
        if len(offsets_so_far) >= 2:
            var = sum((x - avg_offset) ** 2 for x in offsets_so_far) / len(offsets_so_far)
            ur = 10.0 * (var ** 0.5)
        else:
            ur = 0.0
    else:
        avg_offset = 0.0
        ur = 0.0
    recent_offsets = tuple(offsets_so_far[-60:])  # last 60 ticks

    hit_light_age = tuple(
        (t_ms - lhc[0]) if lhc[1] else 99999
        for lhc in last_hit_per_col
    )
    hit_light_jud = tuple(lhc[1] for lhc in last_hit_per_col)
    hit_offset_per_col = tuple(lhc[2] for lhc in last_hit_per_col)

    combo_age_ms = t_ms - last_combo_change_t

    if gameplay_end_ms > 0:
        song_progress = min(1.0, max(0.0, t_ms / gameplay_end_ms))
    else:
        song_progress = 0.0

    is_kiai = any(s <= t_ms <= e for s, e in plan.kiai_ranges)

    key_press_age_ms_arr = []
    key_press_counts_arr = []
    for c in range(key_count):
        times = plan.press_iters[c]
        n_pressed = _br(times, t_ms)          # rising edges up to now
        key_press_counts_arr.append(n_pressed)
        idx = n_pressed - 1
        if idx >= 0:
            key_press_age_ms_arr.append(t_ms - times[idx])
        else:
            key_press_age_ms_arr.append(99999)
    key_press_age_ms = tuple(key_press_age_ms_arr)
    key_press_counts = tuple(key_press_counts_arr)

    miss_break_age = 99999
    for mt in reversed(plan.miss_break_times):
        if mt <= t_ms:
            miss_break_age = t_ms - mt
            break

    total_so_far = sum(running.values())
    if total_so_far == 0:
        acc_so_far = 100.0
    else:
        # osu!mania accuracy as the website / lazer UI shows it.
        weighted = (
            50 * running["50"] + 100 * running["100"]
            + 200 * running["katu"] + 300 * running["300"]
            + 305 * running["geki"]
        )
        acc_so_far = (weighted / (305 * total_so_far)) * 100

    # End-of-song blend toward the .osr-recorded authoritative values.
    _ENDGAME_BLEND_MS = 500
    _endgame_blend_t = max(0.0, min(1.0, (
        t_ms - (gameplay_end_ms - _ENDGAME_BLEND_MS)
    ) / _ENDGAME_BLEND_MS)) if gameplay_end_ms > 0 else 0.0
    if _endgame_blend_t > 0.0:
        b = _endgame_blend_t
        score_so_far = int(
            score_so_far + (replay.score - score_so_far) * b
        )
        # accuracy is NOT blended any more: judgments are reconciled to the .osr
        # tallies up front, so acc_so_far already lands exactly on replay.accuracy
        # — blending it caused the visible end-of-song "patch".
        pp_live = pp_live + (plan.player_pp - pp_live) * b

    # Score / accuracy tween — single-pole low-pass per frame.
    tween_alpha = 0.25
    score_smoothed += (score_so_far - score_smoothed) * tween_alpha
    accuracy_smoothed += (acc_so_far - accuracy_smoothed) * tween_alpha
    # Results overlay fades in over ~400ms once the gap ends.
    if t_ms >= results_start_ms:
        results_opacity = min(1.0, (t_ms - results_start_ms) / 400.0)
    else:
        results_opacity = 0.0

    # Full-screen black fade.
    if t_ms < START_FADE_MS:
        fade = 1.0 - (t_ms / START_FADE_MS)
    elif t_ms >= gameplay_end_ms and t_ms < results_start_ms:
        fade_start = results_start_ms - END_FADE_MS
        if t_ms >= fade_start:
            fade = (t_ms - fade_start) / END_FADE_MS
        else:
            fade = 0.0
    else:
        fade = 0.0
    fade = max(0.0, min(1.0, fade))
    scene_full = scene.__class__(
        t_ms=scene.t_ms, visible_notes=scene.visible_notes,
        keys_held=scene.keys_held, visual_mods=scene.visual_mods,
        active_judgments=active,
        score=score_so_far, combo=combo_at_t,
        max_combo=replay.max_combo,
        accuracy=(replay.accuracy if results_opacity > 0 else acc_so_far),
        mod_acronyms=plan.acronyms,
        results_opacity=results_opacity,
        grade=_compute_grade_from_replay(replay),
        judgment_counts=(
            replay.count_geki, replay.count_300,
            replay.count_katu, replay.count_100,
            replay.count_50, replay.count_miss,
        ),
        recent_offsets=recent_offsets,
        avg_hit_offset_ms=avg_offset,
        unstable_rate=ur,
        pp=(plan.player_pp if results_opacity > 0 else pp_live),
        max_pp=plan.max_pp,
        fade_to_black=fade,
        hit_light_age_ms=hit_light_age,
        hit_light_judgment=hit_light_jud,
        hit_offset_per_col=hit_offset_per_col,
        combo_age_ms=combo_age_ms,
        score_smoothed=int(score_smoothed),
        accuracy_smoothed=accuracy_smoothed,
        song_progress=song_progress,
        hp=hp,
        is_kiai=is_kiai,
        per_column_ur=plan.per_column_ur,
        key_press_age_ms=key_press_age_ms,
        key_press_counts=key_press_counts,
        miss_break_age_ms=miss_break_age,
    )
    return scene_full, score_smoothed, accuracy_smoothed


async def render_mania(
    *,
    osr_path: Path,
    beatmap_dir: Path,
    output_path: Path,
    options: RenderOptions,
    progress_callback: Callable[[float], Awaitable[None]] | None = None,
    log_path: Path | None = None,
    skin_dir: Path | None = None,
    allow_converted: bool = False,
    convert_to_keys: int = 4,
) -> None:
    log.info("render_start", extra={"osr": str(osr_path), "out": str(output_path)})

    plan = await build_render_plan(
        osr_path=osr_path, beatmap_dir=beatmap_dir, output_path=output_path,
        options=options, skin_dir=skin_dir, allow_converted=allow_converted,
        convert_to_keys=convert_to_keys,
    )

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
                rc, options, skin_dir=skin_dir,
                beatmap_dir=beatmap_dir,
                first_note_ms=plan.first_note_ms,
            )
            if plan.bg_path and plan.bg_path.exists():
                fr.set_background(plan.bg_path)
            fr.set_banner_text(plan.banner_text)
            reader = FrameReader(gl.ctx, gl.fbo, components=3)

            last_progress_t = 0.0
            score_smoothed = 0.0
            accuracy_smoothed = 100.0
            # Per-phase accumulators (pre-draw Python / GPU draw / readback).
            _phase_pre_draw_s = 0.0
            _phase_draw_s = 0.0
            _phase_read_write_s = 0.0
            for frame_n in range(plan.total_frames):
                _phase_t0 = _t.perf_counter()
                if loop.time() > deadline:
                    raise RenderTimeoutError(
                        f"render exceeded {options.timeout_seconds}s"
                    )
                t_ms = int(frame_n * 1000 / options.fps)
                scene_full, score_smoothed, accuracy_smoothed = build_frame_state(
                    plan, t_ms, score_smoothed, accuracy_smoothed,
                )
                _phase_pre_draw_s += _t.perf_counter() - _phase_t0
                _phase_t1 = _t.perf_counter()
                fr.draw(scene_full)
                _phase_draw_s += _t.perf_counter() - _phase_t1
                _phase_t2 = _t.perf_counter()
                frame = reader.read()
                await pipe.write_frame(frame)
                _phase_read_write_s += _t.perf_counter() - _phase_t2

                if progress_callback and (loop.time() - last_progress_t > 0.5):
                    await progress_callback(frame_n / plan.total_frames)
                    last_progress_t = loop.time()

            # Flush any frames still in flight in the readback PBO ring.
            for tail_frame in reader.drain():
                await pipe.write_frame(tail_frame)
            log.info(
                "phase_timing pre_draw=%.2fs draw=%.2fs read_write=%.2fs frames=%d",
                _phase_pre_draw_s, _phase_draw_s, _phase_read_write_s,
                plan.total_frames,
            )

            if progress_callback:
                await progress_callback(1.0)
        await pipe.close(output_path)
    except BaseException:
        # Kill ffmpeg on any error
        if pipe.proc and pipe.proc.returncode is None:
            try:
                pipe.proc.kill()
            except ProcessLookupError:
                pass
        raise
    finally:
        # Drop the temp hitsound WAV regardless of success / failure.
        if plan.hitsound_wav is not None:
            try:
                plan.hitsound_wav.unlink(missing_ok=True)
            except OSError:
                pass

    log.info("render_done", extra={"out": str(output_path)})


def _find_osu(beatmap_dir: Path, expected_md5: str) -> Path:
    try:
        entries = list(beatmap_dir.iterdir())
    except FileNotFoundError as e:
        raise BeatmapParseError(f"beatmap_dir does not exist: {beatmap_dir}") from e
    for f in entries:
        if f.suffix.lower() != ".osu":
            continue
        h = hashlib.md5(f.read_bytes()).hexdigest()
        if h == expected_md5:
            return f
    # Fall back to first .osu file.
    for f in entries:
        if f.suffix.lower() == ".osu":
            return f
    raise BeatmapParseError(f"no .osu file in {beatmap_dir}")


def _extract_kiai_ranges(osu_path: Path) -> list[tuple[int, int]]:
    """Walk the [TimingPoints] section once and produce (start_ms, end_ms)
    windows for every kiai burst. Kiai is encoded as the bit-0 flag in
    `effects` on each timing point; a kiai region runs from a TP with bit
    0 set until the next TP that clears it (or end of map).
    """
    try:
        text = osu_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    out: list[tuple[int, int]] = []
    in_tp = False
    open_start: int | None = None
    for raw in text.splitlines():
        line = raw.rstrip("\r")
        if line.startswith("["):
            in_tp = (line == "[TimingPoints]")
            continue
        if not in_tp or not line or line.startswith("//"):
            continue
        parts = line.split(",")
        if len(parts) < 8:
            continue
        try:
            t = int(float(parts[0]))
            effects = int(parts[7])
        except ValueError:
            continue
        kiai = bool(effects & 1)
        if kiai and open_start is None:
            open_start = t
        elif not kiai and open_start is not None:
            out.append((open_start, t))
            open_start = None
    if open_start is not None:
        out.append((open_start, 10**9))  # open-ended kiai → far future
    return out


def _compute_miss_break_times(events, threshold: int = 20) -> list[int]:
    """Walk the (note-time-sorted) judgment timeline and emit the time_ms
    of every miss that broke a combo ≥ ``threshold``. The renderer uses
    these to trigger the playfield-shake/flash on big combo breaks."""
    out: list[int] = []
    combo = 0
    for j in events:
        if j.judgment == "miss":
            if combo >= threshold:
                out.append(j.time_ms)
            combo = 0
        else:
            combo += 1
    return out


def _rising_edges_per_col(events, key_count: int) -> list[list[int]]:
    """Same as the rising-edge helper in judgments.py, but exposed at the
    render level so we can walk it per-frame with bisect."""
    presses: list[list[int]] = [[] for _ in range(key_count)]
    prev = 0
    for e in events:
        new_pressed = e.keys_held & ~prev
        for c in range(key_count):
            if new_pressed & (1 << c):
                presses[c].append(e.time_ms)
        prev = e.keys_held
    return presses


def _per_column_ur(events, key_count: int) -> tuple[float, ...]:
    """UR computed per-column from the final judgment timeline. Same
    formula as the global UR (10 × stddev of signed offsets) but bucketed
    by column. Returns 0 for columns with no recorded hits."""
    per_col: list[list[float]] = [[] for _ in range(key_count)]
    for j in events:
        if (j.hit_offset_ms is not None and 0 <= j.column < key_count
                and j.judgment != "miss"):
            per_col[j.column].append(j.hit_offset_ms)
    out: list[float] = []
    for offsets in per_col:
        if len(offsets) < 2:
            out.append(0.0)
            continue
        avg = sum(offsets) / len(offsets)
        var = sum((x - avg) ** 2 for x in offsets) / len(offsets)
        out.append(10.0 * (var ** 0.5))
    return tuple(out)


def _compute_grade_from_replay(replay) -> str:
    """osu!mania grade boundaries — purely accuracy-based, no "no misses"
    requirement (osu! wiki: Game Mode/osu!mania#Grades). 96.84% with 2
    misses is still S; only an all-320/300 play gets SS."""
    if (replay.count_300 == 0 and replay.count_katu == 0
            and replay.count_100 == 0 and replay.count_50 == 0
            and replay.count_miss == 0):
        return "SS"
    if replay.accuracy >= 95.0:
        return "S"
    if replay.accuracy >= 90.0:
        return "A"
    if replay.accuracy >= 80.0:
        return "B"
    if replay.accuracy >= 70.0:
        return "C"
    return "D"


def _aggregate_score(events: tuple, t_ms: int) -> tuple[int, int]:
    """Cumulative score + current combo at time t."""
    weights = {"geki": 320, "300": 300, "katu": 200, "100": 100, "50": 50, "miss": 0}
    score = 0
    combo = 0
    for j in events:
        if j.time_ms > t_ms:
            break
        score += weights[j.judgment] * (1 + combo // 10)
        if j.judgment == "miss":
            combo = 0
        else:
            combo += 1
    return score, combo
