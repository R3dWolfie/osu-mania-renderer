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
from dataclasses import dataclass, field, replace as _dc_replace
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
from osu_mania_renderer_v2.models import HoldNote, KeyEvent, RenderOptions
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

# ScoreV3 (standardised) mod-score multipliers (ppy/osu#37967 rework). NC is
# stored as DT|NC, so drop the implied DT bit or 1.23× squares to 1.51×.
_MOD_SCORE_MULT = {1 << 1: 0.80, 1 << 3: 1.04, 1 << 4: 1.09, 1 << 6: 1.23,
                   1 << 8: 0.55, 1 << 9: 1.23, 1 << 10: 1.20, 1 << 12: 0.95}


def mods_score_multiplier(mods: int) -> float:
    mods = int(mods or 0)
    if mods & (1 << 9):        # NC set → clear implied DT (count speed once)
        mods &= ~(1 << 6)
    m = 1.0
    for bit, mult in _MOD_SCORE_MULT.items():
        if mods & bit:
            m *= mult
    return m


# ScoreV3 accuracy base per mania judgment; geki (MAX) uses the replay's max
# weight (305 lazer / 300 stable), applied at use.
_SD_ACC_WEIGHT: dict[str, int] = {"300": 300, "katu": 200, "100": 100,
                                  "50": 50, "miss": 0}

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
    # ScoreV3 (standardised) precomputed constants — defaults keep any other
    # RenderPlan construction path valid.
    n_scoring: int = 0
    max_combo_portion: float = 0.0
    mod_mult: float = 1.0
    mania_mw: int = 305


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

    # Rate mods (DT/NC/HT): the .osr keypress timeline is in MAP time, but
    # apply_mods rescaled every note to REAL (video) time. Rescale the replay
    # events identically, or the judgment sim pairs presses against notes up
    # to 33% (DT) / 25% (HT) of the song-position away and reads a clean play
    # as a fail (Tono HT 92.87% rendered as 44% FAILED). Hit windows stay
    # unscaled: stable-mania windows are rate-independent in real time.
    # NOTE: parse_beatmap above must keep consuming the RAW events - the
    # std->mania convert recovery matches them against raw .osu times.
    if mod_res.audio_rate != 1.0:
        replay = _dc_replace(replay, key_events=tuple(
            KeyEvent(time_ms=int(e.time_ms / mod_res.audio_rate),
                     keys_held=e.keys_held)
            for e in replay.key_events))

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
    # scoring-only: ScoreV1 hold tails are visual events (scoring=False) and
    # must not dilute the score/accuracy pacing — see reconcile_to_counts.
    total_quality = max(
        1, sum(_HIT_SCORE_WEIGHT[j.judgment]
               for j in judgments.events if j.scoring),
    )
    # ScoreV3 (standardised) constants: max_combo_portion is the full-combo
    # denominator Σ mw·√k over the scoring judgments; mw (geki accuracy weight)
    # cancels in the combo ratio but is kept explicit. mod_mult per ppy/osu#37967.
    _mania_mw = getattr(replay, "mania_max_weight", 305)
    _n_scoring = sum(1 for j in judgments.events if j.scoring)
    _max_combo_portion = sum(_mania_mw * ((k + 1) ** 0.5)
                             for k in range(_n_scoring))
    _mod_mult = mods_score_multiplier(int(getattr(replay, "mods", 0) or 0))
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
        n_scoring=_n_scoring, max_combo_portion=_max_combo_portion,
        mod_mult=_mod_mult, mania_mw=_mania_mw,
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
    # `judgment_events` is note-time-sorted (compute_judgments folds a
    # sorted scoring list), so the window `0 <= t_ms - time < 800` is a
    # contiguous slice — find it with two bisects instead of scanning the
    # whole event list every frame. Sortedness is verified once per render;
    # an unsorted timeline (never in practice) falls back to the original
    # full scan so behaviour is identical either way.
    _je = plan.judgment_events
    _je_times = getattr(plan, "_je_times", None)
    if _je_times is None:
        _je_times = [j.time_ms for j in _je]
        plan._je_times = _je_times
        plan._je_sorted = all(
            _je_times[i] <= _je_times[i + 1]
            for i in range(len(_je_times) - 1)
        )
    if plan._je_sorted:
        # 0 <= t_ms - time < 800  ⇔  t_ms - 800 < time <= t_ms
        active = tuple(
            JudgmentPopup(
                column=j.column, judgment=j.judgment,
                age_ms=t_ms - j.time_ms,
            )
            for j in _je[_br(_je_times, t_ms - 800):_br(_je_times, t_ms)]
        )
    else:
        active = tuple(
            JudgmentPopup(
                column=j.column, judgment=j.judgment,
                age_ms=t_ms - j.time_ms,
            )
            for j in _je
            if 0 <= t_ms - j.time_ms < 800
        )
    # One pass over the press-time-sorted timeline: counts → live accuracy +
    # quality-weighted score + running combo + UR/avg offset + HP +
    # last-hit-per-column + offset-per-column.
    #
    # Incremental across frames: the render loop calls this with a
    # nondecreasing t_ms and this is a pure prefix-fold over
    # judgment_timeline, so the fold state is carried on the plan between
    # calls and only the events NEW since the previous frame are folded in.
    # The accumulation order (and therefore every float) is identical to
    # the from-scratch loop. A t_ms that goes backwards (no caller does)
    # resets the state and refolds from the start, preserving purity.
    _fsc = getattr(plan, "_fs_cache", None)
    if _fsc is None or t_ms < _fsc["last_t"]:
        _fsc = {
            "last_t": t_ms, "idx": 0,
            "running": {"geki": 0, "300": 0, "katu": 0, "100": 0,
                        "50": 0, "miss": 0},
            "quality": 0,
            "offsets": [],
            # Running left-to-right sum of `offsets` in append order — the
            # exact fold builtin sum() performs, so bit-identical to the
            # per-frame sum(offsets_so_far) it replaces.
            "offsets_sum": 0,
            "combo": 0,
            "last_combo_t": 0,
            "hp": 1.0,
            "last_hit_per_col": [
                (-99999, "", 0.0) for _ in range(key_count)
            ],
            "sd_combo": 0, "combo_portion": 0.0, "cur_base": 0.0,
            "n_scored": 0,
            # (len(offsets), avg, ur) at the last stats computation — the
            # avg/UR only change when a new offset lands, so frames without
            # a new judgment reuse the previous values unchanged.
            "stats": (-1, 0.0, 0.0),
        }
        plan._fs_cache = _fsc
    running = _fsc["running"]
    quality_so_far = _fsc["quality"]
    offsets_so_far = _fsc["offsets"]
    _offsets_sum = _fsc["offsets_sum"]
    combo_at_t = _fsc["combo"]
    last_combo_change_t = _fsc["last_combo_t"]
    hp = _fsc["hp"]
    last_hit_per_col = _fsc["last_hit_per_col"]
    # ScoreV3 (standardised) accumulators — scoring judgments only.
    _mw = plan.mania_mw
    _sd_combo = _fsc["sd_combo"]
    _combo_portion = _fsc["combo_portion"]
    _cur_base = _fsc["cur_base"]
    _n_scored = _fsc["n_scored"]
    _tl = plan.judgment_timeline
    _idx = _fsc["idx"]
    _n_tl = len(_tl)
    while _idx < _n_tl:
        eff_t, j = _tl[_idx]
        if eff_t > t_ms:
            break
        _idx += 1
        # ScoreV1 hold tails (scoring=False) are visual-only: they flash the
        # receptor / popup / combo but are NOT recorded judgments, so they
        # stay out of the counts + quality tally (else the final accuracy
        # can't land on the .osr's recorded value on hold-note maps).
        if j.scoring:
            running[j.judgment] += 1
            quality_so_far += _HIT_SCORE_WEIGHT[j.judgment]
            _n_scored += 1
            _cur_base += (_mw if j.judgment == "geki"
                          else _SD_ACC_WEIGHT.get(j.judgment, 0))
            if j.judgment == "miss":
                _sd_combo = 0
            else:
                _sd_combo += 1
                _combo_portion += _mw * (_sd_combo ** 0.5)
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
            _offsets_sum = _offsets_sum + j.hit_offset_ms
    _fsc["last_t"] = t_ms
    _fsc["idx"] = _idx
    _fsc["quality"] = quality_so_far
    _fsc["offsets_sum"] = _offsets_sum
    _fsc["combo"] = combo_at_t
    _fsc["last_combo_t"] = last_combo_change_t
    _fsc["hp"] = hp
    _fsc["sd_combo"] = _sd_combo
    _fsc["combo_portion"] = _combo_portion
    _fsc["cur_base"] = _cur_base
    _fsc["n_scored"] = _n_scored
    # ScoreV3 standardised: 500k·acc·comboProgress + 500k·acc⁵·progress ×mult.
    # acc == the reconciled accuracy the HUD shows (lands on replay.accuracy),
    # so an all-MAX play scores exactly 1,000,000 (×mod multiplier).
    if plan.max_combo_portion > 0 and _n_scored > 0:
        _acc = _cur_base / (_mw * _n_scored)
        _cprog = _combo_portion / plan.max_combo_portion
        _aprog = _n_scored / plan.n_scoring if plan.n_scoring else 1.0
        score_so_far = int(round(
            (500000.0 * _acc * _cprog
             + 500000.0 * (_acc ** 5) * _aprog) * plan.mod_mult))
    else:
        score_so_far = 0
    pp_live = (plan.player_pp * quality_so_far / plan.total_quality
               if plan.total_quality > 0 else 0.0)
    if offsets_so_far:
        _n_off, _avg_cached, _ur_cached = _fsc["stats"]
        if _n_off == len(offsets_so_far):
            # No new offset since the last computation — same list, same
            # avg/UR (they're pure functions of the list contents).
            avg_offset = _avg_cached
            ur = _ur_cached
        else:
            avg_offset = _offsets_sum / len(offsets_so_far)
            if len(offsets_so_far) >= 2:
                var = sum((x - avg_offset) ** 2 for x in offsets_so_far) / len(offsets_so_far)
                ur = 10.0 * (var ** 0.5)
            else:
                ur = 0.0
            _fsc["stats"] = (len(offsets_so_far), avg_offset, ur)
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
        # osu!mania accuracy as the website / lazer UI shows it — using the
        # SAME rainbow-300 weight the replay was parsed with (320 stable / 305
        # lazer) so the running acc, the end-screen and the website card all
        # agree. See replay.parse_replay / osr_parser.
        mw = getattr(plan.replay, "mania_max_weight", 305)
        weighted = (
            50 * running["50"] + 100 * running["100"]
            + 200 * running["katu"] + 300 * running["300"]
            + mw * running["geki"]
        )
        acc_so_far = (weighted / (mw * total_so_far)) * 100

    # End-of-song blend toward the .osr-recorded authoritative values.
    _ENDGAME_BLEND_MS = 500
    _endgame_blend_t = max(0.0, min(1.0, (
        t_ms - (gameplay_end_ms - _ENDGAME_BLEND_MS)
    ) / _ENDGAME_BLEND_MS)) if gameplay_end_ms > 0 else 0.0
    if _endgame_blend_t > 0.0:
        b = _endgame_blend_t
        # score is NOT blended to replay.score any more: it's now the absolute
        # standardised ScoreV3 curve (already lands correctly at the end), and
        # replay.score is mixed-format (stable ScoreV1 / lazer ScoreV3) — the
        # very inconsistency this rework removes. accuracy already lands on
        # replay.accuracy via reconcile. Only pp still eases to its final.
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
