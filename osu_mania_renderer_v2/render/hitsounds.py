"""Pre-mix per-note hitsounds onto a single WAV that ffmpeg layers on the
song.

Honours:
  * per-timing-point sample set (Normal / Soft / Drum) + custom index
  * per-timing-point volume
  * per-note hit-sound bits (normal / whistle / clap / finish)
  * per-note ``hitSample.filename`` override (the most common case in
    detailed mania maps — the mapper supplies their own drum/cymbal WAVs)
  * per-note volume override

Resolution order for a sample's path:
  1. ``note.hit_sample.filename`` → exact file in the beatmap directory
  2. ``{set}-hit{type}{index}.wav|.ogg`` in the beatmap directory
  3. ``{set}-hit{type}.wav|.ogg`` (index-0 fallback)
  4. Skip (silent)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("osu_mania_renderer_v2.render.hitsounds")

DEFAULT_HIT_GAIN = 0.55     # ceiling applied on top of per-note volume
COMBO_BREAK_THRESHOLD = 20  # osu! stable: combobreak.wav only plays on a
                            # combo ≥ 20 break.
COMBO_BREAK_GAIN = 0.65     # slightly louder than normal hits so it cuts
                            # through; matches the stable client.

# Maps the osu! sample-set int → directory-prefix string.
_SET_NAMES: dict[int, str] = {0: "soft", 1: "normal", 2: "soft", 3: "drum"}

# Hit-sound bit → type name. Bit 0 (normal) is implicit; layered additions
# come from the higher bits.
_ADDITIONS: tuple[tuple[int, str], ...] = (
    (2, "whistle"),
    (4, "finish"),
    (8, "clap"),
)

# Bundled osu! default hitsounds (soft/normal/drum x hitnormal/whistle/finish/
# clap) from ppy/osu-resources — the LAST fallback so maps without custom
# samples still sound, matching lazer's beatmap -> skin -> default lookup.
_DEFAULT_HITSOUND_DIR = Path(__file__).resolve().parent.parent / "assets" / "default_hitsounds"


def _active_timing_point(timing_points: tuple, time_ms: int):
    """Last timing point whose time_ms <= ``time_ms`` (or None)."""
    if not timing_points:
        return None
    lo, hi = 0, len(timing_points) - 1
    if timing_points[0].time_ms > time_ms:
        return timing_points[0]   # before the first TP, use the first
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if timing_points[mid].time_ms <= time_ms:
            lo = mid
        else:
            hi = mid - 1
    return timing_points[lo]


def _candidate_paths(
    dirs, set_name: str, type_name: str, index: int,
) -> list[Path]:
    """Sample-file candidates in fallback order, across a dir stack
    (beatmap -> skin(s) -> bundled default), each with the osu! lookup order
    {set}-hit{type}{index} -> {set}-hit{type} (matches lazer LookupNames)."""
    out: list[Path] = []
    suffixes = ("wav", "ogg")
    for d in dirs:
        if index > 0:
            for s in suffixes:
                out.append(d / f"{set_name}-hit{type_name}{index}.{s}")
        for s in suffixes:
            out.append(d / f"{set_name}-hit{type_name}.{s}")
    return out


class _SampleCache:
    """Lazy loader. Decodes each unique file once and stores the float32
    stereo array; misses return None and are remembered so we don't probe
    the filesystem repeatedly for a missing sample."""

    def __init__(self, target_rate: int) -> None:
        self.target_rate = target_rate
        self._cache: dict[str, np.ndarray | None] = {}

    def get(self, path: Path) -> np.ndarray | None:
        key = str(path)
        if key in self._cache:
            return self._cache[key]
        if not path.is_file():
            self._cache[key] = None
            return None
        try:
            import soundfile
            data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
        except Exception as e:  # noqa: BLE001
            log.warning("hitsound_load_failed",
                        extra={"path": str(path), "err": str(e)})
            self._cache[key] = None
            return None
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        if rate != self.target_rate:
            ratio = self.target_rate / rate
            new_len = int(data.shape[0] * ratio)
            idx = (np.arange(new_len) / ratio).astype(np.int64)
            idx = np.clip(idx, 0, data.shape[0] - 1)
            data = data[idx]
        self._cache[key] = data.astype(np.float32, copy=False)
        return self._cache[key]


def _resolve_samples_for_note(
    note, beatmap, cache: _SampleCache,
) -> list[tuple[np.ndarray, float]]:
    """Return a list of (sample_array, gain) pairs to mix at the note's
    press time. Each entry is one sound (normal + any addition bits)."""
    beatmap_dir = cache  # type: ignore[assignment]  # only used for path lookups below
    # (cache only ever sees Path inputs — its `.get(path)` is the loader.)

    # Hit-sample-level overrides → timing-point state → general default.
    sample = getattr(note, "hit_sample", None)
    sample_filename = (sample.filename if sample else "") or ""
    sample_volume = (sample.volume if sample else 0) or 0

    tp = _active_timing_point(beatmap.timing_points, note.time_ms)
    tp_set = tp.sample_set if tp else 0
    tp_index = tp.custom_index if tp else 0
    tp_volume = tp.volume if tp else 100

    # Per-note hitSample's normal_set takes precedence over the timing point.
    effective_set = (sample.normal_set if sample and sample.normal_set
                     else tp_set)
    effective_index = (sample.index if sample and sample.index
                       else tp_index)

    # Set name fallback chain: per-note → timing point → [General] default.
    if effective_set in _SET_NAMES:
        set_name = _SET_NAMES[effective_set]
    else:
        set_name = beatmap.default_sample_set.lower()

    # Volume: per-note override (1-100) → timing point (1-100) → 100.
    vol_pct = sample_volume or tp_volume or 100
    gain = (vol_pct / 100.0) * DEFAULT_HIT_GAIN

    samples: list[tuple[np.ndarray, float]] = []

    def _load_first_existing(candidates: list[Path]) -> np.ndarray | None:
        # The loader is `_SampleCache.get`. `cache` here is the cache obj.
        for p in candidates:
            data = _cache_get(cache, p)
            if data is not None:
                return data
        return None

    # 1. Custom filename override (beatmap-specified exact file): only that
    #    file plays. Skipped when beatmap hitsounds are disabled.
    if sample_filename and getattr(cache, "_use_beatmap", True):
        path = _bm_dir(beatmap, cache) / sample_filename
        arr = _cache_get(cache, path)
        if arr is not None:
            samples.append((arr, gain))
            return samples
        # If the filename was set but the file is missing, fall through to
        # the resolved default so the note still makes a sound.

    # 2. Normal hitsound (always played for every non-miss note).
    candidates = _candidate_paths(_sample_dirs(cache),
                                  set_name, "normal", effective_index)
    arr = _load_first_existing(candidates)
    if arr is not None:
        samples.append((arr, gain))

    # 3. Layered additions (whistle / finish / clap) — only when set.
    addition_set = (sample.addition_set if sample and sample.addition_set
                    else effective_set)
    add_set_name = _SET_NAMES.get(addition_set, set_name)
    for bit, type_name in _ADDITIONS:
        if note.hit_sound & bit:
            cands = _candidate_paths(_sample_dirs(cache),
                                     add_set_name, type_name,
                                     effective_index)
            arr = _load_first_existing(cands)
            if arr is not None:
                samples.append((arr, gain))

    return samples


# Tiny wrappers because `cache` here doubles as a path-lookup carrier — we
# also need the beatmap dir, which we stash on the cache attribute below.

def _cache_get(cache: _SampleCache, path: Path) -> np.ndarray | None:
    return cache.get(path)


def _bm_dir(beatmap, cache: _SampleCache) -> Path:
    return cache._beatmap_dir  # type: ignore[attr-defined]


def _sample_dirs(cache: _SampleCache) -> tuple:
    return getattr(cache, "_sample_dirs", (cache._beatmap_dir,))  # type: ignore[attr-defined]


def _find_combobreak_sample(
    beatmap_dir: Path, skin_dirs: tuple[Path, ...],
) -> Path | None:
    """Look up combobreak.wav: beatmap dir first (mapper override), then any
    of the bot's configured skin directories (Night05, etc.)."""
    candidates = [beatmap_dir / "combobreak.wav", beatmap_dir / "combobreak.ogg"]
    for sd in skin_dirs:
        candidates.append(sd / "combobreak.wav")
        candidates.append(sd / "combobreak.ogg")
    for p in candidates:
        if p.is_file():
            return p
    return None


def build_hitsound_track(
    *,
    judgments_events,            # tuple[JudgmentEvent, ...]
    beatmap,                     # BeatmapInfo
    beatmap_dir: Path,
    output_wav: Path,
    duration_ms: int,
    target_sample_rate: int = 44100,
    audio_rate: float = 1.0,     # noqa: ARG001 — note times already modded
    skin_dirs: tuple[Path, ...] = (),
    beatmap_hitsounds: bool = True,
    nightcore: bool = False,
) -> Path | None:
    """Mix each non-miss note's resolved hitsound(s) at its press time into
    one stereo WAV at ``output_wav``. Returns the path or None on failure.
    """
    # OOM/waste guard (audit #52): the whole build needs soundfile — to
    # decode samples AND to write the WAV. When it is not importable the
    # build is a no-op that would still allocate a ~420MB whole-song
    # float32 buffer (total_samples x 2 x 4B) and then fail the write, so
    # skip early and render without replay hitsounds — identical outcome,
    # minus the wasted allocation + long-map OOM risk.
    import importlib.util
    if importlib.util.find_spec("soundfile") is None:
        log.warning("hitsound_skip_no_soundfile",
                    extra={"output": str(output_wav)})
        return None
    cache = _SampleCache(target_sample_rate)
    cache._beatmap_dir = beatmap_dir  # type: ignore[attr-defined]
    # Sample lookup stack (osu!/lazer BeatmapHitsounds): the beatmap dir when
    # the beatmap-hitsounds toggle is on, then any skin dirs, then the bundled
    # default set. First existing file wins.
    _dirs: list[Path] = []
    if beatmap_hitsounds:
        _dirs.append(beatmap_dir)
    _dirs.extend(skin_dirs)
    if _DEFAULT_HITSOUND_DIR.is_dir():
        _dirs.append(_DEFAULT_HITSOUND_DIR)
    cache._sample_dirs = tuple(_dirs)       # type: ignore[attr-defined]
    cache._use_beatmap = beatmap_hitsounds  # type: ignore[attr-defined]

    total_samples = int(duration_ms / 1000 * target_sample_rate)
    track = np.zeros((total_samples, 2), dtype=np.float32)

    # Build a quick (time_ms → Note) index so we can pull each note's
    # hitsound metadata for a judgment in O(1) average instead of scanning
    # the notes list for every event.
    notes_by_key = {}
    for n in beatmap.notes:
        notes_by_key.setdefault((n.column, n.time_ms), n)

    # Combo-break sample (osu! plays it on any miss that breaks a combo of
    # ≥ COMBO_BREAK_THRESHOLD). Found once up-front; None ⇒ no combobreak
    # WAV available, just skip the miss audio path entirely.
    cb_path = _find_combobreak_sample(beatmap_dir, skin_dirs)
    cb_sample: np.ndarray | None = None
    if cb_path is not None:
        cb_sample = cache.get(cb_path)

    placed = 0
    breaks = 0
    skipped_unknown = 0
    combo = 0
    for j in judgments_events:
        if j.hit_offset_ms is not None and j.judgment != "miss":
            note = notes_by_key.get((j.column, j.time_ms))
            if note is None:
                skipped_unknown += 1
                combo += 1
                continue
            samples = _resolve_samples_for_note(note, beatmap, cache)
            press_ms = j.time_ms + j.hit_offset_ms
            start = int(press_ms / 1000 * target_sample_rate)
            if 0 <= start < total_samples:
                for arr, gain in samples:
                    end = min(start + arr.shape[0], total_samples)
                    length = end - start
                    track[start:end] += arr[:length] * gain
                placed += 1
            combo += 1
        elif j.judgment == "miss":
            # Combo-break sound only fires if the broken combo was big
            # enough — matches osu! stable's 20-hit threshold.
            if cb_sample is not None and combo >= COMBO_BREAK_THRESHOLD:
                # The miss happens at the note's scheduled time (no press
                # to anchor on); place combobreak there.
                start = int(j.time_ms / 1000 * target_sample_rate)
                if 0 <= start < total_samples:
                    end = min(start + cb_sample.shape[0], total_samples)
                    length = end - start
                    track[start:end] += cb_sample[:length] * COMBO_BREAK_GAIN
                    breaks += 1
            combo = 0

    nc_layered = 0
    if nightcore:
        nc_layered = _layer_nightcore(
            track, beatmap.timing_points, cache, skin_dirs,
            target_sample_rate, duration_ms,
        )

    np.clip(track, -1.0, 1.0, out=track)

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    import soundfile
    soundfile.write(str(output_wav), track, target_sample_rate, subtype="PCM_16")
    log.info("hitsound_track_built",
             extra={"path": str(output_wav), "hits": placed,
                    "breaks": breaks, "skipped": skipped_unknown,
                    "nightcore_beats": nc_layered})
    return output_wav


_NIGHTCORE_GAIN = 0.35      # lower than per-note hits so it doesn't dominate


def _layer_nightcore(
    track: np.ndarray, timing_points: tuple, cache: "_SampleCache",
    skin_dirs: tuple[Path, ...], sample_rate: int, duration_ms: int,
) -> int:
    """Layer NC-mod-style claps + finishes on each beat across the song.
    Mirrors lazer's mania NC behaviour: clap on every beat, with a finish
    cymbal on beat 1 of each measure (assumed 4/4)."""
    clap = _find_skin_sample(("normal-hitclap.wav", "soft-hitclap.wav",
                              "drum-hitclap.wav"), skin_dirs, cache)
    finish = _find_skin_sample(("normal-hitfinish.wav", "soft-hitfinish.wav",
                                "drum-hitfinish.wav"), skin_dirs, cache)
    if clap is None and finish is None:
        return 0

    # Walk the uninherited (BPM) timing points in order; each segment has
    # its own beat_length until the next uninherited TP starts.
    red_tps = [tp for tp in timing_points if tp.uninherited]
    if not red_tps:
        return 0

    laid = 0
    for i, tp in enumerate(red_tps):
        beat_ms = max(60.0, tp.beat_length_ms)   # cap < 60ms (>1000 BPM) sanity
        end_ms = red_tps[i + 1].time_ms if i + 1 < len(red_tps) else duration_ms
        t_ms = tp.time_ms
        beat_idx_in_measure = 0
        while t_ms < end_ms and t_ms < duration_ms:
            sample = finish if (beat_idx_in_measure == 0 and finish is not None) else clap
            if sample is not None:
                start = int(t_ms / 1000 * sample_rate)
                if 0 <= start < track.shape[0]:
                    end = min(start + sample.shape[0], track.shape[0])
                    track[start:end] += sample[:end - start] * _NIGHTCORE_GAIN
                    laid += 1
            t_ms += beat_ms
            beat_idx_in_measure = (beat_idx_in_measure + 1) % 4
    return laid


def _find_skin_sample(
    filenames: tuple[str, ...], skin_dirs: tuple[Path, ...],
    cache: "_SampleCache",
) -> np.ndarray | None:
    """First match wins. Used by the nightcore overlay to grab clap/finish
    WAVs out of the bundled or user-uploaded skin dirs."""
    for skin_dir in skin_dirs:
        for name in filenames:
            p = skin_dir / name
            if p.is_file():
                arr = cache.get(p)
                if arr is not None:
                    return arr
    return None
