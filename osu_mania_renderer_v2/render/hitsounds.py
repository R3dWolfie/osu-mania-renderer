"""Resolve replay samples and mix a stereo PCM track before ffmpeg's song mix.

Samples fall back from beatmap to skin to bundled defaults. Production stable
and lazer triggers come from their separate immutable gameplay facts.
"""
from __future__ import annotations

import importlib
import logging
import tempfile
from collections import OrderedDict
from heapq import merge
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from osu_mania_renderer_v2.beatmap.models import HoldNote, Note
from osu_mania_renderer_v2.errors import RendererError

log = logging.getLogger("osu_mania_renderer_v2.render.hitsounds")

COMBO_BREAK_GAIN = 1.0  # AudioEngine.PlaySample / lazer SampleInfo default volume.
_SET_NAMES = {1: "normal", 2: "soft", 3: "drum"}
_ADDITIONS = ((2, "whistle"), (4, "finish"), (8, "clap"))
_DEFAULT_HITSOUND_DIR = Path(__file__).resolve().parent.parent / "assets" / "default_hitsounds"
_DEFAULT_NC_DIR = Path(__file__).resolve().parent.parent / "assets" / "default_nightcore"


def require_hitsound_runtime():
    # Check before resolving samples or opening the output. A broken libsndfile
    # installation is just as systemic as an absent Python package.
    try:
        return importlib.import_module("soundfile")
    except (ImportError, OSError) as exc:
        raise RendererError(
            "Replay/NC hitsounds require soundfile and libsndfile. Reinstall the renderer's "
            "runtime dependencies (python -m pip install 'soundfile>=0.12'); "
            "source installations may also need system libsndfile."
        ) from exc


def _active_timing_point(timing_points: tuple, time_ms: float):
    if not timing_points:
        return None
    lo, hi = 0, len(timing_points) - 1
    if timing_points[0].time_ms > time_ms:
        return timing_points[0]
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if timing_points[mid].time_ms <= time_ms:
            lo = mid
        else:
            hi = mid - 1
    return timing_points[lo]


def _candidate_paths(dirs, set_name: str, type_name: str, index: int) -> list[Path]:
    out = []
    for directory in dirs:
        if index > 0:
            out.extend(directory / f"{set_name}-hit{type_name}{index}.{ext}"
                       for ext in ("wav", "ogg"))
        out.extend(directory / f"{set_name}-hit{type_name}.{ext}" for ext in ("wav", "ogg"))
    return out


class _SampleCache:
    """Bound decoded sample reuse; invalid files allow the next fallback."""

    def __init__(self, target_rate: int, *, beatmap_dir: Path = Path("."),
                 skin_dirs: tuple[Path, ...] = (), beatmap_hitsounds: bool = True):
        self.target_rate = target_rate
        self.soundfile = require_hitsound_runtime()
        self.beatmap_dir = beatmap_dir
        self.use_beatmap = beatmap_hitsounds
        self.sample_dirs = ((beatmap_dir,) if beatmap_hitsounds else ()) + skin_dirs + (
            _DEFAULT_HITSOUND_DIR,)
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self._missing = set()
        self.max_cache_bytes = 32 * 1024 * 1024

    def get(self, path: Path) -> np.ndarray | None:
        key = str(path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if key in self._missing:
            return None
        if not path.is_file():
            if len(self._missing) >= 4096:
                self._missing.clear()
            self._missing.add(key)
            return None
        try:
            data, rate = self.soundfile.read(str(path), dtype="float32", always_2d=True)
            if not len(data) or not np.isfinite(data).all():
                raise ValueError("empty or non-finite sample")
            if data.shape[1] == 1:
                data = np.repeat(data, 2, axis=1)
            elif data.shape[1] != 2:
                raise ValueError("hitsound sample must be mono or stereo")
            if rate != self.target_rate:
                ratio = self.target_rate / rate
                new_len = max(1, int(len(data) * ratio))
                idx = np.clip((np.arange(new_len) / ratio).astype(np.int64), 0, len(data) - 1)
                data = data[idx]
        except (OSError, ValueError, self.soundfile.SoundFileError) as exc:
            log.warning("hitsound_load_failed", extra={"path": str(path), "err": str(exc)})
            return None
        data = data.astype(np.float32, copy=False)
        while self._cache and self._cache_bytes + data.nbytes > self.max_cache_bytes:
            _, previous = self._cache.popitem(last=False)
            self._cache_bytes -= previous.nbytes
        if data.nbytes <= self.max_cache_bytes:
            self._cache[key] = data
            self._cache_bytes += data.nbytes
        return data


@dataclass(frozen=True)
class SampleLayer:
    samples: np.ndarray
    gain: float
    path: Path
    type_name: str


def _sample_gain(volume: int, type_name: str, *, is_lazer_replay: bool) -> float:
    if is_lazer_replay:
        # SkinnableSound + DrawableHitObject.MINIMUM_SAMPLE_VOLUME.
        return max(volume, 5) / 100.0
    if type_name == "custom":
        # HitObject.PlaySound -> AudioEngine.PlaySample: no layered floor.
        return volume / 100.0
    coefficient = {"normal": .8, "finish": 1.0, "whistle": .85, "clap": .85}[type_name]
    # AudioEngine.PlayHitSamples truncates the percentage to an integer.
    return int(max(volume, 8) * coefficient) / 100.0


def _resolve_samples_for_note(note, beatmap, cache: _SampleCache,
                              *, is_lazer_replay: bool = False) -> list[SampleLayer]:
    sample = note.hit_sample
    tp = _active_timing_point(beatmap.timing_points, note.time_ms + (0 if is_lazer_replay else 2))
    effective_set = sample.normal_set or (tp.sample_set if tp else 0)
    effective_index = sample.index or (tp.custom_index if tp else 0)
    set_name = _SET_NAMES.get(effective_set, beatmap.default_sample_set.lower())
    # A note's zero means inherit; a control point's zero is a real volume.
    volume = sample.volume or (tp.volume if tp else 100)
    layers = []

    def load(candidates, type_name):
        for path in candidates:
            data = cache.get(path)
            if data is not None:
                return SampleLayer(data, _sample_gain(volume, type_name,
                    is_lazer_replay=is_lazer_replay), path, type_name)
        return None

    if sample.filename and cache.use_beatmap:
        layer = load((cache.beatmap_dir / sample.filename,), "custom")
        if layer is not None:
            return [layer]
    layer = load(_candidate_paths(cache.sample_dirs, set_name, "normal", effective_index), "normal")
    if layer is not None:
        layers.append(layer)
    addition_set = _SET_NAMES.get(sample.addition_set, set_name)
    for bit, type_name in _ADDITIONS:
        if note.hit_sound & bit:
            layer = load(_candidate_paths(cache.sample_dirs, addition_set, type_name,
                                           effective_index), type_name)
            if layer is not None:
                layers.append(layer)
    return layers


@dataclass(frozen=True)
class ReplaySoundEvent:
    time_ms: float
    kind: str
    note: Note | HoldNote | None = None
    silent_node: bool = False


def _lazer_sound_events(facts, notes, audio_rate: float):
    # Use original audio-time metadata, with the SAME floating video keys as
    # the combo simulator. Rounded display-note times must not drive lookup.
    heads = {(n.column, n.time_ms / audio_rate): n for n in notes}
    tails = {(n.column, n.end_time_ms / audio_rate): n
             for n in notes if isinstance(n, HoldNote)}
    for fact in facts:
        if fact.kind == "reset":
            yield ReplaySoundEvent(fact.time_ms, "reset")
        elif fact.kind == "increment" and fact.source in ("tap", "head", "tail"):
            key = (fact.column, fact.object_time_ms)
            if fact.source == "tail":
                parent = tails.get(key)
                silent = parent is not None and parent.tail_hit_sample is None
                node = (Note(parent.column, parent.end_time_ms, parent.tail_hit_sound,
                             parent.tail_hit_sample) if parent is not None and not silent else None)
                yield ReplaySoundEvent(fact.time_ms, "increment", node, silent)
            else:
                yield ReplaySoundEvent(fact.time_ms, "increment", heads.get(key))


def _stable_sound_events(judgments_events, notes, audio_rate: float):
    heads = {(n.column, int(n.time_ms / audio_rate)): n for n in notes}
    for event in judgments_events:
        if event.judgment == "miss":
            yield ReplaySoundEvent(event.time_ms, "reset")
        elif event.hit_offset_ms is not None:
            # Compatibility for direct callers without source sound facts.
            # Production uses _stable_source_sound_events and its LN lifecycle.
            yield ReplaySoundEvent(event.time_ms + event.hit_offset_ms, "increment",
                heads.get((event.column, event.time_ms)), silent_node=event.is_tail)


def _stable_source_sound_events(facts, notes, audio_rate):
    parents = {(n.column, n.time_ms / audio_rate): n for n in notes}
    for fact in facts:
        note = parents.get((fact.column, fact.object_time_ms))
        if fact.source == "final" and isinstance(note, HoldNote):
            # Converted stable long notes have an audible end by default;
            # this is independent of lazer's optional/empty tail node.
            note = Note(note.column, note.end_time_ms,
                        note.tail_hit_sound if note.tail_hit_sample is not None else note.hit_sound,
                        note.tail_hit_sample or note.hit_sample)
        yield ReplaySoundEvent(fact.time_ms, "sound", note)


def _combo_break_times(facts, *, is_lazer_replay, threshold=20, mods=0):
    """Only real combo transitions reset/play; repeated zero resets are silent."""
    combo = 0
    first_break = True
    disabled = not is_lazer_replay and mods & ((1 << 7) | (1 << 13))  # Relax / Autopilot
    for fact in facts:
        if fact.kind == "increment":
            combo += 1
        elif fact.kind == "reset":
            old_combo, combo = combo, 0
            if old_combo == 0:
                continue
            play = old_combo > max(0, threshold) or (is_lazer_replay and first_break)
            first_break = False
            if play and not disabled:
                yield fact.time_ms


def _find_combobreak_sample(beatmap_dir: Path | None, skin_dirs: tuple[Path, ...]) -> Path | None:
    directories = ((beatmap_dir,) if beatmap_dir is not None else ()) + skin_dirs + (_DEFAULT_HITSOUND_DIR,)
    for directory in directories:
        for ext in ("wav", "ogg", "mp3"):
            path = directory / f"combobreak.{ext}"
            if path.is_file():
                return path
    return None


@dataclass(frozen=True)
class SamplePlacement:
    start: int
    samples: np.ndarray
    gain: float
    kind: str = "gameplay"


def _mix_blocks(placements, total_samples: int, chunk_frames: int):
    """One chunk plus active sample tails; never allocate duration-sized PCM.

    Gameplay, combo breaks, then overlays retain the previous accumulation
    order within each output frame. Chunk boundaries cannot change rounding.
    """
    pending = iter(enumerate(placements))
    upcoming = next(pending, None)
    active = []
    priority = {"gameplay": 0, "combo_break": 1, "nightcore": 2, "nc_mod": 3}
    for offset in range(0, total_samples, chunk_frames):
        end = min(offset + chunk_frames, total_samples)
        active = [(index, p) for index, p in active if p.start + len(p.samples) > offset]
        while upcoming is not None and upcoming[1].start < end:
            active.append(upcoming)
            upcoming = next(pending, None)
        active.sort(key=lambda item: (priority[item[1].kind], item[0]))
        block = np.zeros((end - offset, 2), dtype=np.float32)
        gameplay_peak = 0.0
        for gameplay in (True, False):
            for _, p in active:
                if (p.kind == "gameplay") != gameplay:
                    continue
                start, stop = max(offset, p.start), min(end, p.start + len(p.samples))
                if stop > start:
                    block[start-offset:stop-offset] += p.samples[start-p.start:stop-p.start] * p.gain
            if gameplay and block.size:
                gameplay_peak = float(np.max(np.abs(np.floor(block * 32768)))) / 32768
        np.clip(block, -1.0, 1.0, out=block)
        yield block, gameplay_peak


def build_hitsound_track(
    *, judgments_events=(), lazer_facts=None, is_lazer_replay: bool = False,
    stable_sound_facts=None, combo_facts=None, combo_break_sound: bool = True,
    combo_break_threshold: int = 20, mods: int = 0,
    sample_notes=None, beatmap, beatmap_dir: Path, output_wav: Path, duration_ms: int,
    target_sample_rate: int = 44100, audio_rate: float = 1.0,
    skin_dirs: tuple[Path, ...] = (), overlay_skin_dirs: tuple[Path, ...] | None = None,
    beatmap_hitsounds: bool = True, miss_hitsound: bool = True, nightcore: bool = False, nc_mod: bool = False,
    gameplay_end_ms: float | None = None, chunk_frames: int = 16384,
) -> Path:
    """Stream gameplay and overlay samples into an atomic stereo PCM WAV.

    PCM memory is bounded by chunk_frames, active tails and the 32 MiB decode
    cache, independent of video duration. Source facts stay immutable.
    """
    if chunk_frames <= 0:
        raise RendererError("Hitsound chunk size must be positive.")
    cache = _SampleCache(target_sample_rate, beatmap_dir=beatmap_dir,
                         skin_dirs=skin_dirs, beatmap_hitsounds=beatmap_hitsounds)
    if is_lazer_replay and lazer_facts is None:
        raise RendererError("Lazer replay hitsounds require source-factual gameplay events.")
    notes = beatmap.notes if sample_notes is None else sample_notes
    events = (_lazer_sound_events(lazer_facts, notes, audio_rate) if is_lazer_replay
              else _stable_source_sound_events(stable_sound_facts, notes, audio_rate)
              if stable_sound_facts is not None
              else _stable_sound_events(judgments_events, notes, audio_rate))
    # The old stable compatibility carrier is in scheduled-note order.
    events = sorted(events, key=lambda event: event.time_ms)
    total_samples = int(duration_ms / 1000 * target_sample_rate)
    combo_events = (combo_facts if combo_facts is not None else lazer_facts if is_lazer_replay else events)
    cb_path = (_find_combobreak_sample(beatmap_dir if beatmap_hitsounds else None, skin_dirs)
               if miss_hitsound and combo_break_sound else None)
    cb_sample = cache.get(cb_path) if cb_path is not None else None
    counts = dict(eligible_hit_events=0, resolved_hit_events=0, resolved_sample_layers=0,
                  unresolved_hit_events=0, mixed_sample_layers=0, combo_break_layers=0,
                  silent_node_events=0, zero_gain_hit_events=0, nightcore_beats=0, nc_mod_beats=0)

    def gameplay_layers():
        for event in events:
            start = int(event.time_ms / 1000 * target_sample_rate)
            in_range = 0 <= start < total_samples
            if event.kind == "reset":
                continue
            if not in_range:
                continue
            if event.silent_node:
                counts["silent_node_events"] += 1
                continue
            counts["eligible_hit_events"] += 1
            layers = (_resolve_samples_for_note(event.note, beatmap, cache,
                        is_lazer_replay=is_lazer_replay) if event.note is not None else [])
            if not layers:
                counts["unresolved_hit_events"] += 1
                continue
            counts["resolved_hit_events"] += 1
            counts["resolved_sample_layers"] += len(layers)
            if all(layer.gain == 0 for layer in layers):
                counts["zero_gain_hit_events"] += 1
            for layer in layers:
                yield SamplePlacement(start, layer.samples, layer.gain)

    def combo_layers():
        if cb_sample is None:
            return
        for time_ms in _combo_break_times(combo_events, is_lazer_replay=is_lazer_replay,
                                          threshold=combo_break_threshold, mods=mods):
            start = int(time_ms / 1000 * target_sample_rate)
            if 0 <= start < total_samples:
                yield SamplePlacement(start, cb_sample, COMBO_BREAK_GAIN, "combo_break")

    overlay_dirs = skin_dirs if overlay_skin_dirs is None else overlay_skin_dirs
    overlays = (_nightcore_layers(beatmap.timing_points, cache, overlay_dirs,
        target_sample_rate, duration_ms, gameplay_end_ms=gameplay_end_ms)
        if nightcore and not nc_mod else ())
    nc_layers = (_nightcore_mod_layers(beatmap.timing_points, cache, overlay_dirs,
        target_sample_rate, duration_ms, audio_rate, play_hats=True,
        gameplay_end_ms=gameplay_end_ms) if nc_mod else ())

    def counted_placements():
        count_key = {"gameplay": "mixed_sample_layers", "combo_break": "combo_break_layers",
                     "nightcore": "nightcore_beats", "nc_mod": "nc_mod_beats"}
        for placement in merge(gameplay_layers(), combo_layers(), overlays, nc_layers, key=lambda p: p.start):
            counts[count_key[placement.kind]] += 1
            yield placement

    temporary = None
    peak = gameplay_peak = sum_squares = 0.0
    try:
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output_wav.parent, prefix=output_wav.name + ".",
                                         suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
        with cache.soundfile.SoundFile(temporary, mode="w", samplerate=target_sample_rate,
                                      channels=2, format="WAV", subtype="PCM_16") as writer:
            for block, hit_peak in _mix_blocks(counted_placements(), total_samples, chunk_frames):
                gameplay_peak = max(gameplay_peak, hit_peak)
                pcm = np.floor(block * 32768).clip(-32768, 32767) / 32768
                peak = max(peak, float(np.max(np.abs(pcm))))
                sum_squares += float(np.sum(pcm * pcm, dtype=np.float64))
                writer.write(block)
        rms = (sum_squares / (total_samples * 2)) ** .5 if total_samples else 0.0
        diagnostics = {"path": str(output_wav), **counts, "peak": peak, "rms": rms,
                       "chunk_frames": chunk_frames}
        expected_audible = counts["eligible_hit_events"] - counts["zero_gain_hit_events"]
        if expected_audible and (not counts["resolved_sample_layers"] or gameplay_peak == 0 or peak == 0):
            log.error("hitsound_track_unusable", extra=diagnostics)
            raise RendererError("Replay hitsounds resolved no audible gameplay samples. Check the "
                                "bundled default_hitsounds assets and sample decoding; see hitsound diagnostics.")
        temporary.replace(output_wav)
    except (OSError, ValueError, cache.soundfile.SoundFileError) as exc:
        raise RendererError(f"Cannot write replay hitsound WAV {output_wav}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    log.info("hitsound_track_built", extra=diagnostics)
    return output_wav


_NIGHTCORE_GAIN = 0.14      # lower than per-note hits so it doesn't dominate


def _nightcore_layers(
    timing_points: tuple, cache: "_SampleCache",
    skin_dirs: tuple[Path, ...], sample_rate: int, duration_ms: int,
    gameplay_end_ms=None,
):
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

    # Beat overlay stops at gameplay end, not into the results outro the
    # video appends (taiko fix ac73af2). horizon is in the same (video-ms)
    # base as duration_ms.
    horizon = duration_ms if gameplay_end_ms is None else min(duration_ms, float(gameplay_end_ms))
    for i, tp in enumerate(red_tps):
        beat_ms = max(60.0, tp.beat_length_ms)   # cap < 60ms (>1000 BPM) sanity
        end_ms = red_tps[i + 1].time_ms if i + 1 < len(red_tps) else horizon
        t_ms = tp.time_ms
        beat_idx_in_measure = 0
        while t_ms < end_ms and t_ms < horizon:
            sample = finish if (beat_idx_in_measure == 0 and finish is not None) else clap
            if sample is not None:
                start = int(t_ms / 1000 * sample_rate)
                if 0 <= start < int(duration_ms / 1000 * sample_rate):
                    yield SamplePlacement(start, sample, _NIGHTCORE_GAIN, "nightcore")
            t_ms += beat_ms
            beat_idx_in_measure = (beat_idx_in_measure + 1) % 4
    return


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


# --- ModNightcore beat overlay (NC-mod-gated, distinct from the metronome) -----

_NC_MOD_GAIN = 0.20      # nightcore-kick/clap/hat/finish drums


def _nightcore_mod_layers(
    timing_points: tuple, cache: "_SampleCache",
    skin_dirs: tuple[Path, ...], sample_rate: int, duration_ms: int,
    audio_rate: float, *, play_hats: bool = True, gameplay_end_ms=None,
):
    """osu! ModNightcore beat overlay — the drum pattern osu! plays on each
    beat AUTOMATICALLY while the Nightcore mod is active. NOT the general
    metronome (_layer_nightcore) above; both can lay. Half-beat grid
    (BeatSyncedContainer Divisor=2): per 4/4 bar, kick on beats 1 & 3, clap on
    2 & 4, hat on the off-beats (the '&'s), plus a finish cymbal at the start
    of every 4th bar — from the SKIN's nightcore-kick/-clap/-hat/-finish
    samples (silent sample → silence; missing sample → skipped, no synth).
    Mania has no readily-available SliderTickRate so hats play unconditionally
    (osu gates them on SliderTickRate%2==0); the measure is assumed 4/4 (no
    signature stored on the mania timing point). Map-time beats are mapped to
    video time via `audio_rate` (NC ⇒ 1.5). Returns samples laid.

    Port of osu.Game/Rulesets/Mods/ModNightcore.NightcoreBeatContainer."""
    rate = audio_rate or 1.0
    # Skin dirs first, then the bundled osu! DEFAULT last: a skin's SILENT file
    # (in a skin dir) resolves first and wins; a sample the skin OMITS falls
    # back to the default (osu! default-skin parity).
    nc_dirs = tuple(skin_dirs) + (_DEFAULT_NC_DIR,)
    samples = {
        name: _find_skin_sample(
            (f"nightcore-{name}.wav", f"nightcore-{name}.ogg",
             f"nightcore-{name}.mp3"), nc_dirs, cache)
        for name in ("kick", "clap", "hat", "finish")
    }
    if not any(v is not None for v in samples.values()):
        return 0
    red_tps = [tp for tp in timing_points if tp.uninherited]
    if not red_tps:
        return 0
    # stop at gameplay end, not into the results outro (taiko fix ac73af2)
    dur = duration_ms if gameplay_end_ms is None else min(duration_ms, float(gameplay_end_ms))
    horizon_map = dur * rate                    # video horizon back to map time
    seg_len = 4 * 8                            # 4/4: beatsPerBar(4) * 2 * 4 bars
    for i, tp in enumerate(red_tps):
        beat_ms = max(60.0, tp.beat_length_ms)   # cap <60ms (>1000 BPM) sanity
        half = beat_ms / 2.0
        end_map = red_tps[i + 1].time_ms if i + 1 < len(red_tps) else horizon_map
        end_map = min(end_map, horizon_map)
        k = 0
        t_map = float(tp.time_ms)
        while t_map < end_map:
            bseg = k % seg_len
            r = bseg % 4
            names = []
            if r == 0:
                names.append("kick")
            elif r == 2:
                names.append("clap")
            elif play_hats:
                names.append("hat")
            if bseg == 0:
                names.append("finish")
            start = int((t_map / rate) / 1000.0 * sample_rate)
            if 0 <= start < int(duration_ms / 1000 * sample_rate):
                for name in names:
                    sample = samples.get(name)
                    if sample is not None:
                        yield SamplePlacement(start, sample, _NC_MOD_GAIN, "nc_mod")
            k += 1
            t_map = tp.time_ms + k * half
    return
