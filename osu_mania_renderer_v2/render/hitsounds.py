"""Resolve replay samples and mix a stereo PCM track before ffmpeg's song mix.

Eligible beatmap banks fall back to skin base names, then bundled defaults.
Stable and lazer triggers come from their separate immutable gameplay facts.
"""
from __future__ import annotations

import importlib
import logging
import tempfile
import math
import re
from collections import OrderedDict
from heapq import merge
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from osu_mania_renderer_v2.beatmap.models import HoldNote, Note
from osu_mania_renderer_v2.errors import RendererError
from osu_mania_renderer_v2.render.sample_paths import is_relative_sample_name, sample_file

log = logging.getLogger("osu_mania_renderer_v2.render.hitsounds")

COMBO_BREAK_GAIN = 1.0  # AudioEngine.PlaySample / lazer SampleInfo default volume.
_SET_NAMES = {1: "normal", 2: "soft", 3: "drum"}
_ADDITIONS = ((2, "whistle"), (4, "finish"), (8, "clap"))
_DEFAULT_HITSOUND_DIR = Path(__file__).resolve().parent.parent / "assets" / "default_hitsounds"
_DEFAULT_NC_DIR = Path(__file__).resolve().parent.parent / "assets" / "default_nightcore"
# Service safety bound per candidate, separate from the retained PCM LRU.
# Includes source decode, mono expansion, finite checking and worst-case
# nearest-neighbour index/conversion/output temporaries. Not a duration limit.
MAX_SAMPLE_WORK_BYTES = 64 * 1024 * 1024


def _sample_decode_geometry(info, target_rate):
    values = (info.frames, info.samplerate, info.channels, target_rate)
    if any(not isinstance(v, (int, float)) or v <= 0 or
           (isinstance(v, float) and (not math.isfinite(v) or not v.is_integer()))
           for v in values):
        raise ValueError("invalid sample frame/rate/channel metadata")
    frames, rate, channels, target = map(int, values)
    if channels not in (1, 2) or not 1 <= rate <= 768000 or not 1 <= target <= 768000:
        raise ValueError("only mono/stereo samples at 1..768000 Hz are supported")
    output_frames = max(1, frames * target // rate)
    work_bytes = frames * channels * 5  # float32 decode + finite mask
    if channels == 1:
        work_bytes += frames * 8  # stereo expansion, source still live
    if rate != target:
        work_bytes += output_frames * 48  # float/int index temporaries + stereo output
    if work_bytes > MAX_SAMPLE_WORK_BYTES:
        raise ValueError(f"estimated decode/resample work {work_bytes} bytes exceeds "
                         f"{MAX_SAMPLE_WORK_BYTES}-byte safety limit; shorten or resample this hitsound")
    return frames, rate, output_frames


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


@dataclass(frozen=True)
class SampleCandidate:
    path: Path
    beatmap: bool = False


def _candidate_paths(cache, set_name: str, type_name: str, index: int,
                     *, legacy_names: bool = False) -> list[SampleCandidate]:
    """Legacy beatmap banks differ from ordinary skin/default sample names."""
    base = f"{set_name}-hit{type_name}"
    sources = []
    if cache.use_beatmap and index >= 1:
        sources.append((cache.beatmap_dir, base + (str(index) if index >= 2 else ""), True))
    sources.extend((directory, base, False) for directory in (*cache.skin_dirs, _DEFAULT_HITSOUND_DIR))
    out = []
    for directory, name, beatmap in sources:
        for ext in ("wav", "mp3", "ogg"):
            path = directory / f"{name}.{ext}"
            # Stable's null-coalescing byte lookup happens within each codec,
            # before decoding. A corrupt banked file cannot try the old name
            # for that codec; a missing banked file can (only before v5).
            if beatmap and legacy_names and cache.resolve(path) is None:
                path = directory / f"hit{type_name}.{ext}"
            out.append(SampleCandidate(path, beatmap))
    return out


class _SampleCache:
    """Bound decoded sample reuse; invalid files allow the next fallback."""

    def __init__(self, target_rate: int, *, beatmap_dir: Path = Path("."),
                 skin_dirs: tuple[Path, ...] = (), beatmap_hitsounds: bool = True,
                 overlay_skin_dirs: tuple[Path, ...] = ()):
        self.target_rate = target_rate
        self.soundfile = require_hitsound_runtime()
        self.beatmap_dir = beatmap_dir
        self.skin_dirs = skin_dirs
        self.use_beatmap = beatmap_hitsounds
        self.roots = tuple(dict.fromkeys(p.absolute() for p in (
            *((beatmap_dir,) if beatmap_hitsounds else ()), *skin_dirs, *overlay_skin_dirs,
            _DEFAULT_HITSOUND_DIR, _DEFAULT_NC_DIR)))
        self._cache = OrderedDict()
        self._cache_bytes = 0
        # Static failure identities live for this render; unlike PCM they must
        # not be evicted and retried thousands of times on dense charts.
        self._missing = set()
        self._beatmap_sizes = {}
        # A valid zero-length BASS sample: resolved silence, distinct from None.
        self._silence = np.zeros((1, 2), dtype=np.float32)
        self._silence.flags.writeable = False
        self._explicit_paths = {}  # Stable LoadBeatmapSample cache is keyed by stem.
        self.max_cache_bytes = 32 * 1024 * 1024

    def resolve(self, path):
        path = Path(path).absolute()
        for root in self.roots:
            try:
                relative = str(path.relative_to(root if path.is_relative_to(root) else root.resolve()))
            except ValueError:
                continue
            # Do not let an invalid path under one root become eligible under
            # another broader root. Every candidate has one permitted tier.
            return sample_file(root, relative)
        return None

    def stable_file(self, filename, *, format_version=14):
        if not is_relative_sample_name(filename):
            return None, None
        filename = filename.replace("\\", "/")
        path = self.beatmap_dir / filename
        # Stable uses IndexOf/LastIndexOf over the whole filename string,
        # including directory dots, and excludes a dot at position zero.
        has_extension = filename.find(".") > 0
        stem = filename[:filename.rfind(".")] if has_extension else filename
        key = (stem.lower(), format_version)
        if key not in self._explicit_paths:
            # An existing exact name is authoritative, even if decoding fails.
            candidates = ([path] if has_extension and self.resolve(path) is not None else
                          [self.beatmap_dir / f"{name}.{ext}" for ext in ("wav", "ogg", "mp3")
                           for name in ((stem, filename) if format_version < 5 else (stem,))])
            # LoadBeatmapSample selects the first existing bytes, then calls
            # BASS once. A corrupt selected stem must not try another codec.
            resolved = next((p for candidate in candidates
                             if (p := self.resolve(candidate)) is not None), None)
            data = self.get(resolved, stable_beatmap=True) if resolved is not None else None
            self._explicit_paths[key] = resolved if data is not None else None
            return (resolved, data) if data is not None else (None, None)
        resolved = self._explicit_paths[key]
        return (resolved, self.get(resolved, stable_beatmap=True)) if resolved is not None else (None, None)

    def get(self, path: Path, *, stable_beatmap: bool = False) -> np.ndarray | None:
        path = self.resolve(path)
        if path is None:
            return None
        key = str(path)
        if stable_beatmap:
            try:
                if key not in self._beatmap_sizes:
                    self._beatmap_sizes[key] = path.stat().st_size
                if self._beatmap_sizes[key] < 1024:
                    return self._silence
            except OSError:
                pass  # normal failure handling below
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if key in self._missing:
            return None
        if not path.is_file():
            self._missing.add(key)
            return None
        try:
            # Preflight before any PCM decode, then validate the opened file
            # again so changed metadata cannot bypass the allocation bound.
            _sample_decode_geometry(self.soundfile.info(str(path)), self.target_rate)
            with self.soundfile.SoundFile(str(path)) as reader:
                frames, rate, new_len = _sample_decode_geometry(reader, self.target_rate)
                data = reader.read(frames=frames, dtype="float32", always_2d=True)
            if not len(data) or not np.isfinite(data).all():
                raise ValueError("empty or non-finite sample")
            if data.shape[1] == 1:
                data = np.repeat(data, 2, axis=1)
            elif data.shape[1] != 2:
                raise ValueError("hitsound sample must be mono or stereo")
            if rate != self.target_rate:
                ratio = self.target_rate / rate
                new_len = max(1, len(data) * self.target_rate // rate)
                idx = np.clip((np.arange(new_len) / ratio).astype(np.int64), 0, len(data) - 1)
                data = data[idx]
        except ValueError as exc:
            log.warning("hitsound_sample_rejected", extra={"path": str(path), "err": str(exc)})
            self._missing.add(key)
            return None
        except (OSError, self.soundfile.SoundFileError) as exc:
            log.warning("hitsound_load_failed", extra={"path": str(path), "err": str(exc)})
            self._missing.add(key)
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


def _general_sample_bank(value: str, *, is_lazer_replay: bool) -> int:
    # Both loaders use case-sensitive Enum.Parse, including unnamed int32
    # values. Normalize decimal numeric spellings; never turn arbitrary text into a
    # path. Malformed input retains the defensive client zero-bank fallback.
    names = {"None": 0, "Normal": 1, "Soft": 2, "Drum": 3}
    if not is_lazer_replay:
        names["All"] = -1
    value = value.strip()
    if value in names:
        result = names[value]
    elif re.fullmatch(r"[+-]?[0-9]{1,10}", value):
        result = int(value)
        if not -(2**31) <= result < 2**31:
            return 0
    else:
        return 0
    return result


def _sample_control_time(note, *, source_mode: int, is_lazer_replay: bool):
    if not is_lazer_replay:
        return note.time_ms + 2
    if source_mode == 3:
        # LegacyBeatmapDecoder applies samples before pass-through conversion.
        # Native ConvertHold is not IHasRepeats: its head inherits end + 5.
        return (note.end_time_ms if isinstance(note, HoldNote) else note.time_ms) + 5
    # Converted carriers lack the original slider/repeat sample-node identity.
    # Preserve that existing lookup rather than infer it from generated holds.
    return note.time_ms


def _resolve_samples_for_note(note, beatmap, cache: _SampleCache,
                              *, is_lazer_replay: bool = False) -> list[SampleLayer]:
    sample = note.hit_sample
    tp = _active_timing_point(beatmap.timing_points, _sample_control_time(
        note, source_mode=beatmap.source_mode, is_lazer_replay=is_lazer_replay))
    if tp is None and not is_lazer_replay:
        # HitCircleMania.PlaySound only dispatches when ControlPointAtBin
        # returns a point. Stable itself rejects maps without timing points.
        return []
    fields = tp.field_count if tp else 8
    if fields == 3 and not is_lazer_replay:
        return []  # stable ParseTimingPoint requires either two or >=4 fields
    format_version = getattr(beatmap, "format_version", 14)
    timing_set = tp.sample_set if tp else 0
    if tp is not None and fields < 4:
        timing_set = _general_sample_bank(beatmap.default_sample_set, is_lazer_replay=is_lazer_replay)
    timing_index = tp.custom_index if tp else 0
    if fields == 2 and not is_lazer_replay:
        timing_index = getattr(beatmap, "custom_samples", None)
        if timing_index is None:
            timing_index = int(format_version < 4)
    effective_set = sample.normal_set or timing_set
    effective_index = sample.index or timing_index
    # Stable ControlPoint maps None to Soft; lazer's timing decoder maps it
    # to Normal. Neither inherits [General] SampleSet for an explicit zero.
    default_set = "normal" if is_lazer_replay else "soft"
    def bank_name(value):
        if value == 0:
            return default_set
        return _SET_NAMES.get(value, str(value) if is_lazer_replay else "normal")

    set_name = bank_name(effective_set)
    # A note's zero means inherit; a control point's zero is a real volume.
    timing_volume = tp.volume if tp else 100
    if tp is not None and fields < 6:
        timing_volume = (100 if fields == 2 and not is_lazer_replay
                         else getattr(beatmap, "sample_volume", 100))
    volume = sample.volume or timing_volume
    legacy_names = not is_lazer_replay and format_version < 5
    layers = []

    def bank_index(set_value):
        # Stable's built-in cache switch maps All/unknown values to Normal,
        # but its extra-bank preload only loads Normal/Soft/Drum identities.
        if not is_lazer_replay and set_value not in (0, 1, 2, 3) and effective_index > 2:
            return 0
        return effective_index

    def load(candidates, type_name):
        for candidate in candidates:
            path = cache.resolve(candidate.path)
            if path is None:
                continue
            data = cache.get(path, stable_beatmap=candidate.beatmap and not is_lazer_replay)
            if data is not None:
                return SampleLayer(data, _sample_gain(volume, type_name,
                    is_lazer_replay=is_lazer_replay), path, type_name)
        return None

    if sample.filename and cache.use_beatmap:
        if not is_lazer_replay:
            path, data = cache.stable_file(sample.filename, format_version=format_version)
            if data is not None:
                return [SampleLayer(data, _sample_gain(volume, "custom", is_lazer_replay=False), path, "custom")]
    if sample.filename and is_lazer_replay:
        # FileHitSampleInfo forces bank Normal/CSS1. Each legacy skin tries
        # Filename, extensionless stem, then its ordinary lookup names before
        # moving to the next skin. ResourceStore appends wav/mp3/ogg to each.
        filename = sample.filename.replace("\\", "/")
        names = ([filename, str(Path(filename).with_suffix(""))]
                 if is_relative_sample_name(filename) else [])
        names.extend(("Gameplay/normal-hitnormal", "hitnormal"))
        sources = ((cache.beatmap_dir,) if cache.use_beatmap else ()) + cache.skin_dirs + (_DEFAULT_HITSOUND_DIR,)
        for root in sources:
            for name in dict.fromkeys(part for n in names for part in (n, n.rsplit("/", 1)[-1])):
                layer = load([SampleCandidate(root / (name + suffix))
                              for suffix in ("", ".wav", ".mp3", ".ogg")], "custom")
                if layer is not None:
                    layers.append(layer)
                    break
            if layers:
                break
        # FileHitSampleInfo replaces only the primary carrier in lazer;
        # additions remain independent even if that primary cannot resolve.
    elif beatmap.source_mode != 3 or note.hit_sound == 0 or note.hit_sound & 1:
        # Native Mania never adds layered hitnormal to addition-only flags.
        # Lazer converts allow that carrier; retain converted behavior here.
        layer = load(_candidate_paths(cache, set_name, "normal", bank_index(effective_set),
                                     legacy_names=legacy_names), "normal")
        if layer is not None:
            layers.append(layer)
    addition_set = bank_name(sample.addition_set) if sample.addition_set else set_name
    for bit, type_name in _ADDITIONS:
        if note.hit_sound & bit:
            layer = load(_candidate_paths(cache, addition_set, type_name,
                                           bank_index(sample.addition_set or effective_set),
                                           legacy_names=legacy_names), type_name)
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
                # Keep the raw parent HoldNote so its end-time sample metadata
                # survives display-time rounding/rate changes at the head.
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
        if fact.source in ("final", "empty-final") and isinstance(note, HoldNote):
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


def _find_combobreak_sample(beatmap_dir: Path | None, skin_dirs: tuple[Path, ...], cache=None,
                           *, is_lazer_replay=False) -> SampleCandidate | None:
    directories = ([(beatmap_dir, True)] if beatmap_dir is not None else [])
    directories.extend((directory, False) for directory in (*skin_dirs, _DEFAULT_HITSOUND_DIR))
    for directory, beatmap in directories:
        for ext in ("wav", "mp3", "ogg"):
            path = sample_file(directory, f"combobreak.{ext}")
            if path is not None and (cache is None or cache.get(
                    path, stable_beatmap=beatmap and not is_lazer_replay) is not None):
                return SampleCandidate(path, beatmap)
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

    PCM output uses chunk_frames, active tails and a 32 MiB retained cache.
    Each candidate has a separate 64 MiB decode/resample work limit. Neither
    ceiling caps total process RSS or simultaneous tails. Source facts stay immutable.
    """
    if chunk_frames <= 0:
        raise RendererError("Hitsound chunk size must be positive.")
    cache = _SampleCache(target_sample_rate, beatmap_dir=beatmap_dir,
                         skin_dirs=skin_dirs, beatmap_hitsounds=beatmap_hitsounds,
                         overlay_skin_dirs=overlay_skin_dirs or ())
    if is_lazer_replay and lazer_facts is None:
        raise RendererError("Lazer replay hitsounds require source-factual gameplay events.")
    notes = beatmap.notes if sample_notes is None else sample_notes
    if not is_lazer_replay and cache.use_beatmap:
        # Stable processes filenames while constructing objects, not in the
        # order replay presses later happen to trigger their audio.
        for note in sorted(notes, key=lambda note: note.time_ms):
            for sample in (note.hit_sample, getattr(note, "tail_hit_sample", None)):
                if sample is not None and sample.filename:
                    cache.stable_file(sample.filename, format_version=getattr(beatmap, "format_version", 14))
    events = (_lazer_sound_events(lazer_facts, notes, audio_rate) if is_lazer_replay
              else _stable_source_sound_events(stable_sound_facts, notes, audio_rate)
              if stable_sound_facts is not None
              else _stable_sound_events(judgments_events, notes, audio_rate))
    # The old stable compatibility carrier is in scheduled-note order.
    events = sorted(events, key=lambda event: event.time_ms)
    total_samples = int(duration_ms / 1000 * target_sample_rate)
    combo_events = (combo_facts if combo_facts is not None else lazer_facts if is_lazer_replay else events)
    cb_candidate = (_find_combobreak_sample(beatmap_dir if beatmap_hitsounds else None, skin_dirs, cache,
                                           is_lazer_replay=is_lazer_replay)
               if miss_hitsound and combo_break_sound else None)
    cb_sample = (cache.get(cb_candidate.path, stable_beatmap=cb_candidate.beatmap and not is_lazer_replay)
                 if cb_candidate is not None else None)
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
                       "gameplay_peak": gameplay_peak, "chunk_frames": chunk_frames}
        # Valid authored silence (including stable's tiny-file sentinel) is
        # a resolved sample. Waveform energy cannot identify decoder failure.
        if counts["eligible_hit_events"] and not counts["resolved_sample_layers"]:
            log.error("hitsound_track_unusable", extra=diagnostics)
            raise RendererError("Replay hitsounds resolved no valid gameplay samples. Check the "
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
