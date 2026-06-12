"""Shared dataclasses for the renderer pipeline."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HitSample:
    """Per-note hitsound override pulled from the .osu HitObject's tail
    field ``normalSet:additionSet:index:volume:filename``. Any field that's
    0 / empty means "use the active timing point's value"."""
    normal_set: int = 0       # 0=auto, 1=normal, 2=soft, 3=drum
    addition_set: int = 0
    index: int = 0            # 0=auto, N>0 = soft-hitnormalN.wav etc.
    volume: int = 0           # 0=auto, 1-100 = override
    filename: str = ""        # explicit override sample (relative to beatmap)


@dataclass(frozen=True)
class Note:
    column: int  # 0-indexed
    time_ms: int
    hit_sound: int = 0        # bitfield: 0=normal-only, 2=whistle, 4=finish, 8=clap
    hit_sample: HitSample = HitSample()


@dataclass(frozen=True)
class HoldNote:
    column: int
    time_ms: int
    end_time_ms: int
    hit_sound: int = 0
    hit_sample: HitSample = HitSample()

    @property
    def duration_ms(self) -> int:
        return self.end_time_ms - self.time_ms


@dataclass(frozen=True)
class TimingPoint:
    """Sample + scroll-velocity state for a section of the song.

    Hitsounds: ``sample_set`` + ``custom_index`` + ``volume`` come from the
    .osu line directly.

    Scroll: ``sv_multiplier`` is what mania interprets the "negative
    beatLength" field as on INHERITED timing points (red TP keeps SV=1.0).
    Notes scheduled while this TP is active scroll at that multiplier — so
    SV=2.0 means double the scroll speed for that section.
    """
    time_ms: int
    sample_set: int           # 0=default, 1=normal, 2=soft, 3=drum
    custom_index: int         # 0=default sample, N>0 = numbered variant
    volume: int               # 0-100
    sv_multiplier: float = 1.0
    uninherited: bool = True   # True = red TP (BPM); False = green (SV/SS)
    # On uninherited (red) TPs this is the ms-per-beat from the .osu file
    # (e.g. 500 → 120 BPM). Carried so the nightcore-hitsounds overlay can
    # know where the beats are without scanning the .osu again.
    beat_length_ms: float = 500.0


@dataclass(frozen=True)
class KeyEvent:
    time_ms: int
    keys_held: int  # bitmask, bit i = column i

    def is_held(self, column: int) -> bool:
        return bool(self.keys_held & (1 << column))


@dataclass(frozen=True)
class VisualMods:
    hidden: bool = False
    fade_in: bool = False
    flashlight: bool = False
    score_v2: bool = False


@dataclass(frozen=True)
class BeatmapInfo:
    key_count: int
    notes: tuple  # tuple[Note | HoldNote, ...]
    audio_filename: str | None
    background_filename: str | None
    total_duration_ms: int
    audio_lead_in_ms: int
    artist: str
    title: str
    difficulty: str
    creator: str
    beatmap_id: int | None
    beatmapset_id: int | None
    base_scroll_speed: float = 20.0  # osu! default; affects pixel-per-ms
    # Hitsound-related metadata. ``default_sample_set`` is osu!'s sample-set
    # name from [General] (Normal/Soft/Drum); ``timing_points`` is sorted by
    # time_ms and used to look up the active sample state at any moment.
    default_sample_set: str = "Soft"
    timing_points: tuple = ()  # tuple[TimingPoint, ...]
    # OD from `[Difficulty] OverallDifficulty`. Drives lazer-style
    # OD-scaled hit windows in the local judgment classifier.
    overall_difficulty: float = 5.0


@dataclass(frozen=True)
class ReplayInfo:
    mode: int  # 0..3
    beatmap_md5: str
    player_name: str
    replay_md5: str
    mods: int
    key_events: tuple  # tuple[KeyEvent, ...]
    score: int
    accuracy: float
    max_combo: int
    count_geki: int
    count_300: int
    count_katu: int
    count_100: int
    count_50: int
    count_miss: int
    grade: str
    # Mania accuracy denominator weight for the rainbow-300 (geki): 320 for
    # stable replays, 305 for lazer / Score V2 — matches what osu! displays
    # and the bot's website card (osr_parser). Default 305 = lazer.
    mania_max_weight: int = 305


@dataclass(frozen=True)
class RenderOptions:
    resolution: tuple[int, int]
    fps: int
    encoder: str = "auto"  # "auto" | "h264_vaapi" | "libx264"
    encoder_device: str | None = None
    timeout_seconds: int = 600
    audio_required: bool = False
    # 2.5 Mbps is plenty for 720p mania (mostly flat colour over a still bg);
    # gets file size to ~30 MB for a 3-minute song so Discord's inline embed
    # player can buffer it instead of timing out on large downloads.
    video_bitrate: str = "2500k"
    audio_bitrate: str = "160k"
    # Visual toggles — every "feel like lazer" feature is gated by one of
    # these so the future settings page can map each to a checkbox. Defaults
    # match the current rendered look.
    show_hit_lighting: bool = True
    show_receptor_pulse: bool = True
    show_stage_light_flash: bool = True
    # Trails triple the draw-calls per visible note (note + 2 ghosts) and
    # measurably slow down dense charts. Off by default; flip on for
    # cosmetic flair if you don't mind the longer render time.
    show_note_trail: bool = False
    show_hit_error_popup: bool = True
    show_combo_pop: bool = True
    show_combo_tier_color: bool = True
    show_miss_shake: bool = True
    show_progress_bar: bool = True
    show_hp_bar: bool = True
    show_ur_bar: bool = True
    show_kiai_highlight: bool = True
    show_player_sidebar: bool = False  # opt-in; needs osu! API call
    # HUD draw-call gates surfaced by the web settings page. Each toggles
    # one optional element off; defaults keep current behaviour.
    show_score: bool = True
    show_grade: bool = True
    show_key_overlay: bool = True       # the receptor key-press flash
    show_key_counter: bool = True       # bottom-right per-column press counter
    show_combo: bool = True             # the centred combo counter
    show_judgment: bool = True          # the hit-judgement text/sprite burst
    show_pp_counter: bool = False       # off by default; needs rosu-pp live
    show_result_screen: bool = True
    hide_judgement_line: bool = False   # the horizontal line at the receptor
    skip_intro: bool = True             # skip audio_lead_in_ms intro silence
    watermark_text: str = ""
    smooth_sv_transitions: bool = True
    background_dim: float = 0.70       # 0=none, 1=fully black bg (legacy)
    # Stage-aware dim: dim ramps independently in three phases. None ⇒ fall
    # back to background_dim. Phases:
    #   intro  - t < first_note_time
    #   game   - during the note window
    #   breaks - between sections (not yet detected; same as game for now)
    bg_dim_intro: float | None = None
    bg_dim_game: float | None = None
    bg_dim_breaks: float | None = None
    # Background blur: 0 = none, 10 = heavy. Implemented as N gaussian
    # passes at composite time.
    bg_blur: int = 0
    # Scroll speed (osu!mania scale, 1-40). None = use baseline (17). Higher
    # values shrink the approach window proportionally — 34 is twice as fast
    # as 17, 8 is roughly half. Range clamped at CLI level.
    scroll_speed: int | None = None
    # Per-channel audio gain (0.0 - 1.0). 1.0 = unchanged. Applied as ffmpeg
    # `volume=` filters on the song and hitsound chains before amix.
    music_volume: float = 1.0
    hitsound_volume: float = 1.0
    # Audio toggles
    normalize_loudness: bool = True
    audio_fade_out_ms: int = 600
    combo_break_sound: bool = True
    combo_break_threshold: int = 20
    # When False, skips the per-note hitsound dub entirely (just the song).
    use_replay_hitsounds: bool = True
    # When True, layer nightcore-style claps (beats 1+3) and finishes
    # (beat 2) over each measure. Driven by the .osu timing points.
    nightcore_hitsounds: bool = False
    # Use the .osk's WAV samples (normal-hit*, soft-hit*, drum-hit*) in
    # preference to the bundled defaults. Falls back to defaults when a
    # specific sample is missing.
    use_skin_hitsounds: bool = False
    # Combo color source for note tints when an .osk is loaded.
    #   "beatmap" — use Beatmap.colours
    #   "skin"    — use Skin.colours
    # Currently informational; the renderer atlas is pre-tinted so a full
    # implementation needs a colour-override pass.
    skin_combo_colors: str = "beatmap"
    # Try to download the player's chosen skin (from osu! profile). Needs
    # an authenticated API call + skin-mirror integration; not yet wired.
    use_replay_skin: bool = False
