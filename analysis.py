"""Audio analysis: adaptive voice activity detection and silence mapping.

The old frontend used a fixed -35 dBFS threshold, which is why auto-cut felt
random: a quiet room sits around -55 dB and a hot mic peaks at -6 dB, so one
global number cannot fit both. Everything here is plain DSP - no ML.

Pipeline
--------
1. frame the signal (20 ms hop)
2. per-frame RMS -> dBFS
3. estimate the *noise floor* and *speech level* from percentiles
4. derive an adaptive threshold between them
5. hysteresis so we do not chatter on breaths
6. morphological cleanup: drop blips, bridge gaps
7. invert to kept segments, then pad/merge to taste
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict, field

import numpy as np

try:
    from . import ffmpeg
except ImportError:
    import ffmpeg

EPS = 1e-10
FLOOR_DB = -100.0


# --------------------------------------------------------------------------
# Framing
# --------------------------------------------------------------------------

def frame_db(samples: np.ndarray, rate: int, window_ms: float = 20.0) -> tuple[np.ndarray, float]:
    """Return (per-frame dBFS, hop_seconds)."""
    if samples.size == 0:
        return np.zeros(0, dtype=np.float64), window_ms / 1000.0

    hop = max(1, int(rate * window_ms / 1000.0))
    n_frames = samples.size // hop
    if n_frames < 1:
        # Shorter than one window: treat the whole thing as a single frame.
        power = np.array([float(np.mean(samples.astype(np.float64) ** 2))])
    else:
        # Centre-crop to whole frames so no partial tail skews the stats.
        trimmed = samples[: n_frames * hop].reshape(n_frames, hop)
        power = np.mean(trimmed.astype(np.float64) ** 2, axis=1)

    db = 10.0 * np.log10(np.maximum(power, EPS))
    return np.clip(db, FLOOR_DB, 0.0), hop / rate


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return FLOOR_DB
    return float(np.percentile(values, q))


# --------------------------------------------------------------------------
# Thresholding
# --------------------------------------------------------------------------

@dataclass
class VadParams:
    """Tunables exposed to the UI."""

    # 0..1 dial. 0 = pick the threshold automatically from the noise floor,
    # 1 = treat almost everything as silence (very aggressive cutting).
    sensitivity: float = 0.5
    # Minimum silence length worth cutting, seconds.
    min_silence: float = 0.35
    # Seconds of audio kept either side of a detected speech run, so words
    # are not clipped.
    padding: float = 0.06
    # Discard kept segments shorter than this (they read as a glitch).
    min_segment: float = 0.18
    # Hysteresis in dB: a frame must exceed threshold+hysteresis to count as
    # speech again once we have dropped into a pause.
    hysteresis: float = 3.0
    # Absolute overrides. When None they are derived from the signal.
    threshold_db: float | None = None
    # Use ffmpeg's own silencedetect instead of the RMS VAD above. It is
    # cruder but very hard to fool, which matters for music and room tone.
    use_ffmpeg_detect: bool = False

    def clamp(self) -> "VadParams":
        self.sensitivity = float(min(1.0, max(0.0, self.sensitivity)))
        self.min_silence = float(min(10.0, max(0.05, self.min_silence)))
        self.padding = float(min(2.0, max(0.0, self.padding)))
        self.min_segment = float(min(10.0, max(0.02, self.min_segment)))
        self.hysteresis = float(min(20.0, max(0.0, self.hysteresis)))
        return self


@dataclass
class AudioStats:
    duration: float = 0.0
    noise_floor_db: float = FLOOR_DB
    speech_level_db: float = 0.0
    peak_db: float = 0.0
    threshold_db: float = FLOOR_DB
    silence_db: float = 0.0
    speech_db: float = 0.0
    # For the waveform view: min/max envelope per bucket, in dBFS.
    envelope: list[tuple[float, float]] = field(default_factory=list)


def estimate_threshold(
    db: np.ndarray, params: VadParams, stats: AudioStats
) -> float:
    """Work out where silence ends and speech begins for *this* recording."""
    if params.threshold_db is not None:
        return float(params.threshold_db)

    # True digital silence: every frame is at the floor, so nothing can ever
    # clear a threshold. Bail out early rather than invent one.
    if float(np.max(db)) <= FLOOR_DB + 0.5:
        return FLOOR_DB

    # The 10th percentile is a solid noise floor estimate: even speech-heavy
    # clips are mostly pauses, and the 90th percentile is a solid speech peak.
    noise = percentile(db, 10)
    speech = percentile(db, 90)
    stats.noise_floor_db = noise
    stats.speech_level_db = speech

    if not np.isfinite(speech) or speech - noise < 3.0:
        # Flat but non-silent signal (a constant tone, a held note). There is
        # no pause to find, so park the threshold just under the level and
        # keep the whole clip.
        return speech - 6.0

    # Place the cut on a log scale between floor and speech, then let the
    # sensitivity dial slide it. 0.5 lands at the geometric midpoint.
    lo = noise + 3.0
    hi = speech - 6.0
    if hi <= lo:
        return lo
    frac = 0.5 + (params.sensitivity - 0.5) * 0.8
    frac = min(0.95, max(0.05, frac))
    return lo + (hi - lo) * frac


def detect_voice(
    db: np.ndarray,
    hop: float,
    duration: float,
    params: VadParams,
    stats: AudioStats,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Return (silence_ranges, kept_ranges) in seconds."""
    if db.size == 0:
        return [], []

    stats.peak_db = float(np.max(db))
    stats.silence_db = percentile(db, 15)
    stats.speech_db = percentile(db, 85)

    threshold = estimate_threshold(db, params, stats)
    stats.threshold_db = threshold

    # --- 1. hysteresis state machine -----------------------------------
    is_speech = np.zeros(db.size, dtype=bool)
    state = False  # start pessimistic: assume silence until proven otherwise
    hi = threshold + params.hysteresis
    for i, level in enumerate(db):
        if state:
            if level < threshold:
                state = False
        else:
            if level > hi:
                state = True
        is_speech[i] = state

    # --- 2. morphological cleanup ---------------------------------------
    # A "speech" run shorter than min_segment is a click or a lip smack, not
    # a word, so treat it as silence.
    min_silence_frames = max(1, round(params.min_silence / hop))
    min_speech_frames = max(1, round(params.min_segment / hop))

    for start, end, value in _runs(is_speech, hop):
        if value and (end - start) < min_speech_frames:
            is_speech[start:end] = False

    # Conversely, a pause shorter than min_silence is a breath between words
    # and should stay in the video, so promote it back to speech.
    is_silence = ~is_speech
    for start, end, value in _runs(is_silence, hop):
        if value and (end - start) < min_silence_frames:
            is_speech[start:end] = True

    # --- 3. split into silence + kept ranges -----------------------------
    # _runs yields contiguous, gap-free, non-overlapping ranges covering the
    # whole signal, so we can emit them straight out.
    silences: list[tuple[float, float]] = []
    kept: list[tuple[float, float]] = []

    for start, end, is_speech_run in _runs(is_speech, hop):
        s, e = start * hop, end * hop
        if is_speech_run:
            kept.append((s, e))
        else:
            silences.append((s, e))

    # --- 4. pad the kept ranges outward, then merge ---------------------
    if params.padding > 0:
        padded: list[tuple[float, float]] = []
        for s, e in kept:
            s = max(0.0, s - params.padding)
            e = min(duration, e + params.padding)
            if padded and s <= padded[-1][1]:
                padded[-1] = (padded[-1][0], max(padded[-1][1], e))
            else:
                padded.append((s, e))
        kept = padded

    kept = [(s, e) for s, e in kept if (e - s) >= params.min_segment]

    # Padding may have swallowed small silences; recompute for display.
    silences = invert(kept, duration)

    return silences, kept


def _runs(mask: np.ndarray, hop: float) -> list[tuple[int, int, bool]]:
    """Compress a boolean mask into (start_idx, end_idx, value) runs."""
    if mask.size == 0:
        return []
    change = np.flatnonzero(np.diff(mask.astype(np.int8)))
    edges = np.concatenate(([0], change + 1, [mask.size]))
    out: list[tuple[int, int, bool]] = []
    for i in range(len(edges) - 1):
        s, e = int(edges[i]), int(edges[i + 1])
        out.append((s, e, bool(mask[s])))
    return out


def invert(kept: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    """Complement of the kept ranges."""
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for s, e in kept:
        if s > cursor:
            gaps.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < duration:
        gaps.append((cursor, duration))
    return gaps


def merge_segments(
    segments: list[tuple[float, float]],
    max_gap: float,
    duration: float,
) -> list[tuple[float, float]]:
    """Join kept ranges separated by less than `max_gap`."""
    if not segments:
        return []
    out = [list(segments[0])]
    for s, e in segments[1:]:
        if s - out[-1][1] <= max_gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(max(0.0, s), min(duration, e)) for s, e in out if e > s]


def build_envelope(
    samples: np.ndarray, rate: int, buckets: int = 1400
) -> list[tuple[float, float]]:
    """Peak envelope (min, max) per bucket for the waveform canvas."""
    if samples.size == 0 or buckets <= 0:
        return []
    buckets = min(buckets, max(1, samples.size))
    edges = np.linspace(0, samples.size, buckets + 1, dtype=np.int64)
    lows: list[float] = []
    highs: list[float] = []
    for i in range(buckets):
        s, e = edges[i], max(edges[i] + 1, edges[i + 1])
        chunk = samples[s:e]
        if chunk.size == 0:
            lows.append(0.0)
            highs.append(0.0)
            continue
        lows.append(float(np.min(chunk)))
        highs.append(float(np.max(chunk)))
    return list(zip(lows, highs))


# --------------------------------------------------------------------------
# ffmpeg silencedetect (used as a cross-check / for huge files)
# --------------------------------------------------------------------------

_START = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_END = re.compile(r"silence_end:\s*(-?\d+(?:\.\d+)?)")


def silencedetect(
    path: str, noise_db: float, min_duration: float, duration: float
) -> list[tuple[float, float]]:
    """Regions ffmpeg's own silencedetect considers silent."""
    import subprocess

    cmd = [
        ffmpeg.FFMPEG, "-hide_banner", "-nostdin",
        "-i", path,
        "-af", f"silencedetect=noise={noise_db}dB:d={min_duration}",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                          encoding="utf-8", errors="replace")
    text = proc.stderr or ""
    starts = [float(m) for m in _START.findall(text)]
    ends = [float(m) for m in _END.findall(text)]
    out: list[tuple[float, float]] = []
    for s, e in zip(starts, ends):
        if e > s:
            out.append((max(0.0, s), min(duration, e)))
    return out


# --------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------

def analyze(
    path: str,
    params: VadParams | None = None,
    duration: float = 0.0,
    envelope_buckets: int = 1400,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], AudioStats]:
    """Full analysis. Returns (silences, kept, stats)."""
    params = (params or VadParams()).clamp()

    samples = ffmpeg.decode_pcm(path)
    rate = ffmpeg.ANALYSIS_RATE
    stats = AudioStats()
    stats.duration = duration or (samples.size / rate if samples.size else 0.0)

    if samples.size == 0:
        return [], [], stats

    db, hop = frame_db(samples, rate)
    silences, kept = detect_voice(db, hop, stats.duration, params, stats)
    stats.envelope = build_envelope(samples, rate, envelope_buckets)
    return silences, kept, stats


def detect(
    path: str,
    params: VadParams | None = None,
    duration: float = 0.0,
    has_audio: bool = True,
    envelope_buckets: int = 1400,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], AudioStats]:
    """Single entry point for "which parts of this file should we keep?".

    Honours params.use_ffmpeg_detect, and keeps the whole clip when there is no
    audio track -- cutting a silent video on its nonexistent audio would delete
    real footage. Both the /api/analyze route and the render worker call this so
    the preview and the export can never disagree.
    """
    params = (params or VadParams()).clamp()

    if not has_audio:
        stats = AudioStats(duration=duration)
        whole = [(0.0, duration)] if duration > 0 else []
        return [], whole, stats

    if params.use_ffmpeg_detect:
        # ffmpeg's detector takes a fixed threshold and duration, so the
        # sensitivity dial is mapped onto both.
        noise_db = params.threshold_db
        if noise_db is None:
            noise_db = -30.0 - 20.0 * params.sensitivity
        raw = silencedetect(path, noise_db, params.min_silence, duration)
        kept = merge_segments(
            [seg for seg in invert(raw, duration)
             if seg[1] - seg[0] >= params.min_segment],
            0.0,
            duration,
        )
        padded = []
        for s, e in kept:
            s = max(0.0, s - params.padding)
            e = min(duration, e + params.padding)
            if e - s >= params.min_segment:
                padded.append((s, e))
        silences = invert(padded, duration)
        stats = AudioStats(
            duration=duration,
            noise_floor_db=noise_db,
            speech_level_db=-20.0,
            threshold_db=noise_db,
        )
        return silences, padded, stats

    return analyze(path, params, duration, envelope_buckets)


def stats_to_dict(stats: AudioStats, include_envelope: bool = False) -> dict:
    """Serialisable stats.

    The envelope is ~1400 [lo, hi] pairs. It is excluded by default because it
    dwarfs every other field in the JSON and the UI gets its waveform from the
    dedicated /waveform endpoint instead.
    """
    d = asdict(stats)
    if include_envelope:
        d["envelope"] = [[lo, hi] for lo, hi in stats.envelope]
    else:
        d.pop("envelope", None)
    return d
