"""Ground-truth tests for the VAD. Builds audio with known speech/silence
layout, decodes it back through ffmpeg, and checks the detected segments."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .. import ffmpeg
from ..analysis import VadParams, analyze, frame_db, merge_segments

TMP = Path(tempfile.gettempdir()) / "vadtest"
TMP.mkdir(exist_ok=True)
RATE = ffmpeg.ANALYSIS_RATE

PASS = FAIL = 0


def check(label: str, cond: bool, extra: object = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"PASS  {label}")
    else:
        FAIL += 1
        print(f"FAIL  {label}  {extra}")


def tone(seconds: float, freq: float, amp: float, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    wave = amp * np.sin(2 * np.pi * freq * t)
    if noise > 0:
        rng = np.random.default_rng(seed)
        wave = wave + rng.normal(0, noise, wave.size)
    return wave.astype(np.float32)


def silence(seconds: float, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    if noise <= 0:
        return np.zeros(int(seconds * RATE), dtype=np.float32)
    rng = np.random.default_rng(seed)
    return rng.normal(0, noise, int(seconds * RATE)).astype(np.float32)


def write_wav(samples: np.ndarray, name: str) -> Path:
    out = TMP / name
    subprocess.run(
        [ffmpeg.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "f32le", "-ar", str(RATE), "-ac", "1", "-i", "-",
         "-c:a", "pcm_s16le", str(out)],
        input=samples.tobytes(), check=True,
    )
    return out


def wrap_in_video(wav: Path, name: str, w: int = 640, h: int = 360) -> Path:
    out = TMP / name
    subprocess.run(
        [ffmpeg.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc=size={w}x{h}:rate=25",
         "-i", str(wav), "-shortest",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out)],
        check=True,
    )
    return out


def total(segments) -> float:
    return sum(e - s for s, e in segments)


def short_stats(st) -> dict:
    return {k: v for k, v in vars(st).items() if isinstance(v, (int, float))}


# ---------------------------------------------------------------- case 1
# Clean: 4 speech bursts separated by long silences.
print("\n--- case 1: clean speech, obvious silences ---")
audio = np.concatenate([
    tone(1.0, 220, 0.4), silence(1.0),
    tone(1.5, 220, 0.4), silence(0.8),
    tone(0.7, 220, 0.4), silence(1.2),
    tone(2.0, 220, 0.4),
])
truth = [(0, 1), (2, 3.5), (4.3, 5.0), (6.2, 8.2)]
p = write_wav(audio, "clean.wav")
s, k, st = analyze(str(p), VadParams(sensitivity=0.5), duration=8.2)
print("  stats:", {k2: round(v, 1) for k2, v in short_stats(st).items()})
print("  kept:", [(round(a, 2), round(b, 2)) for a, b in k])
check("found 4 kept segments", len(k) == 4, len(k))
check("kept duration ~5.2s", abs(total(k) - 5.2) < 0.6, total(k))
check("every kept segment overlaps a truth segment",
      all(any(a < tb + 0.3 and b > ta - 0.3 for ta, tb in truth) for a, b in k), k)
check("noise floor measured low", st.noise_floor_db < -60, st.noise_floor_db)
check("threshold sits between floor and speech",
      st.noise_floor_db < st.threshold_db < st.speech_level_db, st)

# ---------------------------------------------------------------- case 2
# THE reported bug: a fixed -35 dB threshold on a quiet recording.
print("\n--- case 2: quiet recording with room tone (fixed threshold would fail) ---")
audio = np.concatenate([
    tone(1.2, 200, 0.012),  # ~-38 dBFS speech
    silence(0.9, noise=0.0006, seed=1),  # ~-64 dBFS room tone
    tone(1.0, 200, 0.010),
    silence(0.9, noise=0.0006, seed=2),
    tone(1.3, 200, 0.013),
])
p = write_wav(audio, "quiet.wav")
s, k, st = analyze(str(p), VadParams(sensitivity=0.5), duration=5.3)
print("  stats:", {k2: round(v, 1) for k2, v in short_stats(st).items()})
print("  kept:", [(round(a, 2), round(b, 2)) for a, b in k])
check("found 3 segments in a quiet clip", len(k) == 3, len(k))
check("adaptive threshold is near the room tone, not -35",
      st.threshold_db < -45, st.threshold_db)
check("kept duration ~3.5s", abs(total(k) - 3.5) < 0.7, total(k))

# same clip, but what a hard -35 dBFS ffmpeg threshold would say
from ..analysis import silencedetect
naive = silencedetect(str(p), -35.0, 0.35, 5.3)
print(f"  ffmpeg silencedetect@-35dB regions: {[(round(a,2),round(b,2)) for a,b in naive]}")
check("fixed -35dB swallows the WHOLE clip (this was the bug)", len(naive) == 1, naive)
check("adaptive VAD still finds the silences", len(s) >= 2, s)

# ---------------------------------------------------------------- case 3
# Hysteresis: a decaying tail should not split one utterance into pieces.
print("\n--- case 3: hysteresis keeps a decaying tail together ---")
audio = np.concatenate([
    np.concatenate([tone(0.8, 250, 0.3), tone(0.6, 250, 0.05), tone(0.5, 250, 0.004)]),
    silence(1.0),
    tone(1.5, 250, 0.3),
])
p = write_wav(audio, "tail.wav")
s, k, st = analyze(str(p), VadParams(sensitivity=0.5), duration=4.4)
print("  kept:", [(round(a, 2), round(b, 2)) for a, b in k])
check("decaying tail not split into extra segments", len(k) == 2, len(k))
check("first segment keeps the tail (~1.9s)", 1.5 < (k[0][1] - k[0][0]) < 2.4,
      k[0][1] - k[0][0] if k else None)

# ---------------------------------------------------------------- case 4
# min_silence must actually filter short pauses.
print("\n--- case 4: short pauses are kept, long ones cut ---")
audio = np.concatenate([
    tone(0.6, 300, 0.3), silence(0.15), tone(0.6, 300, 0.3),  # 0.15s breath
    silence(1.2), tone(0.8, 300, 0.3),                        # 1.2s pause
])
p = write_wav(audio, "pause.wav")
s, k, st = analyze(str(p), VadParams(sensitivity=0.5, min_silence=0.5), duration=3.35)
print("  kept:", [(round(a, 2), round(b, 2)) for a, b in k])
check("0.15s breath kept, 1.2s pause cut -> 2 segments", len(k) == 2, len(k))
check("first segment spans the breath (~1.35s)",
      abs((k[0][1] - k[0][0]) - 1.35) < 0.3, k[0][1] - k[0][0] if k else None)

# ---------------------------------------------------------------- case 5
# Sensitivity dial must move the threshold monotonically.
print("\n--- case 5: sensitivity dial is monotonic ---")
thresholds = []
for sens in (0.0, 0.25, 0.5, 0.75, 1.0):
    _, _, st = analyze(str(TMP / "quiet.wav"), VadParams(sensitivity=sens), duration=5.3)
    thresholds.append(round(st.threshold_db, 1))
print("  thresholds:", thresholds)
check("threshold rises with sensitivity", thresholds == sorted(thresholds), thresholds)

# ---------------------------------------------------------------- case 6
# Padding must not clip word onsets.
print("\n--- case 6: padding widens segments ---")
_, k0, _ = analyze(str(TMP / "clean.wav"), VadParams(padding=0.0), duration=8.2)
_, k1, _ = analyze(str(TMP / "clean.wav"), VadParams(padding=0.15), duration=8.2)
print(f"  no padding: {total(k0):.2f}s   with 0.15s padding: {total(k1):.2f}s")
check("padding increases kept time", total(k1) > total(k0) + 0.3, (total(k0), total(k1)))

# ---------------------------------------------------------------- case 7
# Real container path (AAC audio inside MP4) must behave the same.
print("\n--- case 7: AAC audio inside an MP4 container ---")
wav = write_wav(np.concatenate([
    tone(1.0, 220, 0.3), silence(0.9), tone(1.4, 220, 0.3), silence(0.9), tone(0.9, 220, 0.3)
]), "aac.wav")
mp4 = wrap_in_video(wav, "aac.mp4")
s, k, st = analyze(str(mp4), VadParams(sensitivity=0.5), duration=5.1)
print("  kept:", [(round(a, 2), round(b, 2)) for a, b in k])
check("3 segments through an MP4", len(k) == 3, len(k))
check("kept ~3.3s of 5.1s", abs(total(k) - 3.3) < 0.7, total(k))

# ---------------------------------------------------------------- case 8
# Degenerate inputs must not explode.
print("\n--- case 8: degenerate inputs ---")
p = write_wav(np.zeros(RATE * 2, dtype=np.float32), "digital_silence.wav")
s, k, st = analyze(str(p), VadParams(), duration=2.0)
check("digital silence -> no kept segments", len(k) == 0, k)
check("digital silence -> stats still populated", st.duration == 2.0, st.duration)

p = write_wav(tone(2.0, 440, 0.5), "constant.wav")
s, k, st = analyze(str(p), VadParams(), duration=2.0)
check("constant tone -> one full segment", len(k) == 1 and abs(k[0][1] - 2.0) < 0.05, k)

p = write_wav(np.array([0.0], dtype=np.float32), "one_sample.wav")
s, k, st = analyze(str(p), VadParams(), duration=0.0)
check("1-sample file does not crash", True)

# ---------------------------------------------------------------- case 9
print("\n--- case 9: helpers ---")
m = merge_segments([(0, 1), (1.05, 2), (3, 4)], max_gap=0.2, duration=5)
check("merge joins 1.0/1.05 but not 2.0/3.0", m == [(0, 2), (3, 4)], m)
m = merge_segments([(0, 1), (9, 10)], max_gap=0.2, duration=10)
check("merge leaves distant segments alone", m == [(0, 1), (9, 10)], m)

db, hop = frame_db(np.zeros(1000), 16000, 20)
check("frame_db floors digital silence at -100", db.max() <= -99.9, db.max())
check("frame hop is 20ms", abs(hop - 0.02) < 1e-9, hop)

# ---------------------------------------------------------------- case 10
# detect() is the shared entry point for the preview and the export, so it has
# to cover the two cases analyze() cannot see on its own.
print("\n--- case 10: detect() routes by detector flag and audio presence ---")
from ..analysis import detect
from ..jobs import vad_key_of
from ..models import VadSettings

audio = np.concatenate([
    tone(1.0, 220, 0.3), silence(0.9), tone(1.4, 220, 0.3),
    silence(0.9), tone(0.9, 220, 0.3),
])
p = write_wav(audio, "detect.wav")
dur = 5.1

_s, k_vad, _ = detect(str(p), VadParams(min_silence=0.5), dur)
_s, k_ff, _ = detect(str(p), VadParams(min_silence=0.5, use_ffmpeg_detect=True), dur)
print("  RMS VAD kept:", [(round(a, 2), round(b, 2)) for a, b in k_vad])
print("  silencedetect kept:", [(round(a, 2), round(b, 2)) for a, b in k_ff])
check("both detectors find the speech bursts", len(k_vad) == 3 and len(k_ff) == 3,
      (len(k_vad), len(k_ff)))

# No audio track: keep everything, cut nothing.
_s, k_silent, st_silent = detect(str(p), VadParams(), dur, has_audio=False)
check("no audio -> whole clip kept", k_silent == [(0.0, dur)], k_silent)
check("no audio -> no silences reported", _s == [], _s)
check("no audio -> stats still know the duration", st_silent.duration == dur,
      st_silent.duration)

# The cache key must notice a detector switch even when every dial is identical.
plain = VadSettings(min_silence=0.5)
ff = VadSettings(min_silence=0.5, use_ffmpeg_detect=True)
check("vad_key_of distinguishes the detectors",
      vad_key_of(plain) != vad_key_of(ff), (vad_key_of(plain), vad_key_of(ff)))
check("vad_key_of is stable for equal settings", vad_key_of(ff) == vad_key_of(ff))

# stats_to_dict must not ship the ~1400 pair envelope in every response.
from ..analysis import stats_to_dict
_, _, st_full = analyze(str(p), VadParams(), dur)
d = stats_to_dict(st_full)
check("stats_to_dict drops the envelope by default", "envelope" not in d, list(d))
check("stats_to_dict can still include it", "envelope" in stats_to_dict(st_full, True))


print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
