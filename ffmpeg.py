"""Low level ffmpeg / ffprobe process helpers.

Everything here shells out to the native ffmpeg binary. No wasm, no memory
ceiling, real libx264.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

FFMPEG = os.environ.get("FFMPEG_BINARY", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE_BINARY", "ffprobe")

# Windows consoles default to cp1252 which mangles utf-8 filter output.
_UTF8 = {"PYTHONIOENCODING": "utf-8"}

_CLOCK = re.compile(r"^(\d+):(\d{2}):(\d{2}(?:\.\d+)?)$")


class FFmpegError(RuntimeError):
    """An ffmpeg/ffprobe invocation failed. Carries the tail of stderr."""

    def __init__(self, message: str, stderr: str = "", returncode: int = -1):
        self.stderr = stderr
        self.returncode = returncode
        tail = "\n".join(stderr.strip().splitlines()[-25:])
        super().__init__(f"{message}{'\n' + tail if tail else ''}")


class Cancelled(RuntimeError):
    """The caller asked the running job to stop."""


def _base_cmd(binary: str) -> list[str]:
    """Common flags.

    `-nostdin` is ffmpeg-only. ffprobe rejects it ("Failed to set value
    '-loglevel' for option 'nostdin'"), so it must not be passed there.
    """
    cmd = [binary, "-hide_banner", "-loglevel", "error"]
    probe_name = os.path.basename(str(FFPROBE)).lower()
    if os.path.basename(str(binary)).lower() != probe_name:
        cmd.append("-nostdin")
    return cmd


def _decode_clock(value: str) -> float:
    m = _CLOCK.match(value.strip())
    if not m:
        return 0.0
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def run(
    args: Sequence[str],
    *,
    binary: str = FFMPEG,
    capture_stdout: bool = True,
    on_stderr_line: Callable[[str], None] | None = None,
    timeout: float | None = None,
) -> str:
    """Run ffmpeg/ffprobe and return stdout. Raises FFmpegError on failure."""
    cmd = _base_cmd(binary) + list(args)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, **_UTF8},
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise FFmpegError(f"{binary} timed out after {timeout}s")

    if on_stderr_line and err:
        for line in err.splitlines():
            on_stderr_line(line)

    if proc.returncode != 0:
        raise FFmpegError(
            f"{binary} exited with code {proc.returncode}", err, proc.returncode
        )
    return out or ""


def run_with_progress(
    args: Sequence[str],
    *,
    total_duration: float = 0.0,
    on_progress: Callable[[float], None] | None = None,
    cancel: threading.Event | None = None,
    timeout: float | None = None,
) -> None:
    """Run ffmpeg, translating `-progress` output into 0..1 callbacks.

    We parse the `out_time=HH:MM:SS.fff` line rather than `out_time_ms`,
    which despite its name is really microseconds in most ffmpeg builds.
    """
    cmd = (
        _base_cmd(FFMPEG)
        + ["-progress", "pipe:1", "-nostats"]
        + list(args)
    )
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, **_UTF8},
    )

    stderr_chunks: list[str] = []

    def drain_stderr() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_chunks.append(line)

    t = threading.Thread(target=drain_stderr, daemon=True)
    t.start()

    cancelled = False
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            if cancel is not None and cancel.is_set():
                cancelled = True
                proc.kill()
                break
            line = raw.strip()
            if line.startswith("out_time=") and on_progress and total_duration > 0:
                done = _decode_clock(line.split("=", 1)[1])
                on_progress(max(0.0, min(1.0, done / total_duration)))
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise FFmpegError(f"ffmpeg timed out after {timeout}s")
    finally:
        t.join(timeout=2)
        for pipe in (proc.stdout, proc.stderr):
            if pipe is not None:
                try:
                    pipe.close()
                except OSError:
                    pass

    if cancelled:
        raise Cancelled()

    if proc.returncode != 0:
        raise FFmpegError(
            f"ffmpeg exited with code {proc.returncode}",
            "".join(stderr_chunks),
            proc.returncode,
        )


# --------------------------------------------------------------------------
# ffprobe
# --------------------------------------------------------------------------

def probe(path: str | Path) -> dict:
    """Return the full ffprobe JSON for a file."""
    out = run(
        [
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        binary=FFPROBE,
    )
    return json.loads(out or "{}")


@dataclass
class MediaInfo:
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_video: bool = False
    has_audio: bool = False
    video_codec: str = ""
    audio_codec: str = ""
    audio_sample_rate: int = 0
    audio_channels: int = 0
    rotation: int = 0
    display_width: int = 0
    display_height: int = 0
    bitrate: int = 0

    @property
    def aspect(self) -> float:
        if self.display_width and self.display_height:
            return self.display_width / self.display_height
        if self.width and self.height:
            return self.width / self.height
        return 16 / 9


def _parse_fps(value: str | None) -> float:
    if not value:
        return 0.0
    if "/" in value:
        num, _, den = value.partition("/")
        try:
            n, d = float(num), float(den)
            return n / d if d else 0.0
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def media_info(path: str | Path) -> MediaInfo:
    """Summarise a media file, accounting for the rotation side-data matrix."""
    data = probe(path)
    info = MediaInfo()

    fmt = data.get("format") or {}
    try:
        info.duration = float(fmt.get("duration") or 0.0)
    except (TypeError, ValueError):
        info.duration = 0.0
    try:
        info.bitrate = int(fmt.get("bit_rate") or 0)
    except (TypeError, ValueError):
        info.bitrate = 0

    for stream in data.get("streams") or []:
        kind = stream.get("codec_type")
        if kind == "video" and not info.has_video:
            info.has_video = True
            info.video_codec = stream.get("codec_name") or ""
            info.width = int(stream.get("width") or 0)
            info.height = int(stream.get("height") or 0)
            info.fps = _parse_fps(stream.get("avg_frame_rate")) or _parse_fps(
                stream.get("r_frame_rate")
            )
            # Phones record a rotation matrix instead of swapping w/h.
            for sd in stream.get("side_data_list") or []:
                if "rotation" in sd:
                    try:
                        info.rotation = int(round(float(sd["rotation"]))) % 360
                    except (TypeError, ValueError):
                        pass
            info.display_width = info.width
            info.display_height = info.height
            if info.rotation in (90, 270):
                info.display_width, info.display_height = info.height, info.width
        elif kind == "audio" and not info.has_audio:
            info.has_audio = True
            info.audio_codec = stream.get("codec_name") or ""
            try:
                info.audio_sample_rate = int(stream.get("sample_rate") or 0)
            except (TypeError, ValueError):
                info.audio_sample_rate = 0
            info.audio_channels = int(stream.get("channels") or 0)

    # Some containers only carry duration on the stream.
    if info.duration <= 0:
        for stream in data.get("streams") or []:
            try:
                info.duration = max(info.duration, float(stream.get("duration") or 0.0))
            except (TypeError, ValueError):
                pass

    if info.fps <= 0:
        info.fps = 30.0
    return info


# --------------------------------------------------------------------------
# Audio extraction for analysis
# --------------------------------------------------------------------------

ANALYSIS_RATE = 16000
ANALYSIS_CHANNELS = 1


def decode_pcm(path: str | Path, rate: int = ANALYSIS_RATE) -> np.ndarray:
    """Decode any media file to mono float32 PCM in [-1, 1].

    Uses ffmpeg's native resampler so we don't need soundfile/librosa.
    Returns the samples; the rate is implicit in `rate`.
    """
    cmd = _base_cmd(FFMPEG) + [
        "-i", str(path),
        "-vn", "-sn", "-dn",
        "-ac", str(ANALYSIS_CHANNELS),
        "-ar", str(rate),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-",
    ]
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={**os.environ, **_UTF8},
    )
    if proc.returncode != 0:
        raise FFmpegError(
            "could not decode audio",
            proc.stderr.decode("utf-8", "replace"),
            proc.returncode,
        )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def concat_demuxer(segments: Iterable[tuple[float, float]], source: Path, target: Path) -> Path:
    """Write a concat demuxer file listing each kept time range.

    This is the fast path for "just remove the silences": no re-encode of the
    video, only stream copy when the codec allows it.
    """
    lines = ["ffconcat version 1.0"]
    for start, end in segments:
        lines.append(f"file '{source.resolve().as_posix()}'")
        lines.append(f"inpoint {start:.3f}")
        lines.append(f"outpoint {end:.3f}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


@dataclass
class RunResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""
    warnings: list[str] = field(default_factory=list)


def has_encoder(name: str) -> bool:
    try:
        out = run(["-encoders"], capture_stdout=True)
        return re.search(rf"\s{name}\s", out) is not None
    except FFmpegError:
        return False
