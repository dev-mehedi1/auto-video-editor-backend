"""Encoder presets and final command assembly."""
from __future__ import annotations

from pathlib import Path

from .graph import Plan
from .models import EditRequest, ExportSettings

# video_codec -> (ffmpeg encoder, default container extension, pixel format)
VIDEO_CODECS: dict[str, tuple[str, str, str]] = {
    "h264": ("libx264", "mp4", "yuv420p"),
    "h265": ("libx265", "mp4", "yuv420p"),
    "vp9": ("libvpx-vp9", "webm", "yuv420p"),
    "gif": ("gif", "gif", "yuv420p"),
}

AUDIO_CODECS: dict[str, str] = {
    "aac": "aac",
    "mp3": "libmp3lame",
}

# Audio codecs each container actually accepts. Picking aac for a .webm makes
# the muxer refuse the file outright.
CONTAINER_AUDIO = {
    "webm": "libopus",
    "mp4": {"aac": "aac", "mp3": "libmp3lame", "none": "none"},
    "gif": "none",
}


def audio_codec_for(export: ExportSettings, container: str) -> str:
    if export.audio_codec == "none":
        return "none"
    allowed = CONTAINER_AUDIO.get(container, "aac")
    if isinstance(allowed, str):
        return allowed
    return allowed.get(export.audio_codec, "aac")


def build_command(
    req: EditRequest,
    plan: Plan,
    source: Path,
    output: Path,
) -> list[str]:
    """Full ffmpeg argv for one render."""
    ex = req.export
    enc, ext, pix_fmt = VIDEO_CODECS.get(ex.video_codec, VIDEO_CODECS["h264"])
    acodec = audio_codec_for(ex, ext)

    args: list[str] = ["-y", "-i", str(source)]
    args += plan.extra_input_args

    filter_complex = plan.filter_complex
    maps = list(plan.maps)

    if enc == "gif":
        # GIF needs a fixed 256-colour palette, which has to be generated from
        # a split of the finished video stream. Only one -filter_complex is
        # allowed, so this chains onto the plan's final label.
        filter_complex += (
            f";{plan.video_label}split[gs0][gs1];"
            "[gs0]palettegen=max_colors=200:stats_mode=diff[gp];"
            "[gs1][gp]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle[gout]"
        )
        # The gif muxer rejects audio streams outright.
        maps = ["-map", "[gout]"]

    args += ["-filter_complex", filter_complex]
    args += maps

    if enc == "libx264":
        args += [
            "-c:v", enc,
            "-preset", ex.preset,
            "-crf", str(ex.crf),
            "-pix_fmt", pix_fmt,
            # Needed for smooth playback in browsers and on phones.
            "-profile:v", "high",
            "-level", "4.1",
            "-movflags", "+faststart",
        ]
    elif enc == "libx265":
        args += [
            "-c:v", enc,
            "-preset", ex.preset,
            "-crf", str(ex.crf),
            "-pix_fmt", pix_fmt,
            # x265 spams stderr at default verbosity; cap it and log it away.
            "-x265-params", "log-level=error",
        ]
    elif enc == "libvpx-vp9":
        args += [
            "-c:v", enc,
            "-crf", str(ex.crf),
            "-b:v", "0",
            "-pix_fmt", pix_fmt,
            "-row-mt", "1",
        ]
    elif enc == "gif":
        args += ["-loop", "0"]

    if ex.audio_codec != "none" and plan.has_output_audio:
        args += ["-c:a", acodec, "-b:a", ex.audio_bitrate]

    args.append(str(output))
    return args


def render(
    req: EditRequest,
    plan: Plan,
    source: Path,
    output: Path,
    *,
    on_progress=None,
    cancel=None,
) -> None:
    """Run the render, streaming progress back to the caller."""
    from . import ffmpeg, graph

    try:
        ffmpeg.run_with_progress(
            build_command(req, plan, source, output),
            total_duration=plan.output_duration,
            on_progress=on_progress,
            cancel=cancel,
        )
    finally:
        # Temp drawtext files must go whether ffmpeg succeeded, failed or was
        # cancelled.
        graph.cleanup(plan)
