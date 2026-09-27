"""Pydantic schemas for the edit pipeline.

One flat `EditRequest` so the frontend can post the whole "recipe" at once
and the user can customise every knob.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

AspectPreset = Literal["source", "16:9", "9:16", "1:1", "4:5", "21:9", "4:3"]
SpeedMode = Literal["normal", "rampUp", "rampDown", "rampUpDown"]
TransitionKind = Literal["cut", "fade", "xfade"]
XfadeTransition = Literal[
    "fade", "fadeblack", "fadewhite", "slideleft", "slideright",
    "slideup", "slidedown", "circleopen", "circleclose", "dissolve",
    "pixelize", "wipeleft", "wiperight", "radial", "smoothleft",
]
ZoomStyle = Literal["none", "punchIn", "punchOut", "shake", "left", "right"]
TextAlign = Literal["topLeft", "top", "topRight", "left", "center", "right",
                    "bottomLeft", "bottom", "bottomRight"]
FitMode = Literal["fit", "fill", "stretch"]


class VadSettings(BaseModel):
    sensitivity: float = Field(0.5, ge=0.0, le=1.0)
    min_silence: float = Field(0.35, ge=0.05, le=10.0)
    padding: float = Field(0.06, ge=0.0, le=2.0)
    min_segment: float = Field(0.18, ge=0.02, le=10.0)
    hysteresis: float = Field(3.0, ge=0.0, le=20.0)
    # When true, the analysis uses ffmpeg's own silencedetect instead of the
    # adaptive VAD (useful for music beds with a hard dynamic range).
    use_ffmpeg_detect: bool = False
    threshold_db: float | None = Field(None, ge=-90.0, le=0.0)


class AutoCutSettings(BaseModel):
    enabled: bool = True
    vad: VadSettings = Field(default_factory=VadSettings)
    # Merge kept segments separated by less than this (keeps the flow).
    bridge_gap: float = Field(0.0, ge=0.0, le=5.0)
    zoom: ZoomStyle = "punchIn"
    # 0..1, mapped to a 1.0 - 1.3 punch-in.
    zoom_amount: float = Field(0.6, ge=0.0, le=1.0)
    # Punch in on every Nth cut, so it does not feel mechanical.
    zoom_every: int = Field(2, ge=1, le=20)
    speed: SpeedMode = "normal"
    # 0.5 .. 2.0
    speed_value: float = Field(1.0, ge=0.25, le=4.0)
    # Seconds of silence to leave at the very start / end.
    lead_in: float = Field(0.0, ge=0.0, le=5.0)
    tail_out: float = Field(0.0, ge=0.0, le=5.0)


class AudioSettings(BaseModel):
    # High-pass to kill desk rumble / fan noise.
    highpass_hz: int = Field(0, ge=0, le=2000)
    lowpass_hz: int = Field(0, ge=0, le=20000)
    # afftdn strength, 0 = off.
    denoise: float = Field(0.0, ge=0.0, le=1.0)
    # dB
    gain_db: float = Field(0.0, ge=-40.0, le=40.0)
    # EBU R128 target
    normalize: bool = True
    loudness_target: float = Field(-14.0, ge=-40.0, le=-5.0)
    # Compressor
    compress: bool = False
    comp_threshold: float = Field(-18.0, ge=-60.0, le=0.0)
    comp_ratio: float = Field(3.0, ge=1.0, le=20.0)
    deesser: bool = False
    # Replace the audio with silence (for visual-only edits)
    mute: bool = False
    # Per-segment fade in/out at the cuts, seconds.
    cut_fade: float = Field(0.0, ge=0.0, le=1.0)


class VideoSettings(BaseModel):
    # Rescale / reframe to a delivery aspect.
    aspect: AspectPreset = "source"
    fit: FitMode = "fit"
    # Bar colour used when `fit` has to letterbox.
    pad_color: str = "black"
    # Target long edge; 0 keeps the source size.
    max_width: int = Field(0, ge=0, le=7680)
    max_height: int = Field(0, ge=0, le=4320)
    # fps; 0 keeps the source rate
    fps: float = Field(0.0, ge=0.0, le=120.0)
    # eq
    brightness: float = Field(0.0, ge=-1.0, le=1.0)
    contrast: float = Field(1.0, ge=0.0, le=3.0)
    saturation: float = Field(1.0, ge=0.0, le=3.0)
    gamma: float = Field(1.0, ge=0.1, le=3.0)
    sharpen: float = Field(0.0, ge=0.0, le=2.0)
    vignette: float = Field(0.0, ge=0.0, le=1.0)
    # Rotate 0/90/180/270
    rotate: int = Field(0)
    # Fades at the very start and end of the final video.
    fade_in: float = Field(0.0, ge=0.0, le=10.0)
    fade_out: float = Field(0.0, ge=0.0, le=10.0)
    fade_color: str = "black"

    @field_validator("rotate")
    @classmethod
    def _check_rotate(cls, v: int) -> int:
        return int(round(v / 90.0) * 90) % 360


class TextSettings(BaseModel):
    enabled: bool = False
    # "title" draws one static line; "captions" burns an uploaded .srt/.vtt.
    kind: Literal["title", "captions"] = "title"
    text: str = ""
    font_file: str = "C:/Windows/Fonts/arial.ttf"
    font_size: int = Field(48, ge=8, le=400)
    font_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    outline_width: int = Field(3, ge=0, le=20)
    shadow: int = Field(2, ge=0, le=20)
    box: bool = False
    box_color: str = "#000000@0.55"
    align: TextAlign = "bottom"
    margin_v: int = Field(60, ge=0, le=2000)
    margin_h: int = Field(60, ge=0, le=2000)
    # Only show between these times; 0 = whole video.
    start: float = Field(0.0, ge=0.0)
    duration: float = Field(3.0, ge=0.1, le=600.0)
    # captions only
    subtitle_file: str = ""
    # Burn a repeated watermark image (png/jpg) instead of text.
    watermark_file: str = ""
    watermark_position: TextAlign = "topRight"
    watermark_width_pct: float = Field(0.0, ge=0.0, le=100.0)
    watermark_opacity: float = Field(0.6, ge=0.0, le=1.0)


class TransitionSettings(BaseModel):
    kind: TransitionKind = "cut"
    xfade: XfadeTransition = "fade"
    # Overlap length, seconds. Capped to half the shortest segment.
    duration: float = Field(0.3, ge=0.02, le=3.0)


class ExportSettings(BaseModel):
    # "source" keeps the input codec, otherwise H.264 in MP4.
    video_codec: Literal["h264", "h265", "vp9", "gif"] = "h264"
    crf: int = Field(20, ge=0, le=51)
    preset: Literal[
        "ultrafast", "superfast", "veryfast", "faster", "fast",
        "medium", "slow", "slower",
    ] = "veryfast"
    audio_codec: Literal["aac", "mp3", "none"] = "aac"
    audio_bitrate: str = "192k"
    # Optional hard size cap, e.g. "1080p", "720p", "480p", "360p".
    resolution: Literal["source", "2160p", "1440p", "1080p", "720p", "480p", "360p"] = "source"
    filename: str = ""


class EditRequest(BaseModel):
    asset_id: str
    auto_cut: AutoCutSettings = Field(default_factory=AutoCutSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    video: VideoSettings = Field(default_factory=VideoSettings)
    text: TextSettings = Field(default_factory=TextSettings)
    transition: TransitionSettings = Field(default_factory=TransitionSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)
    # Optional manual override; when present the VAD result is ignored.
    manual_segments: list[list[float]] | None = None


class AnalyzeRequest(BaseModel):
    asset_id: str
    vad: VadSettings = Field(default_factory=VadSettings)


class SegmentOut(BaseModel):
    start: float
    end: float
    duration: float


class AnalyzeResponse(BaseModel):
    asset_id: str
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    has_video: bool
    silence_regions: list[SegmentOut]
    keep_segments: list[SegmentOut]
    stats: dict
    removed_seconds: float
    removed_percent: float
