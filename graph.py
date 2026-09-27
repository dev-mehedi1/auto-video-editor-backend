"""Filtergraph construction.

Design rules, all verified against real ffmpeg (see tests/test_graph.py):

1. `concat` refuses to run unless every input has identical width, height,
   frame rate, pixel format, sample format and channel layout. So each
   per-segment branch ends with the same normalisation chain.
2. `fps` must come BEFORE `zoompan`, never after. zoompan leaves the stream
   on the source time base; a trailing `fps` filter then decides the frames
   are ~512x too close together and duplicates every single one. A 5 second
   clip came out as 1538 seconds before this was fixed.
3. Filter chains are joined with `;`. Using `,` glues two chains together
   into one nonsense chain.
4. Any `zoompan` option value containing a comma must be single quoted, or
   the comma is read as an option separator.
5. `xfade` needs an explicit offset per step: the running total length minus
   the overlaps already consumed.
"""
from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

try:
    from .ffmpeg import MediaInfo
    from .models import (
        AudioSettings,
        EditRequest,
        Overlay,
        TextSettings,
        VideoSettings,
        XfadeTransition,
        ZoomStyle,
    )
except ImportError:
    from ffmpeg import MediaInfo
    from models import (
        AudioSettings,
        EditRequest,
        Overlay,
        TextSettings,
        VideoSettings,
        XfadeTransition,
        ZoomStyle,
    )


ASPECTS: dict[str, tuple[int, int]] = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
    "21:9": (2560, 1080),
    "4:3": (1440, 1080),
}

# Where uploaded overlay photos live. Kept in step with main.IMAGE_ROOT.
IMAGE_ROOT = Path(__file__).resolve().parent / "storage" / "images"

RESOLUTIONS: dict[str, int] = {
    "2160p": 2160, "1440p": 1440, "1080p": 1080,
    "720p": 720, "480p": 480, "360p": 360,
}

_XFADE: dict[str, XfadeTransition] = {
    "fade": "fade", "fadeblack": "fadeblack", "fadewhite": "fadewhite",
    "slideleft": "slideleft", "slideright": "slideright",
    "slideup": "slideup", "slidedown": "slidedown",
    "circleopen": "circleopen", "circleclose": "circleclose",
    "dissolve": "dissolve", "pixelize": "pixelize",
    "wipeleft": "wipeleft", "wiperight": "wiperight",
    "radial": "radial", "smoothleft": "smoothleft",
}

# Each entry is an (x, y) expression pair. They are always emitted as
# separately quoted options: an unquoted expression containing ":" is
# indistinguishable from the option separator and ffmpeg dies with
# "No option name near '...'".
_TEXT_POS: dict[str, tuple[str, str]] = {
    "topLeft": ("(w-text_w)/2", "h-text_h-{mv}"),
    "top": ("(w-text_w)/2", "h-text_h-{mv}"),
    "topRight": ("w-text_w-{mh}", "h-text_h-{mv}"),
    "left": ("{mh}", "(h-text_h)/2"),
    "center": ("(w-text_w)/2", "(h-text_h)/2"),
    "right": ("w-text_w-{mh}", "(h-text_h)/2"),
    "bottomLeft": ("{mh}", "h-text_h-{mv}"),
    "bottom": ("(w-text_w)/2", "h-text_h-{mv}"),
    "bottomRight": ("w-text_w-{mh}", "h-text_h-{mv}"),
}

_WM_POS: dict[str, tuple[str, str]] = {
    "topLeft": ("20", "20"),
    "top": ("(W-w)/2", "20"),
    "topRight": ("W-w-20", "20"),
    "left": ("20", "(H-h)/2"),
    "center": ("(W-w)/2", "(H-h)/2"),
    "right": ("W-w-20", "(H-h)/2"),
    "bottomLeft": ("20", "H-h-20"),
    "bottom": ("(W-w)/2", "H-h-20"),
    "bottomRight": ("W-w-20", "H-h-20"),
}

_HEX = re.compile(r"^#?([0-9a-fA-F]{6})([0-9a-fA-F]{2})?$")


# --------------------------------------------------------------------------
# Escaping
# --------------------------------------------------------------------------

def esc_path(path: str | Path) -> str:
    """Escape a filesystem path for a filter option.

    Windows drive letters, backslashes and the filtergraph metacharacters
    all need escaping or the parser eats them.
    """
    p = str(path).replace("\\", "/")
    for ch in ("\\", ":", "'", "[", "]", ",", ";"):
        p = p.replace(ch, "\\" + ch)
    return p


def esc_text(text: str) -> str:
    """Escape a literal string for a filter option value."""
    out = text.replace("\\", "\\\\")
    for ch in ("'", ":", "%", "[", "]", ",", ";"):
        out = out.replace(ch, "\\" + ch)
    return out


def color(value: str, default: str = "black") -> str:
    """Normalise #RRGGBB / RRGGBBAA into something ffmpeg accepts."""
    v = (value or "").strip()
    if v in {"black", "white", "red", "green", "blue", "yellow", "gray", "none"}:
        return v
    m = _HEX.match(v)
    if m:
        return f"0x{m.group(1)}{m.group(2) or ''}"
    return default


@dataclass
class Plan:
    """A fully resolved render recipe."""

    segments: list[tuple[float, float]]
    speeds: list[float]
    filter_complex: str
    maps: list[str]
    output_duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    has_output_audio: bool = False
    # Label of the finished video stream, so the encoder can chain onto it
    # (GIF needs a split + palettegen pass).
    video_label: str = "[vpost]"
    warnings: list[str] = field(default_factory=list)
    extra_input_args: list[str] = field(default_factory=list)
    # Temp files created while building (drawtext text files) to remove once
    # ffmpeg has finished.
    cleanup_paths: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

@dataclass
class Geometry:
    """Two sizes: what every branch is normalised to, and what ships."""

    pre_w: int
    pre_h: int
    out_w: int
    out_h: int
    # Applied once after concat, e.g. "pad=...:..." or "crop=...:...".
    fit_filter: str | None = None


def _even(n: float) -> int:
    n = int(round(n))
    return max(2, n - (n % 2))


def _cap(resolution: str, w: int, h: int) -> tuple[int, int]:
    if resolution in RESOLUTIONS and h > RESOLUTIONS[resolution]:
        ratio = RESOLUTIONS[resolution] / h
        return _even(w * ratio), _even(RESOLUTIONS[resolution])
    return w, h


def geometry(
    info: MediaInfo, video: VideoSettings, export_resolution: str
) -> Geometry:
    """Resolve frame geometry.

    Order matters: the delivery box is picked and size-capped first, then the
    content is fitted into *that* box. Capping afterwards would leave the pad
    or crop pointing at the pre-cap dimensions.
    """
    w = info.display_width or info.width or 1920
    h = info.display_height or info.height or 1080
    if video.rotate in (90, 270):
        w, h = h, w

    # 1. the delivery box
    if video.aspect == "source" or video.aspect not in ASPECTS:
        tw, th = w, h
    else:
        tw, th = ASPECTS[video.aspect]
        if video.fit == "stretch":
            tw, th = ASPECTS[video.aspect]

    tw, th = _cap(export_resolution, _even(tw), _even(th))
    if video.max_width and tw > video.max_width:
        ratio = video.max_width / tw
        tw, th = _even(tw * ratio), _even(th * ratio)
    if video.max_height and th > video.max_height:
        ratio = video.max_height / th
        tw, th = _even(tw * ratio), _even(th * ratio)
    tw, th = _even(tw), _even(th)

    # 2. how the content maps into it
    if video.aspect == "source" or video.aspect not in ASPECTS:
        pre_w, pre_h = _even(w), _even(h)
        out_w, out_h = tw, th
    elif video.fit == "stretch":
        pre_w, pre_h = tw, th
        out_w, out_h = tw, th
    elif video.fit == "fill":
        # Cover the box, then centre-crop. fill may upscale: that is the whole
        # point of asking to fill the frame.
        s = max(tw / w, th / h)
        pre_w, pre_h = _even(w * s), _even(h * s)
        out_w, out_h = tw, th
    else:
        # Fit: shrink to fit, never upscale, then letterbox onto the delivery
        # box so the result really is 9:16 / 1:1 etc. No pad when the content
        # already has that shape: letterboxing 640x360 up to 1920x1080 would
        # only waste bitrate.
        s = min(1.0, tw / w, th / h)
        pre_w, pre_h = _even(w * s), _even(h * s)
        out_w, out_h = tw, th
        # Compare the *source* aspect, not the scaled one. A 405px wide box
        # rounds down to 404, and testing 404/720 against 9/16 would look like
        # a different shape and suppress the letterbox that was asked for.
        if abs((w / h) - (tw / th)) < 0.01:
            out_w, out_h = pre_w, pre_h

    fit_filter = None
    if video.aspect != "source" and video.aspect in ASPECTS:
        if video.fit == "fill":
            x = max(0, (pre_w - out_w) // 2)
            y = max(0, (pre_h - out_h) // 2)
            fit_filter = f"crop={out_w}:{out_h}:{x}:{y}"
        elif video.fit != "stretch" and (pre_w, pre_h) != (out_w, out_h):
            fit_filter = (
                f"pad={out_w}:{out_h}:{(out_w - pre_w) // 2}:{(out_h - pre_h) // 2}"
                f":color={color(video.pad_color, 'black')}"
            )

    if (out_w, out_h) != (pre_w, pre_h) and fit_filter is None:
        # Capped without a pad/crop: rescale after the concat.
        fit_filter = f"scale={out_w}:{out_h}:flags=bicubic"

    return Geometry(pre_w, pre_h, out_w, out_h, fit_filter)



# --------------------------------------------------------------------------
# Per-branch chains
# --------------------------------------------------------------------------

def _zoompan(style: ZoomStyle, amount: float, out_w: int, out_h: int, fps: float) -> str | None:
    """A complete zoompan filter, or None when no punch-in is wanted."""
    peak = 1.0 + 0.30 * max(0.0, min(1.0, amount))
    if style == "none" or peak <= 1.001:
        return None

    size = f"{out_w}x{out_h}"
    if style == "punchIn":
        z = f"min(zoom+0.0012,{peak:.4f})"
        x, y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    elif style == "punchOut":
        z = "max(1.0,zoom-0.0012)"
        x, y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    elif style == "shake":
        # 'on' is the output frame counter; there is no wall clock in zoompan.
        z = f"min({peak:.4f},1.0+0.05*abs(sin(on*0.35)))"
        jitter = max(1, out_w // 60)
        x = f"iw/2-(iw/zoom/2)+{jitter}*sin(on*0.9)"
        y = "ih/2-(ih/zoom/2)"
    elif style == "left":
        z = f"min(zoom+0.0012,{peak:.4f})"
        x, y = "0", "ih/2-(ih/zoom/2)"
    elif style == "right":
        z = f"min(zoom+0.0012,{peak:.4f})"
        x, y = "iw-iw/zoom", "ih/2-(ih/zoom/2)"
    else:
        return None

    # x and y must be emitted as two separate options: a comma between them
    # would be parsed as the x/y option separator rather than as a separator
    # inside an expression.
    return (
        f"zoompan=z='{z}':d=1:x='{x}':y='{y}':s={size}:fps={fps:g}"
    )


def _video_branch(
    idx: int,
    start: float,
    end: float,
    out_w: int,
    out_h: int,
    fps: float,
    rotate: int,
    zoompan: str | None,
    speed: float,
) -> str:
    parts: list[str] = [
        f"trim=start={start:.3f}:end={end:.3f}",
        "setpts=PTS-STARTPTS",
    ]

    if abs(speed - 1.0) > 1e-4:
        parts.append(f"setpts=PTS/{speed:.6f}")

    if rotate == 90:
        parts.append("transpose=1")
    elif rotate == 180:
        parts.append("transpose=1,transpose=1")
    elif rotate == 270:
        parts.append("transpose=2")

    if zoompan:
        # fps BEFORE scale/zoompan -- see rule 2 in the module docstring.
        parts.append(f"fps={fps:g}")
        parts.append(f"scale={out_w}:{out_h}:flags=bicubic")
        parts.append("setsar=1")
        parts.append("format=yuv420p")
        parts.append(zoompan)
    else:
        parts.append(f"scale={out_w}:{out_h}:flags=bicubic")
        parts.append("setsar=1")
        parts.append(f"fps={fps:g}")
        parts.append("format=yuv420p")

    parts.append("settb=AVTB")
    return f"[0:v]{','.join(parts)}[v{idx}]"


def _audio_branch(
    idx: int,
    start: float,
    end: float,
    speed: float,
    audio: AudioSettings,
    cut_fade: float,
) -> str | None:
    if audio.mute:
        return None

    parts: list[str] = [
        f"atrim=start={start:.3f}:end={end:.3f}",
        "asetpts=PTS-STARTPTS",
    ]

    if abs(speed - 1.0) > 1e-4:
        parts.append(f"atempo={speed:.6f}")

    dur = (end - start) / max(1e-4, speed)

    if cut_fade > 0 and dur > cut_fade * 2 + 0.02:
        parts.append(f"afade=t=in:st=0:d={cut_fade:.3f}")
        parts.append(f"afade=t=out:st={dur - cut_fade:.3f}:d={cut_fade:.3f}")

    if audio.highpass_hz > 0:
        parts.append(f"highpass=f={audio.highpass_hz}")
    if audio.lowpass_hz > 0:
        parts.append(f"lowpass=f={audio.lowpass_hz}")
    if audio.denoise > 0:
        nr = int(round(min(97.0, audio.denoise * 97.0)))
        parts.append(f"afftdn=nr={nr}:nf=-25")
    if audio.deesser:
        parts.append("deesser=i=0.4:m=0.5:f=0.5")
    if audio.compress:
        parts.append(
            f"acompressor=threshold={audio.comp_threshold:.1f}dB"
            f":ratio={audio.comp_ratio:.2f}:attack=5:release=50:makeup=1"
        )
    if abs(audio.gain_db) > 1e-3:
        parts.append(f"volume={audio.gain_db:.2f}dB")

    # Pin the format so concat/xfade cannot reject a mismatch.
    parts.append("aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo")
    parts.append("asettb=AVTB")
    return f"[0:a]{','.join(parts)}[a{idx}]"


# --------------------------------------------------------------------------
# Post-processing chains
# --------------------------------------------------------------------------

def _colour_chain(video: VideoSettings) -> list[str]:
    out: list[str] = []
    needs_eq = (
        abs(video.brightness) > 1e-3
        or abs(video.contrast - 1.0) > 1e-3
        or abs(video.saturation - 1.0) > 1e-3
        or abs(video.gamma - 1.0) > 1e-3
    )
    if needs_eq:
        parts = [
            f"brightness={video.brightness:.3f}",
            f"contrast={video.contrast:.3f}",
            f"saturation={video.saturation:.3f}",
        ]
        if abs(video.gamma - 1.0) > 1e-3:
            # eq's gamma wants the exponent, so invert our perceptual dial.
            parts.append(f"gamma={1.0 / max(0.05, video.gamma):.4f}")
        out.append("eq=" + ":".join(parts))
    if video.sharpen > 0:
        out.append(f"unsharp=5:5:{min(2.0, video.sharpen) * 1.5:.3f}:5:5:0.0")
    if video.vignette > 0:
        angle = 3.14159265 / max(3.0, 5.0 - 4.0 * video.vignette)
        out.append(f"vignette=angle={angle:.4f}")
    return out


def _text_chain(text: TextSettings, out_w: int, out_h: int) -> tuple[list[str], list[str]]:
    """Return (filter chains, temp files to delete afterwards).

    The text goes through `textfile=` rather than `text=`: no amount of
    backslash-escaping reliably survives both the filtergraph parser and
    drawtext's own expansion parser for input like "a,b: 100% [x] 'q'".
    A temp file sidesteps the escaping problem entirely.
    """
    if not text.enabled or text.kind != "title" or not text.text.strip():
        return [], []
    if not Path(text.font_file).exists():
        return [], []

    size = max(10, int(round(text.font_size * out_w / 1920.0)))
    px, py = _TEXT_POS.get(text.align, _TEXT_POS["bottom"])
    px = px.format(mv=text.margin_v, mh=text.margin_h)
    py = py.format(mv=text.margin_v, mh=text.margin_h)

    handle, txt_path = tempfile.mkstemp(prefix="edittxt_", suffix=".txt")
    with os.fdopen(handle, "w", encoding="utf-8") as fh:
        fh.write(text.text)

    chain = [
        # The whole value must be quoted: an escaped colon alone is undone by
        # the graph parser before the AVOption splitter sees it, so "C:/..." is
        # still read as the start of a new option.
        f"fontfile='{esc_path(text.font_file)}'",
        f"textfile='{esc_path(txt_path)}'",
        "expansion=none",
        f"fontsize={size}",
        f"fontcolor={color(text.font_color, 'white')}",
    ]
    if text.outline_width > 0:
        chain.append(f"bordercolor={color(text.outline_color, 'black')}")
        chain.append(f"borderw={text.outline_width}")
    if text.shadow > 0:
        chain.append(f"shadowcolor=black@0.6:shadowx=0:shadowy={text.shadow}")
    if text.box:
        chain.append("box=1")
        chain.append(f"boxcolor={color(text.box_color, 'black@0.55')}")
        chain.append("boxborderw=14")
    else:
        chain.append("box=0")
    chain.append(f"x='{px}'")
    chain.append(f"y='{py}'")
    if text.start > 0 or text.duration > 0:
        # Commas inside between() must be escaped or they are read as filter
        # chain separators and ffmpeg goes looking for a filter named "0.000".
        end = text.start + text.duration
        chain.append(f"enable='between(t\\,{text.start:.3f}\\,{end:.3f})'")

    return ["drawtext=" + ":".join(chain)], [txt_path]


def _caption_chain(text: TextSettings, out_w: int, out_h: int) -> list[str]:
    if not text.enabled or text.kind != "captions" or not text.subtitle_file:
        return []
    if not Path(text.subtitle_file).exists():
        return []
    style = (
        f"FontName=Arial,FontSize={max(10, int(out_w / 26))},"
        f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        f"BorderStyle=3,Outline=2,Shadow=1,"
        f"MarginV={max(10, int(out_h * 0.07))},Alignment=2"
    )
    return [f"subtitles='{esc_path(text.subtitle_file)}':force_style='{style}'"]


# --------------------------------------------------------------------------
# Overlays
# --------------------------------------------------------------------------

def _overlay_box(overlay: Overlay, out_w: int, out_h: int) -> tuple[int, int]:
    """The box an overlay is fitted into, in output pixels."""
    return (
        max(2, int(round(overlay.width_pct * out_w))),
        max(2, int(round(overlay.height_pct * out_h))),
    )


def _overlay_enable(overlay: Overlay) -> str:
    """The between() window an overlay is visible for.

    Commas have to be escaped or the graph parser reads them as chain
    separators and goes looking for a filter named "3.000".
    """
    end = overlay.start + overlay.duration
    return f"between(t\\,{overlay.start:.3f}\\,{end:.3f})"


def _image_overlay_chain(
    overlay: Overlay, input_index: int, out_w: int, out_h: int, label: str
) -> str:
    """Build `[idx:v]<prepare>[label]` for one photo overlay.

    `force_original_aspect_ratio=decrease` fits the photo inside the box, which
    is exactly what `object-contain` does in the browser preview, so a photo
    dragged to a size lands in the same place in both.
    """
    box_w, box_h = _overlay_box(overlay, out_w, out_h)
    parts = [f"scale={box_w}:{box_h}:force_original_aspect_ratio=decrease:flags=bicubic"]

    if abs(overlay.rotation) > 0.01:
        rad = overlay.rotation * 3.141592653589793 / 180.0
        # c=none keeps the frame transparent, and rotw/roth grow the canvas to
        # fit the rotated corners so nothing is clipped away.
        parts.append(f"rotate={rad:.6f}:c=none:ow=rotw(ih):oh=roth(ih)")

    parts.append("format=rgba")
    if overlay.opacity < 0.999:
        parts.append(f"colorchannelmixer=aa={overlay.opacity:.3f}")

    return f"[{input_index}:v]{','.join(parts)}[{label}]"


def _text_overlay_chain(
    overlay: Overlay, out_w: int, out_h: int
) -> tuple[str, str] | None:
    """Build a drawtext filter for one text overlay, plus its temp file.

    Returns None when there is nothing drawable. Like the legacy title, the
    string goes through `textfile=` so no amount of escaping can be eaten by the
    filtergraph parser.
    """
    if not overlay.text.strip() or not Path(overlay.font_file).exists():
        return None

    size = max(10, int(round(overlay.font_size * out_w / 1920.0)))

    handle, txt_path = tempfile.mkstemp(prefix="ovtext_", suffix=".txt")
    with os.fdopen(handle, "w", encoding="utf-8") as fh:
        fh.write(overlay.text)

    chain = [
        f"fontfile='{esc_path(overlay.font_file)}'",
        f"textfile='{esc_path(txt_path)}'",
        "expansion=none",
        f"fontsize={size}",
        f"fontcolor={color(overlay.color, 'white')}",
    ]
    if overlay.outline_width > 0:
        chain.append(f"bordercolor={color(overlay.outline_color, 'black')}")
        chain.append(f"borderw={overlay.outline_width}")
    if overlay.box:
        chain.append("box=1")
        chain.append(f"boxcolor={color(overlay.box_color, 'black@0.55')}")
        chain.append("boxborderw=14")
    else:
        chain.append("box=0")
    # Centre-anchored, same convention as the browser preview.
    chain.append(f"x='{overlay.x:.5f}*w-text_w/2'")
    chain.append(f"y='{overlay.y:.5f}*h-text_h/2'")
    chain.append(f"enable='{_overlay_enable(overlay)}'")

    return "drawtext=" + ":".join(chain), txt_path


def _apply_overlays(
    chains: list[str],
    current: str,
    overlays: list[Overlay],
    out_w: int,
    out_h: int,
    output_duration: float,
    next_input_index: int,
    image_root: Path,
) -> tuple[str, int, list[str], list[str], list[str]]:
    """Stack every overlay onto the running video label.

    Returns (label, next_input_index, extra_input_args, temp_files, warnings).
    Each photo needs its own `-i`, so the input counter is threaded through and
    handed back for whatever runs next.
    """
    extra_inputs: list[str] = []
    temp_files: list[str] = []
    warnings: list[str] = []

    for n, overlay in enumerate(overlays):
        out = f"vovl{n}"

        if overlay.kind == "image":
            if not overlay.image_id or not overlay.image_id.isalnum():
                warnings.append(f"Overlay {n + 1} has no image id and was skipped.")
                continue
            matches = list(image_root.glob(f"{overlay.image_id}.*"))
            if not matches:
                warnings.append(
                    f"Overlay {n + 1} refers to an image that no longer exists."
                )
                continue

            # -loop 1 holds a still frame; -t stops the input so an endless
            # stream cannot keep ffmpeg alive after the video ends.
            extra_inputs += [
                "-loop", "1",
                "-t", f"{output_duration + 1.0:.3f}",
                "-i", str(matches[0]),
            ]
            label = f"ov{n}"
            chains.append(
                _image_overlay_chain(overlay, next_input_index, out_w, out_h, label)
            )
            # `overlay=` with an equals sign, not `overlay:`. ffmpeg splits a
            # filter's name from its arguments at the first `=`, so a colon
            # here makes it look for a filter literally named "overlay:x" and
            # it rejects the rest as an option list.
            chains.append(
                f"{current}[{label}]overlay"
                f"=x='({overlay.x:.5f}*main_w-overlay_w)/2'"
                f":y='({overlay.y:.5f}*main_h-overlay_h)/2'"
                f":format=auto:shortest=0"
                f":enable='{_overlay_enable(overlay)}'"
                f"[{out}]"
            )
            current = f"[{out}]"
            next_input_index += 1
            continue

        built = _text_overlay_chain(overlay, out_w, out_h)
        if built is None:
            if overlay.text.strip():
                warnings.append(
                    f"Text overlay {n + 1} needs a font file that does not exist, "
                    f"so it was skipped."
                )
            continue
        filter_str, txt_path = built
        temp_files.append(txt_path)
        chains.append(f"{current}{filter_str}[{out}]")
        current = f"[{out}]"

    return current, next_input_index, extra_inputs, temp_files, warnings


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def _speed_for(mode: str, value: float, idx: int, total: int) -> float:
    if total <= 1:
        return 1.0 if mode == "normal" else value
    frac = idx / (total - 1)
    if mode == "rampUp":
        return 1.0 + (value - 1.0) * frac
    if mode == "rampDown":
        return value + (1.0 - value) * frac
    if mode == "rampUpDown":
        mid = (total - 1) / 2.0
        return 1.0 + (value - 1.0) * (1.0 - abs(idx - mid) / mid) if mid else 1.0
    return value


def _xfade_offsets(durations: list[float], xdur: float) -> list[float]:
    """Offset for merging segment i (i >= 1).

    After i merges the accumulated length is sum(d[0..i-1]) - xdur*i, and the
    new segment has to start xdur before that so the blend has room.
    """
    offsets: list[float] = []
    acc = 0.0
    for i, d in enumerate(durations):
        if i == 0:
            acc = d
            continue
        offsets.append(max(0.0, acc - xdur))
        acc = acc + d - xdur
    return offsets


def build(
    req: EditRequest,
    info: MediaInfo,
    segments: list[tuple[float, float]],
    duration: float,
) -> Plan:
    """Turn an EditRequest plus a cut list into a concrete ffmpeg plan."""
    warnings: list[str] = []

    # A manual timeline drawn in the UI always wins over the detected cuts.
    if req.manual_segments:
        segments = [
            (float(pair[0]), float(pair[1])) for pair in req.manual_segments
        ]
    elif not req.auto_cut.enabled:
        # Auto-cut off means "keep everything", not "keep nothing".
        segments = [(0.0, duration)] if duration > 0 else list(segments)

    segments = clean_segments(segments, duration)
    if not segments:
        raise ValueError(
            "nothing left to keep - the whole video reads as silence. "
            "Lower the sensitivity or reduce the minimum silence length."
        )

    geo = geometry(info, req.video, req.export.resolution)
    fps = req.video.fps if req.video.fps > 0 else (info.fps or 30.0)
    fps = min(60.0, max(1.0, round(float(fps), 3)))
    has_audio = info.has_audio and not req.audio.mute
    # `mute` means the user wants no audio track at all, so it wins over the
    # "keep the container valid" behaviour below. GIF cannot carry audio in
    # any case.
    want_audio = (
        not req.audio.mute
        and req.export.audio_codec != "none"
        and req.export.video_codec != "gif"
    )

    total = len(segments)
    speeds = [
        _speed_for(req.auto_cut.speed, req.auto_cut.speed_value, i, total)
        for i in range(total)
    ]
    speeds = [max(0.25, min(4.0, s)) for s in speeds]

    chains: list[str] = []
    v_labels: list[str] = []
    a_labels: list[str] = []

    zoom_filter = _zoompan(
        req.auto_cut.zoom, req.auto_cut.zoom_amount, geo.pre_w, geo.pre_h, fps
    )
    every = max(1, req.auto_cut.zoom_every)

    for i, (start, end) in enumerate(segments):
        # Never punch in on the opening shot.
        use_zoom = zoom_filter if (zoom_filter and i > 0 and i % every == 0) else None
        chains.append(
            _video_branch(i, start, end, geo.pre_w, geo.pre_h, fps, req.video.rotate,
                          use_zoom, speeds[i])
        )
        v_labels.append(f"[v{i}]")

        achain = (
            _audio_branch(i, start, end, speeds[i], req.audio, req.audio.cut_fade)
            # A source with no audio stream has no [0:a] to reference, so we
            # must not emit the branch at all.
            if info.has_audio and want_audio
            else None
        )
        if achain:
            chains.append(achain)
            a_labels.append(f"[a{i}]")

    out_durations = [
        (e - s) / max(1e-4, sp) for (s, e), sp in zip(segments, speeds)
    ]

    # --- transitions ------------------------------------------------------
    tr = req.transition
    xdur = 0.0
    # "fade" and "xfade" both mean an overlapping cross-dissolve; the xfade
    # field picks the style. Only "cut" is a hard join.
    if tr.kind in ("fade", "xfade") and total > 1:
        xdur = max(0.02, min(tr.duration, min(out_durations) * 0.5))
        name = _XFADE.get(tr.xfade, "fade")
        offsets = _xfade_offsets(out_durations, xdur)
        prev = v_labels[0]
        for i in range(1, total):
            out = f"[vx{i}]"
            chains.append(
                f"{prev}{v_labels[i]}xfade=transition={name}"
                f":duration={xdur:.3f}:offset={offsets[i - 1]:.3f}{out}"
            )
            prev = out
        v_labels = [prev]

        if a_labels:
            aprev = a_labels[0]
            for i in range(1, len(a_labels)):
                out = f"[ax{i}]"
                chains.append(
                    f"{aprev}{a_labels[i]}acrossfade=d={xdur:.3f}:c1=tri:c2=tri{out}"
                )
                aprev = out
            a_labels = [aprev]
    else:
        chains.append(f"{''.join(v_labels)}concat=n={total}:v=1:a=0[vcat]")
        v_labels = ["[vcat]"]
        if a_labels:
            chains.append(f"{''.join(a_labels)}concat=n={total}:v=0:a=1[acat]")
            a_labels = ["[acat]"]

    output_duration = sum(out_durations) - (xdur * (total - 1) if xdur else 0.0)

    # --- watermark needs its own input ----------------------------------
    extra_inputs: list[str] = []
    temp_files: list[str] = []
    next_input_index = 1
    wm_file = req.text.watermark_file
    if wm_file and Path(wm_file).exists():
        extra_inputs = ["-i", wm_file]
        next_input_index = 2
        wm_x, wm_y = _WM_POS.get(req.text.watermark_position, _WM_POS["bottomRight"])
        wm_chain = ""
        if 0 < req.text.watermark_width_pct < 100:
            target_w = max(2, int(geo.out_w * req.text.watermark_width_pct / 100.0))
            wm_chain = f"scale={target_w}:-1:flags=bicubic,"
        wm_chain += (
            f"format=rgba,colorchannelmixer=aa={req.text.watermark_opacity:.2f}"
        )
        # One chain, no trailing separator: an extra ';' would leave an empty
        # filter behind and ffmpeg aborts with "No such filter: ''".
        chains.append(f"[1:v]{wm_chain}[wm]")
        chains.append(
            f"{v_labels[0]}[wm]overlay=x='{wm_x}':y='{wm_y}'"
            f":format=auto:shortest=0[vovl]"
        )
        v_labels = ["[vovl]"]

    # --- colour / captions / framing ------------------------------------
    post: list[str] = []
    if geo.fit_filter:
        post.append(geo.fit_filter)
    post += _colour_chain(req.video)
    post += _caption_chain(req.text, geo.out_w, geo.out_h)

    if post:
        chains.append(f"{v_labels[0]}{','.join(post)}[vbase]")
        v_labels = ["[vbase]"]

    # --- draggable overlays ---------------------------------------------
    # After the framing step, so x/y fractions address the delivery frame the
    # user actually sees, and before the fades, so a fade to black covers them.
    if req.overlays:
        (
            v_labels[0],
            next_input_index,
            overlay_inputs,
            overlay_temps,
            overlay_warnings,
        ) = _apply_overlays(
            chains,
            v_labels[0],
            req.overlays,
            geo.out_w,
            geo.out_h,
            output_duration,
            next_input_index,
            IMAGE_ROOT,
        )
        extra_inputs += overlay_inputs
        temp_files += overlay_temps
        warnings += overlay_warnings

    # --- fades and the legacy title -------------------------------------
    text_filters, temp_files_from_text = _text_chain(req.text, geo.out_w, geo.out_h)
    temp_files += temp_files_from_text

    post = []
    if req.video.fade_in > 0:
        post.append(
            f"fade=t=in:st=0:d={req.video.fade_in:.3f}:color={color(req.video.fade_color)}"
        )
    if req.video.fade_out > 0:
        tail = max(0.0, output_duration - req.video.fade_out)
        post.append(
            f"fade=t=out:st={tail:.3f}:d={req.video.fade_out:.3f}"
            f":color={color(req.video.fade_color)}"
        )
    post += text_filters

    if post:
        chains.append(f"{v_labels[0]}{','.join(post)}[vpost]")
        v_labels = ["[vpost]"]

    # --- audio master ----------------------------------------------------
    if a_labels:
        if req.audio.normalize:
            chains.append(
                f"{a_labels[0]}loudnorm=I={req.audio.loudness_target:.1f}"
                f":TP=-1.5:LRA=11[aout]"
            )
        else:
            chains.append(f"{a_labels[0]}alimiter=limit=0.97[aout]")
        a_labels = ["[aout]"]
    elif want_audio and not info.has_audio:
        # The source is genuinely silent (e.g. a screen recording) but the
        # user asked for an audio track, so lay down a silent one.
        chains.append(
            f"anullsrc=r=48000:cl=stereo,atrim=0:{output_duration:.3f},"
            f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[aout]"
        )
        a_labels = ["[aout]"]
        warnings.append("Source has no audio track, so a silent one was added.")

    maps = ["-map", v_labels[0]] + (["-map", a_labels[0]] if a_labels else [])

    return Plan(
        segments=segments,
        speeds=speeds,
        filter_complex=";".join(chains),
        maps=maps,
        output_duration=output_duration,
        width=geo.out_w,
        height=geo.out_h,
        fps=fps,
        has_audio=has_audio,
        has_output_audio=bool(a_labels),
        video_label=v_labels[0],
        warnings=warnings,
        extra_input_args=extra_inputs,
        cleanup_paths=temp_files,
    )


def cleanup(plan: Plan) -> None:
    """Remove temp files the plan created. Safe to call twice."""
    for path in plan.cleanup_paths:
        try:
            os.unlink(path)
        except OSError:
            pass
    plan.cleanup_paths.clear()


def clean_segments(
    segments: list[tuple[float, float]], duration: float, min_len: float = 0.02
) -> list[tuple[float, float]]:
    """Clamp to the file, sort, drop overlaps and merge touching ranges."""
    if duration <= 0:
        return [(round(s, 3), round(e, 3)) for s, e in segments if e > s]
    items = sorted(
        (max(0.0, float(s)), min(duration, float(e))) for s, e in segments
    )
    out: list[tuple[float, float]] = []
    for s, e in items:
        if e - s <= min_len:
            continue
        if out and s <= out[-1][1] + 1e-6:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((round(s, 3), round(e, 3)))
    return out
