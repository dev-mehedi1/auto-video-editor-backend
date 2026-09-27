"""FastAPI application.

Routes:
  POST   /api/upload          upload a source clip, get an asset id
  GET    /api/asset/{id}      metadata for an asset
  POST   /api/analyze         run the voice/silence analysis
  GET    /api/asset/{id}/waveform  peak data for the timeline
  POST   /api/render          queue a render, get a job id
  GET    /api/job/{id}        poll job status
  POST   /api/job/{id}/cancel stop a queued or running job
  GET    /api/job/{id}/file   download the finished file
  GET    /api/health          readiness + ffmpeg capabilities
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

try:
    from . import analysis, ffmpeg, graph, jobs
    from .models import AnalyzeResponse, EditRequest, SegmentOut
except ImportError:
    import analysis, ffmpeg, graph, jobs
    from models import AnalyzeResponse, EditRequest, SegmentOut

BASE_DIR = Path(__file__).resolve().parent
STORAGE = BASE_DIR / "storage"
UPLOADS = STORAGE / "uploads"
OUTPUTS = STORAGE / "outputs"
IMAGES = STORAGE / "images"

ALLOWED_SUFFIXES = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg",
    ".ts", ".wmv", ".flv", ".3gp", ".m2ts", ".mts", ".ogv", ".gif",
}

ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

MAX_UPLOAD_BYTES = 8 * 1024 * 1024 * 1024  # 8 GB
MAX_IMAGE_BYTES = 32 * 1024 * 1024  # 32 MB

app = FastAPI(title="Video Editor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    for folder in (UPLOADS, OUTPUTS, IMAGES):
        folder.mkdir(parents=True, exist_ok=True)
    jobs.manager.start(workers=2)


@app.on_event("shutdown")
def _shutdown() -> None:
    jobs.manager.shutdown()


# --------------------------------------------------------------------------
# Assets
# --------------------------------------------------------------------------

def _asset_path(asset_id: str) -> Path:
    """Resolve an asset id to a file, refusing anything path-like."""
    if not asset_id.isalnum():
        raise HTTPException(400, "bad asset id")
    matches = list(UPLOADS.glob(f"{asset_id}.*"))
    if not matches:
        raise HTTPException(404, "unknown asset")
    return matches[0]


class UploadResponse(BaseModel):
    asset_id: str
    filename: str
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    has_video: bool
    size: int


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile) -> UploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            400, f"unsupported file type '{suffix}'. Allowed: "
                 + ", ".join(sorted(ALLOWED_SUFFIXES))
        )

    asset_id = uuid.uuid4().hex[:16]
    target = UPLOADS / f"{asset_id}{suffix}"

    written = 0
    with target.open("wb") as fh:
        while chunk := await file.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                fh.close()
                target.unlink(missing_ok=True)
                raise HTTPException(413, "file is too large")
            fh.write(chunk)

    if written == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(400, "the uploaded file was empty")

    try:
        info = ffmpeg.media_info(target)
    except ffmpeg.FFmpegError as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(400, f"could not read that video: {exc}") from exc

    if not info.has_video:
        target.unlink(missing_ok=True)
        raise HTTPException(400, "that file has no video track")

    return UploadResponse(
        asset_id=asset_id,
        filename=file.filename or target.name,
        duration=info.duration,
        width=info.display_width or info.width,
        height=info.display_height or info.height,
        fps=info.fps,
        has_audio=info.has_audio,
        has_video=info.has_video,
        size=written,
    )


@app.get("/api/asset/{asset_id}")
def asset_info(asset_id: str) -> dict[str, Any]:
    path = _asset_path(asset_id)
    info = ffmpeg.media_info(path)
    return {
        "asset_id": asset_id,
        "filename": path.name,
        "duration": info.duration,
        "width": info.display_width or info.width,
        "height": info.display_height or info.height,
        "fps": info.fps,
        "has_audio": info.has_audio,
        "has_video": info.has_video,
        "video_codec": info.video_codec,
        "audio_codec": info.audio_codec,
        "rotation": info.rotation,
        "size": path.stat().st_size,
    }


@app.get("/api/asset/{asset_id}/file")
def asset_file(asset_id: str) -> FileResponse:
    path = _asset_path(asset_id)
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.get("/api/asset/{asset_id}/waveform")
def asset_waveform(asset_id: str, buckets: int = 900) -> dict[str, Any]:
    """Peak envelope for the timeline, cheap enough to call on every load."""
    path = _asset_path(asset_id)
    buckets = max(50, min(4000, buckets))
    pcm = ffmpeg.decode_pcm(path)
    if pcm.size == 0:
        return {"buckets": [], "peak": 0.0}
    per = max(1, len(pcm) // buckets)
    usable = (len(pcm) // per) * per
    frames = pcm[:usable].reshape(-1, per)
    peaks = np.abs(frames).max(axis=1)
    norm = float(peaks.max()) or 1.0
    return {
        "buckets": [round(float(v) / norm, 4) for v in peaks],
        "peak": norm,
    }


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

class AnalyzeBody(BaseModel):
    asset_id: str
    sensitivity: float = Field(0.5, ge=0.0, le=1.0)
    min_silence: float = Field(0.35, ge=0.05, le=10.0)
    padding: float = Field(0.06, ge=0.0, le=2.0)
    min_segment: float = Field(0.18, ge=0.02, le=10.0)
    hysteresis: float = Field(3.0, ge=0.0, le=20.0)
    threshold_db: float | None = Field(None, ge=-90.0, le=0.0)
    use_ffmpeg_detect: bool = False


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(body: AnalyzeBody) -> AnalyzeResponse:
    path = _asset_path(body.asset_id)
    info = ffmpeg.media_info(path)

    if not info.has_audio:
        # Nothing to detect: keep the whole clip and say so plainly.
        whole = [(0.0, info.duration)] if info.duration > 0 else []
        return AnalyzeResponse(
            asset_id=body.asset_id, duration=info.duration,
            width=info.display_width or info.width,
            height=info.display_height or info.height,
            fps=info.fps, has_audio=False, has_video=info.has_video,
            silence_regions=[], keep_segments=[SegmentOut(start=s, end=e, duration=e - s)
                                              for s, e in whole],
            stats={"note": "This file has no audio track."},
            removed_seconds=0.0, removed_percent=0.0,
        )

    params = analysis.VadParams(
        sensitivity=body.sensitivity, min_silence=body.min_silence,
        padding=body.padding, min_segment=body.min_segment,
        hysteresis=body.hysteresis, threshold_db=body.threshold_db,
        use_ffmpeg_detect=body.use_ffmpeg_detect,
    )
    silences, kept, stats = analysis.detect(
        path, params, info.duration, has_audio=info.has_audio
    )

    removed = sum(e - s for s, e in silences)
    return AnalyzeResponse(
        asset_id=body.asset_id,
        duration=info.duration,
        width=info.display_width or info.width,
        height=info.display_height or info.height,
        fps=info.fps,
        has_audio=info.has_audio,
        has_video=info.has_video,
        silence_regions=[SegmentOut(start=s, end=e, duration=e - s) for s, e in silences],
        keep_segments=[SegmentOut(start=s, end=e, duration=e - s) for s, e in kept],
        stats=analysis.stats_to_dict(stats),
        removed_seconds=round(removed, 3),
        removed_percent=round(100.0 * removed / info.duration, 2) if info.duration else 0.0,
    )


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------

@app.post("/api/render")
def render(body: EditRequest) -> dict[str, Any]:
    path = _asset_path(body.asset_id)
    info = ffmpeg.media_info(path)

    # Segments for the job: pre-computed so the worker starts rendering
    # immediately. If the user changes a knob before clicking export the worker
    # detects the stale key and re-analyses.
    segments: list[tuple[float, float]] = []
    vad = body.auto_cut.vad
    if body.auto_cut.enabled and not body.manual_segments:
        _silences, segments, _stats = analysis.detect(
            path,
            analysis.VadParams(
                sensitivity=vad.sensitivity, min_silence=vad.min_silence,
                padding=vad.padding, min_segment=vad.min_segment,
                hysteresis=vad.hysteresis, threshold_db=vad.threshold_db,
                use_ffmpeg_detect=vad.use_ffmpeg_detect,
            ),
            info.duration,
            has_audio=info.has_audio,
        )

    job = jobs.manager.submit(
        asset_id=body.asset_id,
        source=path,
        request=body,
        detected_segments=segments,
        duration=info.duration,
        out_dir=OUTPUTS,
        vad_key=jobs.vad_key_of(vad),
    )
    return {"job_id": job.id, "status": job.status}


@app.get("/api/job/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    job = jobs.manager.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job.public()


@app.post("/api/job/{job_id}/cancel")
def job_cancel(job_id: str) -> dict[str, Any]:
    if jobs.manager.get(job_id) is None:
        raise HTTPException(404, "unknown job")
    if not jobs.manager.cancel(job_id):
        raise HTTPException(409, "job is not cancellable")
    return {"ok": True}


@app.get("/api/job/{job_id}/file")
def job_file(job_id: str) -> FileResponse:
    job = jobs.manager.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    if job.status != "done" or not job.output_path:
        raise HTTPException(409, f"job is {job.status}, not done")
    path = Path(job.output_path)
    if not path.exists():
        raise HTTPException(410, "the output file is gone")
    return FileResponse(path, filename=job.output_name)


# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict[str, Any]:
    caps = {}
    for name in ("libx264", "libx265", "libvpx-vp9", "aac", "libmp3lame", "libopus"):
        caps[name] = ffmpeg.has_encoder(name)
    return {
        "ok": True,
        "ffmpeg": ffmpeg.FFMPEG,
        "encoders": caps,
        "aspects": sorted(graph.ASPECTS),
        "xfades": sorted(graph._XFADE),
    }


@app.delete("/api/asset/{asset_id}")
def delete_asset(asset_id: str) -> JSONResponse:
    path = _asset_path(asset_id)
    try:
        path.unlink()
    except OSError as exc:
        raise HTTPException(500, str(exc)) from exc
    return JSONResponse({"ok": True})


# --------------------------------------------------------------------------
# Images (overlays)
# --------------------------------------------------------------------------

def _image_path(image_id: str) -> Path:
    """Resolve an image id to a file, refusing anything path-like."""
    if not image_id.isalnum():
        raise HTTPException(400, "bad image id")
    matches = list(IMAGES.glob(f"{image_id}.*"))
    if not matches:
        raise HTTPException(404, "unknown image")
    return matches[0]


class ImageResponse(BaseModel):
    image_id: str
    filename: str
    width: int
    height: int
    size: int


@app.post("/api/upload-image", response_model=ImageResponse)
async def upload_image(file: UploadFile) -> ImageResponse:
    """Store a still image so a render can reference it by id.

    The browser keeps using its own object URL for the live preview; this copy
    only exists so FFmpeg has a path to read during the export.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise HTTPException(
            400, f"unsupported image type '{suffix}'. Allowed: "
                 + ", ".join(sorted(ALLOWED_IMAGE_SUFFIXES))
        )

    image_id = uuid.uuid4().hex[:16]
    target = IMAGES / f"{image_id}{suffix}"

    written = 0
    with target.open("wb") as fh:
        while chunk := await file.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_IMAGE_BYTES:
                fh.close()
                target.unlink(missing_ok=True)
                raise HTTPException(413, "that image is too large (32 MB max)")
            fh.write(chunk)

    if written == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(400, "the uploaded file was empty")

    try:
        info = ffmpeg.media_info(target)
    except ffmpeg.FFmpegError as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(400, f"could not read that image: {exc}") from exc

    width = info.display_width or info.width
    height = info.display_height or info.height
    if not width or not height:
        target.unlink(missing_ok=True)
        raise HTTPException(400, "that file is not a readable image")

    # The stored name, not the one the browser sent, so this agrees with
    # `GET /api/image/{id}` and with what `/file` actually serves. The UI keeps
    # the original name in its own layer list.
    return ImageResponse(
        image_id=image_id,
        filename=target.name,
        width=width,
        height=height,
        size=written,
    )


@app.get("/api/image/{image_id}")
def image_info(image_id: str) -> dict[str, Any]:
    path = _image_path(image_id)
    info = ffmpeg.media_info(path)
    return {
        "image_id": image_id,
        "filename": path.name,
        "width": info.display_width or info.width,
        "height": info.display_height or info.height,
        "size": path.stat().st_size,
    }


@app.get("/api/image/{image_id}/file")
def image_file(image_id: str) -> FileResponse:
    path = _image_path(image_id)
    return FileResponse(path, filename=path.name)


@app.delete("/api/image/{image_id}")
def delete_image(image_id: str) -> JSONResponse:
    path = _image_path(image_id)
    try:
        path.unlink()
    except OSError as exc:
        raise HTTPException(500, str(exc)) from exc
    return JSONResponse({"ok": True})
