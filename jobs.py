"""In-process job queue for render jobs.

One render at a time per worker thread. Renders are CPU-bound ffmpeg
processes, so there is no point running many concurrently; the queue keeps
the API responsive and lets the UI show a meaningful position in line.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from . import analysis, encoder, ffmpeg, graph
from .models import EditRequest

JobStatus = Literal["queued", "analyzing", "rendering", "done", "error", "cancelled"]


@dataclass
class Job:
    id: str
    asset_id: str
    status: JobStatus = "queued"
    progress: float = 0.0
    stage: str = ""
    message: str = ""
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    output_name: str = ""
    output_path: str = ""
    output_size: int = 0
    output_duration: float = 0.0
    width: int = 0
    height: int = 0
    warnings: list[str] = field(default_factory=list)
    error: str = ""
    # Kept so the status endpoint can report what is about to happen.
    request: EditRequest | None = None
    # VAD knob values that produced `segments`, so a re-render can skip the
    # analysis when nothing changed.
    vad_key: tuple | None = None

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "asset_id": self.asset_id,
            "status": self.status,
            "progress": round(self.progress, 4),
            "stage": self.stage,
            "message": self.message,
            "output_name": self.output_name,
            "output_size": self.output_size,
            "output_duration": round(self.output_duration, 3),
            "width": self.width,
            "height": self.height,
            "warnings": self.warnings,
            "error": self.error,
            "elapsed": round(
                ((self.finished or time.time()) - self.created), 2
            ),
        }


class JobManager:
    """Background render queue with cancellation."""

    def __init__(self, max_jobs: int = 200) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._cancels: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._max_jobs = max_jobs
        self._workers: list[threading.Thread] = []
        self._started = False
    # -- lifecycle --------------------------------------------------------

    def start(self, workers: int = 2) -> None:
        if self._started:
            return
        self._started = True
        for i in range(workers):
            t = threading.Thread(target=self._worker, name=f"render-{i}", daemon=True)
            t.start()
            self._workers.append(t)

    def shutdown(self) -> None:
        with self._lock:
            for event in self._cancels.values():
                event.set()
        for _ in self._workers:
            self._queue.put(None)
        self._started = False

    # -- submission -------------------------------------------------------

    def submit(
        self,
        asset_id: str,
        source: Path,
        request: EditRequest,
        detected_segments: list[tuple[float, float]],
        duration: float,
        out_dir: Path,
        vad_key: tuple | None = None,
    ) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], asset_id=asset_id, request=request)
        # Only claim the key if we were actually handed matching segments.
        # Otherwise leave it None so the worker recomputes instead of trusting
        # whatever the caller happened to pass.
        job.vad_key = vad_key if detected_segments else None
        cancel = threading.Event()

        # Be forgiving about lifespan: if nothing started the workers (a bare
        # TestClient, or an app mounted without the startup hook), the job
        # would otherwise sit in the queue forever.
        if not self._started:
            self.start()

        with self._lock:
            self._jobs[job.id] = job
            self._cancels[job.id] = cancel
            self._order.append(job.id)
            self._evict_old()

        payload = (job.id, source, request, detected_segments, duration, out_dir)
        self._queue.put(("job", payload))
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.status in {"done", "error", "cancelled"}:
            return False
        with self._lock:
            event = self._cancels.get(job_id)
        if event is not None:
            event.set()
        if job.status == "queued":
            # Never started, so mark it now rather than waiting for the worker.
            job.status = "cancelled"
            job.finished = time.time()
            job.message = "Cancelled before it started."
        return True

    def _evict_old(self) -> None:
        while len(self._order) > self._max_jobs:
            old = self._order.pop(0)
            job = self._jobs.get(old)
            if job and job.status in {"queued", "rendering", "analyzing"}:
                self._order.append(old)
                break
            self._jobs.pop(old, None)
            self._cancels.pop(old, None)

    # -- worker -----------------------------------------------------------

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            _, payload = item
            job_id, source, request, segments, duration, out_dir = payload
            try:
                self._run(job_id, source, request, segments, duration, out_dir)
            except Exception:  # keep the worker alive
                traceback.print_exc()
                job = self.get(job_id)
                if job and job.status not in {"cancelled"}:
                    job.status = "error"
                    job.error = traceback.format_exc(limit=3)
                    job.finished = time.time()

    def _run(
        self,
        job_id: str,
        source: Path,
        request: EditRequest,
        segments: list[tuple[float, float]],
        duration: float,
        out_dir: Path,
    ) -> None:
        job = self.get(job_id)
        if job is None:
            return
        with self._lock:
            event = self._cancels.get(job_id) or threading.Event()
        if event.is_set():
            return

        job.started = time.time()
        job.stage = "planning"
        job.message = "Working out the edit..."

        info = ffmpeg.media_info(source)

        # Auto-cut is (re)computed here when the user changed the VAD knobs
        # since the last analysis, so the render always matches the sliders.
        vad = request.auto_cut.vad
        if request.auto_cut.enabled and not request.manual_segments:
            if job.vad_key != vad_key_of(vad):
                job.stage = "analyzing"
                job.message = "Listening for silence..."
                _silences, segments, _stats = analysis.detect(
                    source,
                    analysis.VadParams(
                        sensitivity=vad.sensitivity,
                        min_silence=vad.min_silence,
                        padding=vad.padding,
                        min_segment=vad.min_segment,
                        hysteresis=vad.hysteresis,
                        threshold_db=vad.threshold_db,
                        use_ffmpeg_detect=vad.use_ffmpeg_detect,
                    ),
                    duration,
                    has_audio=info.has_audio,
                )
                job.vad_key = vad_key_of(vad)
            if request.auto_cut.bridge_gap > 0:
                segments = analysis.merge_segments(
                    segments, request.auto_cut.bridge_gap, duration
                )

        if event.is_set():
            return

        try:
            plan = graph.build(request, info, segments, duration)
        except ValueError as exc:
            job.status = "error"
            job.error = str(exc)
            job.message = str(exc)
            job.finished = time.time()
            return

        job.warnings = list(plan.warnings)
        job.status = "rendering"
        job.stage = "rendering"
        job.output_duration = plan.output_duration
        job.width, job.height = plan.width, plan.height
        job.message = f"Rendering {plan.width}x{plan.height}..."

        out_dir.mkdir(parents=True, exist_ok=True)
        ext = encoder.VIDEO_CODECS.get(
            request.export.video_codec, encoder.VIDEO_CODECS["h264"]
        )[1]
        name = _safe_filename(request.export.filename) or f"{source.stem}-edited.{ext}"
        if not name.lower().endswith(f".{ext}"):
            name = f"{name}.{ext}"
        out_path = out_dir / name

        try:
            encoder.render(
                request, plan, source, out_path,
                on_progress=lambda p: setattr(job, "progress", float(p)),
                cancel=event,
            )
        except ffmpeg.Cancelled:
            graph.cleanup(plan)
            _unlink(out_path)
            job.status = "cancelled"
            job.message = "Cancelled."
            job.finished = time.time()
            return
        except ffmpeg.FFmpegError as exc:
            graph.cleanup(plan)
            _unlink(out_path)
            job.status = "error"
            job.error = str(exc)
            job.message = "Render failed."
            job.finished = time.time()
            return

        job.status = "done"
        job.progress = 1.0
        job.stage = "done"
        job.message = "Done."
        job.finished = time.time()
        job.output_name = out_path.name
        job.output_path = str(out_path)
        try:
            job.output_size = out_path.stat().st_size
        except OSError:
            job.output_size = 0


def vad_key_of(vad) -> tuple:
    """The knobs that change the cut list, for cache invalidation.

    use_ffmpeg_detect belongs here: flipping the detector while leaving the
    dials alone produces a completely different cut list.
    """
    return (
        vad.sensitivity, vad.min_silence, vad.padding,
        vad.min_segment, vad.hysteresis, vad.threshold_db,
        bool(getattr(vad, "use_ffmpeg_detect", False)),
    )


def _safe_filename(name: str) -> str:
    """Strip anything that could escape the output directory."""
    if not name:
        return ""
    cleaned = "".join(
        c for c in name if c.isalnum() or c in " -_.()[]"
    ).strip(" .")
    return cleaned[:120]


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


manager = JobManager()
