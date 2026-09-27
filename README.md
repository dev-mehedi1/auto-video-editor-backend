# Backend — Video Editor API

FastAPI + native FFmpeg. This service does the real work: it stores uploaded
clips, runs silence detection, builds an FFmpeg filtergraph, and renders the
final video. No AI, no cloud — everything runs locally on your machine.

The React app in `../Frontend` is the only client. It talks to this service
through a Vite proxy, so it never needs CORS in normal development.

---

## Requirements

| Tool | Version tested | Notes |
|------|----------------|-------|
| Python | 3.13.5 | 3.10+ should work |
| FFmpeg | 8.1.2 | must be on `PATH`, with `ffprobe` next to it |
| Node.js | not needed | only the frontend uses it |

Check FFmpeg before anything else:

```powershell
ffmpeg -version
ffprobe -version
```

If either command is not recognised, install FFmpeg and add its `bin`
directory to your `PATH`, then open a **new** terminal. Confirm with
`/api/health` later — it reports the resolved binary path and which encoders
are available.

---

## Setup

Create a virtual environment. This matters: a global Python install on many
machines already has unrelated `pandas-ta` / `langchain-community` packages
that break `numpy` imports.

Run these from inside `Backend/`:

```powershell
cd "F:\Mehedi\Experimental Project\Editing\Backend"

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks the activation script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

You can skip activation entirely and just prefix commands with
`.\.venv\Scripts\python.exe`.

---

## Running the server

From inside `Backend/`:

```powershell
python run.py
```

From the repository root, the equivalent is:

```powershell
python -m Backend
```

Both start the API on <http://127.0.0.1:8000> with hot reload enabled:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [18120] using WatchFiles
INFO:     Started server process [18121]
```

Stop it with `Ctrl+C`. The reloader restarts the server whenever you edit a
`.py` file in `Backend/`.

To pick a different port without editing anything:

```powershell
$env:PORT = "9000"
python run.py
```

### Why not `uvicorn Backend.main:app`?

`Backend` is a Python package (`Backend/__init__.py` exists) and `main.py` uses
relative imports such as `from . import analysis`. That only resolves when the
**parent** directory is on `sys.path`. So from the repository root you can use
the raw uvicorn command, but from inside `Backend/` the `Backend` package is
not importable and you get `ModuleNotFoundError: No module named 'Backend'`.

`run.py` and `__main__.py` exist to paper over that: they add the repository
root to `sys.path` for you.

```powershell
# only valid from the repository root:
python -m uvicorn Backend.main:app --reload --port 8000
```

The service binds to `127.0.0.1:8000`, which is exactly where the frontend
proxy points (`../Frontend/vite.config.ts`). If you change the port here, you
must change `server.proxy['/api'].target` there too.

### Verify it is up

```powershell
curl.exe http://127.0.0.1:8000/api/health
```

Or in the browser: <http://127.0.0.1:8000/api/health>
Interactive API docs: <http://127.0.0.1:8000/docs>

A healthy response:

```json
{
  "ok": true,
  "ffmpeg": "ffmpeg",
  "encoders": {
    "libx264": true, "libx265": true, "libvpx-vp9": true,
    "aac": true, "libmp3lame": true, "libopus": true
  },
  "aspects": ["16:9", "1:1", "21:9", "4:3", "4:5", "9:16"],
  "xfades": ["circleclose", "circleopen", "dissolve", "fade", "..."]
}
```

`encoders` should be all `true`. A `false` means your FFmpeg build lacks that
codec — install a full build rather than a minimal one.

---

## Tests

All three suites must run from the **repository root** (one level above
`Backend/`), because the test modules use relative imports such as
`from ..main import app`:

```powershell
cd "F:\Mehedi\Experimental Project\Editing"

# 41 tests — filtergraph construction
python -m unittest Backend.tests.test_graph

# 19 tests — HTTP routes via FastAPI TestClient
python -m unittest Backend.tests.test_api

# 34 checks — silence detection against synthetic audio
python -m Backend.tests.test_analysis
```

`test_analysis.py` is a standalone script (it prints `PASS`/`FAIL` lines and
exits non-zero on failure), so run it with `python -m`, not `unittest`.

`storage/` is git-ignored, and the API tests write throwaway clips into it, so
leftovers are expected and safe to delete.

---

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/health` | readiness, ffmpeg path, encoder support |
| `POST` | `/api/upload` | upload a source clip, returns an `asset_id` |
| `GET` | `/api/asset/{id}` | metadata: duration, size, fps, codecs, rotation |
| `GET` | `/api/asset/{id}/file` | download the original upload |
| `GET` | `/api/asset/{id}/waveform` | peak envelope for the timeline |
| `POST` | `/api/analyze` | silence detection, returns keep/drop segments |
| `POST` | `/api/render` | queue a render, returns a `job_id` |
| `GET` | `/api/job/{id}` | poll job status and progress |
| `POST` | `/api/job/{id}/cancel` | cancel a queued or running job |
| `GET` | `/api/job/{id}/file` | download the finished render |
| `DELETE` | `/api/asset/{id}` | delete an upload |

Renders are asynchronous: `POST /api/render` returns immediately with a
`job_id`, then you poll `/api/job/{id}` until `status` is `done`, and fetch
the result from `/api/job/{id}/file`.

CORS is locked to `http://localhost:5173` and `http://127.0.0.1:5173` (the
Vite dev server). Change `allow_origins` in `main.py` if you serve the
frontend from somewhere else.

---

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `FFMPEG_BINARY` | `ffmpeg` | full path to the ffmpeg executable |
| `FFPROBE_BINARY` | `ffprobe` | full path to the ffprobe executable |

If FFmpeg is not on your `PATH`:

```powershell
$env:FFMPEG_BINARY  = "C:\ffmpeg\bin\ffmpeg.exe"
$env:FFPROBE_BINARY = "C:\ffmpeg\bin\ffprobe.exe"
python -m uvicorn Backend.main:app --port 8000
```

Uploads are capped at 8 GB. Accepted containers: `.mp4 .mov .mkv .avi .webm
.m4v .mpg .mpeg .ts .wmv .flv .3gp .m2ts .mts .ogv .gif`.

---

## Storage

Uploads and renders live in `Backend/storage/`:

```
Backend/storage/
  uploads/   originals, named <asset_id>.<ext>
  outputs/   renders, named <asset_id>-edited.<ext>
```

Both folders are created automatically on startup. The directory is
git-ignored, so nothing here gets committed. To reset all local data, stop the
server and empty those two folders.

---

## Layout

```
Backend/
  main.py       FastAPI app, routes, CORS
  models.py     Pydantic request/response schemas
  analysis.py   silence detection (adaptive VAD + ffmpeg silencedetect)
  graph.py      builds the FFmpeg filtergraph
  encoder.py    codec / container / quality command assembly
  jobs.py       render queue, worker threads, progress, cancellation
  ffmpeg.py     low-level ffmpeg/ffprobe process helpers
  run.py        start the server: `python run.py`
  __main__.py   start the server: `python -m Backend` (from the root)
  tests/        graph, analysis, and API suites
```

Audio for analysis is decoded to 16 kHz mono through FFmpeg's own resampler, so
no `soundfile` or `librosa` dependency is needed.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'Backend'`** — you ran
`uvicorn Backend.main:app` from inside `Backend/`, so the package is not
importable. Use `python run.py` instead, or `cd ..` and use
`python -m Backend`.

**`error: relative import with no known parent package`** — same cause. The app
has to be loaded as `Backend.main:app` with the repository root on `sys.path`.

**Health check reports `ffmpeg` not found** — FFmpeg is missing or not on
`PATH`. Set `FFMPEG_BINARY` / `FFPROBE_BINARY` as shown above.

**`429` on upload** — a browser is still uploading. The frontend shows a
progress bar; wait for it to finish.

**`409` when cancelling** — the job already finished. `404` means the job id is
unknown to this server, which is normal after a restart since jobs are held in
memory only.

**Frontend says the backend is offline** — the backend runs on `8000` by
default. If you started it on another port, update the proxy target in
`../Frontend/vite.config.ts`.
