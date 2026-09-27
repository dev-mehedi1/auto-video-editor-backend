"""End-to-end API tests.

These run the real FastAPI app with the real ffmpeg binary and a real
synthetic video, so they cover upload -> analyze -> render -> download.
"""
from __future__ import annotations

import shutil
import tempfile
import time
import unittest
import wave
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from .. import ffmpeg
from ..main import app

FFMPEG_OK = shutil.which(ffmpeg.FFMPEG) is not None


def _fixture(path: Path, seconds: float = 5.0) -> Path:
    """2.0s of tone, 0.6s of digital silence, repeating.

    The gaps have to be comfortably longer than the min_silence the tests pass,
    otherwise nothing is cuttable and the duration assertions are vacuous.
    """
    rate = 48000
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    gate = (np.mod(t, 2.6) < 2.0).astype(np.float32)
    tone = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.15 * np.sin(2 * np.pi * 660 * t)
    pcm = (tone * gate).astype(np.float32)

    wav = path.with_suffix(".wav")
    with wave.open(str(wav), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())

    ffmpeg.run([
        "-f", "lavfi", "-i", f"color=c=0x1b2a3a:s=320x240:r=25:d={seconds}",
        "-i", str(wav),
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    ], capture_stdout=False)
    return path


@unittest.skipUnless(FFMPEG_OK, "ffmpeg binary not available")
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._dir = Path(tempfile.mkdtemp(prefix="editapi_"))
        cls.clip = _fixture(cls._dir / "clip.mp4")
        # The context manager has to stay open for the whole class: leaving it
        # fires the shutdown hook, which stops the render workers.
        cls._ctx = TestClient(app)
        cls.client = cls._ctx.__enter__()
        r = cls.client.post(
            "/api/upload",
            files={"file": ("clip.mp4", cls.clip.read_bytes(), "video/mp4")},
        )
        cls.upload = r.json()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._ctx.__exit__(None, None, None)
        shutil.rmtree(cls._dir, ignore_errors=True)

    def setUp(self) -> None:
        if "asset_id" not in self.upload:
            self.fail(f"upload failed: {self.upload}")

    # -- upload -----------------------------------------------------------

    def test_health(self) -> None:
        body = self.client.get("/api/health").json()
        self.assertTrue(body["ok"])
        self.assertTrue(body["encoders"]["libx264"], "libx264 missing")

    def test_upload_rejects_audio(self) -> None:
        wav = self._dir / "sound.wav"
        with wave.open(str(wav), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(8000)
            fh.writeframes(b"\x00\x00" * 8000)
        r = self.client.post(
            "/api/upload", files={"file": ("sound.wav", wav.read_bytes(), "audio/wav")}
        )
        self.assertEqual(r.status_code, 400)
        self.assertIn("unsupported", r.json()["detail"].lower())

    def test_upload_rejects_path_traversal(self) -> None:
        from ..main import _asset_path
        from fastapi import HTTPException

        for bad in ("../../etc/passwd", "abc/def", "..", "a.b"):
            with self.assertRaises(HTTPException, msg=bad):
                _asset_path(bad)

    def test_unknown_asset_is_404(self) -> None:
        self.assertEqual(self.client.get("/api/asset/deadbeef").status_code, 404)

    def test_asset_metadata(self) -> None:
        body = self.client.get(f"/api/asset/{self.upload['asset_id']}").json()
        self.assertEqual((body["width"], body["height"]), (320, 240))
        self.assertAlmostEqual(body["fps"], 25.0, delta=0.5)
        self.assertTrue(body["has_audio"])

    def test_waveform(self) -> None:
        body = self.client.get(
            f"/api/asset/{self.upload['asset_id']}/waveform?buckets=200"
        ).json()
        self.assertEqual(len(body["buckets"]), 200)
        self.assertLessEqual(max(body["buckets"]), 1.0)
        self.assertGreater(max(body["buckets"]), 0.5, "waveform looks silent")

    # -- analysis ---------------------------------------------------------

    def test_analyze_finds_silence(self) -> None:
        body = self.client.post(
            "/api/analyze",
            json={
                "asset_id": self.upload["asset_id"],
                "sensitivity": 0.5, "min_silence": 0.3, "min_segment": 0.15,
            },
        ).json()
        self.assertGreaterEqual(len(body["keep_segments"]), 2, body)
        self.assertGreater(body["removed_seconds"], 0.3)
        self.assertGreater(body["removed_percent"], 0.0)
        kept = sum(s["duration"] for s in body["keep_segments"])
        self.assertAlmostEqual(kept + body["removed_seconds"], body["duration"], delta=0.4)

    def test_analyze_sensitivity_monotonic(self) -> None:
        """Higher sensitivity should not keep more than a low setting does."""
        counts = []
        for sens in (0.2, 0.5, 0.8):
            body = self.client.post(
                "/api/analyze",
                json={"asset_id": self.upload["asset_id"], "sensitivity": sens,
                      "min_silence": 0.3, "min_segment": 0.15},
            ).json()
            counts.append(sum(s["duration"] for s in body["keep_segments"]))
        self.assertEqual(counts[0], max(counts), counts)
        self.assertEqual(counts[2], min(counts), counts)

    def test_analyze_via_ffmpeg_detector(self) -> None:
        body = self.client.post(
            "/api/analyze",
            json={"asset_id": self.upload["asset_id"], "sensitivity": 0.5,
                  "min_silence": 0.3, "min_segment": 0.15,
                  "use_ffmpeg_detect": True},
        )
        self.assertEqual(body.status_code, 200, body.text)
        self.assertGreaterEqual(len(body.json()["keep_segments"]), 1)

    # -- render -----------------------------------------------------------

    def _upload(self, path: Path) -> str:
        r = self.client.post(
            "/api/upload", files={"file": (path.name, path.read_bytes(), "video/mp4")}
        )
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()["asset_id"]

    def test_analyze_and_render_a_video_with_no_audio(self) -> None:
        # Auto-cut on a file with no audio track used to fail: the detector
        # found no speech, produced zero segments and the graph raised
        # "nothing left to keep". The whole clip has to survive.
        silent = self._dir / "silent.mp4"
        ffmpeg.run([
            "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=navy:s=320x240:d=3",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(silent),
        ])
        asset_id = self._upload(silent)

        analysis = self.client.post(
            "/api/analyze", json={"asset_id": asset_id, "min_silence": 0.3}
        ).json()
        self.assertFalse(analysis["has_audio"])
        self.assertEqual(len(analysis["keep_segments"]), 1)
        self.assertAlmostEqual(analysis["keep_segments"][0]["end"], 3.0, delta=0.2)

        status = self._render({
            "asset_id": asset_id,
            "auto_cut": {"enabled": True, "vad": {"min_silence": 0.3}},
            "export": {"preset": "ultrafast", "crf": 32, "filename": "no-audio"},
        })
        self.assertEqual(status["status"], "done", status.get("error"))
        self.assertAlmostEqual(status["output_duration"], 3.0, delta=0.2)

    def test_render_via_ffmpeg_detector(self) -> None:
        status = self._render({
            "asset_id": self.upload["asset_id"],
            "auto_cut": {"enabled": True,
                         "vad": {"min_silence": 0.3, "min_segment": 0.15,
                                 "use_ffmpeg_detect": True}},
            "export": {"preset": "ultrafast", "crf": 32, "filename": "detector"},
        })
        self.assertEqual(status["status"], "done", status.get("error"))
        self.assertLess(status["output_duration"], 4.8)

    def _render(self, payload: dict) -> dict:
        r = self.client.post("/api/render", json=payload)
        self.assertEqual(r.status_code, 200, r.text)
        job_id = r.json()["job_id"]

        deadline = time.time() + 180
        while time.time() < deadline:
            status = self.client.get(f"/api/job/{job_id}").json()
            if status["status"] in {"done", "error", "cancelled"}:
                return status
            time.sleep(0.25)
        self.fail(f"render timed out: {status}")

    def test_render_cut_only(self) -> None:
        status = self._render({
            "asset_id": self.upload["asset_id"],
            "auto_cut": {"enabled": True, "vad": {"min_silence": 0.3, "min_segment": 0.15}},
            "export": {"preset": "ultrafast", "crf": 32, "filename": "cut-only"},
        })
        self.assertEqual(status["status"], "done", status.get("error"))
        self.assertGreater(status["output_size"], 0)
        # 5s source: two 0.6s gaps removed, so a little under 4s.
        self.assertLess(status["output_duration"], 4.8)
        self.assertGreater(status["output_duration"], 3.0)

    def test_render_download(self) -> None:
        status = self._render({
            "asset_id": self.upload["asset_id"],
            "auto_cut": {"enabled": False},
            "export": {"preset": "ultrafast", "crf": 32, "filename": "download-me"},
        })
        self.assertEqual(status["status"], "done", status.get("error"))
        r = self.client.get(f"/api/job/{status['id']}/file")
        self.assertEqual(r.status_code, 200)
        self.assertGreater(len(r.content), 1000)
        self.assertEqual(r.headers["content-type"], "video/mp4")

    def test_render_vertical_with_everything(self) -> None:
        status = self._render({
            "asset_id": self.upload["asset_id"],
            "auto_cut": {
                "enabled": True,
                "vad": {"min_silence": 0.3, "min_segment": 0.15},
                "zoom": "punchIn", "zoom_every": 1, "zoom_amount": 0.7,
            },
            "audio": {"denoise": 0.3, "normalize": True, "cut_fade": 0.04},
            "video": {"aspect": "9:16", "fit": "fit", "fps": 30,
                      "contrast": 1.1, "fade_in": 0.2, "fade_out": 0.2},
            "text": {"enabled": True, "kind": "title", "text": "Vertical Cut", "box": True},
            "transition": {"kind": "xfade", "xfade": "dissolve", "duration": 0.2},
            "export": {"preset": "ultrafast", "crf": 32, "resolution": "720p",
                       "filename": "vertical-everything"},
        })
        self.assertEqual(status["status"], "done", status.get("error"))
        # 1080x1920 capped to 720p is 405x720, rounded down to an even 404.
        self.assertEqual((status["width"], status["height"]), (404, 720))

    def test_render_manual_segments(self) -> None:
        status = self._render({
            "asset_id": self.upload["asset_id"],
            "manual_segments": [[0.3, 1.3], [2.5, 3.5]],
            "export": {"preset": "ultrafast", "crf": 32, "filename": "manual"},
        })
        self.assertEqual(status["status"], "done", status.get("error"))
        self.assertAlmostEqual(status["output_duration"], 2.0, delta=0.4)

    def test_render_filename_is_sanitised(self) -> None:
        status = self._render({
            "asset_id": self.upload["asset_id"],
            "auto_cut": {"enabled": False},
            "export": {"preset": "ultrafast", "crf": 34,
                       "filename": "../../evil name?.mp4"},
        })
        self.assertEqual(status["status"], "done", status.get("error"))
        self.assertNotIn("/", status["output_name"])
        self.assertNotIn("..", status["output_name"])
        self.assertNotIn("?", status["output_name"])

    def test_download_before_done_is_rejected(self) -> None:
        job_id = self.client.post(
            "/api/render",
            json={"asset_id": self.upload["asset_id"], "auto_cut": {"enabled": False},
                  "export": {"preset": "ultrafast"}},
        ).json()["job_id"]
        r = self.client.get(f"/api/job/{job_id}/file")
        self.assertIn(r.status_code, {409, 200})

    def test_cancel_unknown_job(self) -> None:
        self.assertEqual(self.client.post("/api/job/nope/cancel").status_code, 404)

    def test_delete_asset(self) -> None:
        r = self.client.post(
            "/api/upload",
            files={"file": ("tmp.mp4", self.clip.read_bytes(), "video/mp4")},
        )
        asset_id = r.json()["asset_id"]
        self.assertEqual(self.client.delete(f"/api/asset/{asset_id}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/asset/{asset_id}").status_code, 404)

    # -- image overlays ---------------------------------------------------

    def _png(self) -> bytes:
        path = self._dir / "still.png"
        ffmpeg.run([
            "-v", "error", "-y", "-f", "lavfi",
            "-i", "color=c=red:s=200x140:d=1", "-frames:v", "1", str(path),
        ], capture_stdout=False)
        return path.read_bytes()

    def _upload_image(self, name: str = "photo.png") -> dict:
        r = self.client.post(
            "/api/upload-image",
            files={"file": (name, self._png(), "image/png")},
        )
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_upload_image(self) -> None:
        body = self._upload_image()
        self.addCleanup(self.client.delete, f"/api/image/{body['image_id']}")
        self.assertTrue(body["image_id"].isalnum())
        self.assertEqual((body["width"], body["height"]), (200, 140))
        self.assertGreater(body["size"], 0)
        self.assertEqual(body["filename"], f"{body['image_id']}.png")

    def test_image_metadata_matches_upload(self) -> None:
        body = self._upload_image()
        self.addCleanup(self.client.delete, f"/api/image/{body['image_id']}")
        got = self.client.get(f"/api/image/{body['image_id']}").json()
        self.assertEqual(got, body)

    def test_image_file_downloads_the_bytes(self) -> None:
        body = self._upload_image()
        self.addCleanup(self.client.delete, f"/api/image/{body['image_id']}")
        r = self.client.get(f"/api/image/{body['image_id']}/file")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.content[:8], b"\x89PNG\r\n\x1a\n")

    def test_delete_image(self) -> None:
        body = self._upload_image()
        image_id = body["image_id"]
        self.assertEqual(self.client.delete(f"/api/image/{image_id}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/image/{image_id}").status_code, 404)

    def test_image_rejects_unsupported_type(self) -> None:
        r = self.client.post(
            "/api/upload-image",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
        self.assertEqual(r.status_code, 400)

    def test_image_rejects_empty_file(self) -> None:
        r = self.client.post(
            "/api/upload-image", files={"file": ("empty.png", b"", "image/png")},
        )
        self.assertEqual(r.status_code, 400)

    def test_image_rejects_path_traversal(self) -> None:
        self.assertEqual(self.client.get("/api/image/..%2F..%2Fetc").status_code in (400, 404), True)

    def test_unknown_image_is_404(self) -> None:
        self.assertEqual(self.client.get("/api/image/deadbeefdeadbeef").status_code, 404)

    def test_render_with_an_image_overlay(self) -> None:
        body = self._upload_image()
        self.addCleanup(self.client.delete, f"/api/image/{body['image_id']}")
        status = self._render({
            "asset_id": self.upload["asset_id"],
            "auto_cut": {"enabled": False},
            "overlays": [{
                "kind": "image", "image_id": body["image_id"],
                "x": 0.5, "y": 0.5, "width_pct": 0.4, "height_pct": 0.4,
                "start": 0.0, "duration": 1.0,
            }],
            "export": {"preset": "ultrafast", "crf": 32, "filename": "with-photo"},
        })
        self.assertEqual(status["status"], "done", status.get("error"))
        self.assertGreater(status["output_size"], 0)

    def test_render_with_a_text_overlay(self) -> None:
        status = self._render({
            "asset_id": self.upload["asset_id"],
            "auto_cut": {"enabled": False},
            "overlays": [{
                "kind": "text", "text": "Dragged title",
                "x": 0.5, "y": 0.5, "start": 0.0, "duration": 1.0, "font_size": 60,
            }],
            "export": {"preset": "ultrafast", "crf": 32, "filename": "with-text"},
        })
        self.assertEqual(status["status"], "done", status.get("error"))
        self.assertGreater(status["output_size"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
