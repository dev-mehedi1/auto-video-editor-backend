"""End-to-end graph tests against the real ffmpeg binary.

These generate synthetic clips with speech-like bursts separated by digital
silence, then render every feature combination and assert on the *output*
file (duration, resolution, stream presence) rather than on the filtergraph
string. A filtergraph that ffmpeg accepts but renders wrong still fails here.
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from .. import encoder, ffmpeg, graph
from ..models import (
    AudioSettings,
    AutoCutSettings,
    EditRequest,
    ExportSettings,
    Overlay,
    TextSettings,
    TransitionSettings,
    VideoSettings,
    VadSettings,
)

FFMPEG_OK = shutil.which(ffmpeg.FFMPEG) is not None

SPEECH_SECONDS = 6.0
TONE_HZ = 440


def _speech_like_pcm(seconds: float = SPEECH_SECONDS, rate: int = 48000) -> "np.ndarray":
    """Alternating 1s of tone and 0.6s of digital silence.

    Built with numpy rather than a lavfi `aevalsrc` expression: the expression
    needs escaped commas and `+` concatenation, and the resulting source
    string is unreadable and easy to get subtly wrong.
    """
    n = int(seconds * rate)
    t = np.arange(n, dtype=np.float32) / rate
    # Period 1.6s: 1.0s of "speech" then 0.6s of silence.
    phase = np.mod(t, 1.6)
    gate = (phase < 1.0).astype(np.float32)
    tone = 0.45 * np.sin(2 * np.pi * TONE_HZ * t) + 0.2 * np.sin(2 * np.pi * TONE_HZ * 1.5 * t)
    return (tone * gate).astype(np.float32)


def _write_wav(path: Path, pcm: "np.ndarray", rate: int = 48000) -> Path:
    """Mono 16-bit PCM WAV via the stdlib wave module."""
    clipped = np.clip(pcm, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(pcm16.tobytes())
    return path


def _make_source(path: Path, seconds: float = SPEECH_SECONDS) -> Path:
    """A clip that is loud for 1s, silent for 0.6s, loud for 1s, ..."""
    wav = _write_wav(path.with_suffix(".wav"), _speech_like_pcm(seconds))
    ffmpeg.run([
        "-f", "lavfi", "-i",
        f"color=c=0x203040:s=640x360:r=30:d={seconds:.3f}",
        "-i", str(wav),
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    ], capture_stdout=False)
    return path


def _make_silent_video(path: Path, seconds: float = 4.0) -> Path:
    """Video with no audio stream at all."""
    ffmpeg.run([
        "-f", "lavfi", "-i",
        f"color=c=0x101010:s=320x240:r=15:d={seconds:.3f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-an", str(path),
    ], capture_stdout=False)
    return path


def _make_watermark(path: Path) -> Path:
    ffmpeg.run([
        "-f", "lavfi", "-i", "color=c=red@0.8:s=64x64:d=1",
        "-frames:v", "1", str(path),
    ], capture_stdout=False)
    return path


def _make_srt(path: Path) -> Path:
    path.write_text(
        "1\n00:00:00,500 --> 00:00:02,000\nHello captions\n\n"
        "2\n00:00:02,500 --> 00:00:04,000\nSecond line\n",
        encoding="utf-8",
    )
    return path


def _make_overlay_photo(path: Path, colour: str = "red") -> Path:
    """A flat coloured PNG, so a rendered frame can be checked for its presence."""
    ffmpeg.run([
        "-f", "lavfi", "-i", f"color=c={colour}:s=200x140:d=1",
        "-frames:v", "1", str(path),
    ], capture_stdout=False)
    return path


FONT = Path("C:/Windows/Fonts/arial.ttf")


@unittest.skipUnless(FFMPEG_OK, "ffmpeg binary not available")
class GraphRenderTests(unittest.TestCase):
    tmp: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls._dir = tempfile.mkdtemp(prefix="edittest_")
        cls.tmp = Path(cls._dir)
        cls.source = _make_source(cls.tmp / "source.mp4")
        cls.info = ffmpeg.media_info(cls.source)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._dir, ignore_errors=True)

    def setUp(self) -> None:
        self.out = self.tmp / f"out_{self._testMethodName}.mp4"
        if self.out.exists():
            self.out.unlink()

    def _request(self, **overrides) -> EditRequest:
        payload: dict = {
            "asset_id": "test",
            "auto_cut": AutoCutSettings(
                vad=VadSettings(sensitivity=0.5, min_silence=0.3, min_segment=0.15)
            ),
            "export": ExportSettings(preset="ultrafast", crf=28),
        }
        for key, value in overrides.items():
            payload[key] = value
        return EditRequest(**payload)

    def _detect(self, **vad_kw) -> list[tuple[float, float]]:
        from .. import analysis
        params = analysis.VadParams(min_silence=0.3, min_segment=0.15, **vad_kw)
        _silences, kept, _stats = analysis.analyze(self.source, params)
        return [tuple(s) for s in kept]

    def _render(self, req: EditRequest, segments=None, source=None) -> Path:
        src = source or self.source
        info = ffmpeg.media_info(src) if source else self.info
        segs = segments if segments is not None else self._detect()
        plan = graph.build(req, info, segs, info.duration)
        encoder.render(req, plan, src, self.out)
        self.assertTrue(self.out.exists(), "ffmpeg produced no file")
        return self.out

    # -- baseline ---------------------------------------------------------

    def test_detect_finds_speech_bursts(self) -> None:
        segments = self._detect()
        self.assertGreaterEqual(len(segments), 2, segments)
        kept = sum(e - s for s, e in segments)
        self.assertLess(kept, self.info.duration, "silence was not removed")

    def test_render_duration_matches_kept_audio(self) -> None:
        req = self._request()
        segments = self._detect()
        out = self._render(req, segments)
        result = ffmpeg.media_info(out)

        expected = sum(e - s for s, e in segments)
        self.assertAlmostEqual(
            result.duration, expected,
            msg=f"expected ~{expected:.2f}s, got {result.duration:.2f}s",
            delta=0.35,
        )
        self.assertEqual((result.width, result.height), (640, 360))
        self.assertTrue(result.has_audio)
        self.assertAlmostEqual(result.fps, 30.0, delta=0.2)

    def test_source_untouched_when_cut_disabled(self) -> None:
        req = self._request(auto_cut=AutoCutSettings(enabled=False))
        out = self._render(req, segments=[(0.0, self.info.duration)])
        result = ffmpeg.media_info(out)
        self.assertAlmostEqual(result.duration, self.info.duration, delta=0.3)

    # -- the fps-after-zoompan bug ---------------------------------------

    def test_zoom_does_not_duplicate_frames(self) -> None:
        """Regression: a trailing fps after zoompan made 6s into ~1500s."""
        for style in ("punchIn", "punchOut", "shake", "left", "right"):
            with self.subTest(zoom=style):
                out = self.out.with_name(f"zoom_{style}.mp4")
                req = self._request(
                    auto_cut=AutoCutSettings(
                        vad=VadSettings(min_silence=0.3, min_segment=0.15),
                        zoom=style,
                        zoom_every=1,
                        zoom_amount=0.8,
                    )
                )
                plan = graph.build(req, self.info, self._detect(), self.info.duration)
                encoder.render(req, plan, self.source, out)
                result = ffmpeg.media_info(out)
                self.assertLess(
                    result.duration, SPEECH_SECONDS * 1.6,
                    f"zoom={style} produced {result.duration:.1f}s",
                )
                self.assertAlmostEqual(
                    result.duration,
                    sum(e - s for s, e in self._detect()),
                    delta=0.6,
                )
                self.assertEqual((result.width, result.height), (640, 360))

    # -- geometry ---------------------------------------------------------

    def test_aspect_presets(self) -> None:
        # The 640x360 fixture already matches 16:9, so that one is a no-op;
        # the others letterbox onto a full-size target canvas.
        cases = {"16:9": (640, 360), "9:16": (1080, 1920), "1:1": (1080, 1080)}
        for preset, expected in cases.items():
            with self.subTest(aspect=preset):
                out = self.out.with_name(f"aspect_{preset.replace(':', 'x')}.mp4")
                req = self._request(video=VideoSettings(aspect=preset, fit="fit"))
                plan = graph.build(req, self.info, self._detect(), self.info.duration)
                encoder.render(req, plan, self.source, out)
                got = ffmpeg.media_info(out)
                self.assertEqual(
                    (got.width, got.height), expected,
                    f"{preset} -> {got.width}x{got.height}",
                )

    def test_rotate(self) -> None:
        req = self._request(video=VideoSettings(rotate=90))
        out = self._render(req)
        result = ffmpeg.media_info(out)
        self.assertEqual((result.width, result.height), (360, 640))

    def test_resolution_cap(self) -> None:
        req = self._request(export=ExportSettings(resolution="360p", preset="ultrafast"))
        out = self._render(req)
        self.assertEqual(ffmpeg.media_info(out).height, 360)

    def test_frame_rate_conversion(self) -> None:
        req = self._request(video=VideoSettings(fps=15))
        out = self._render(req)
        self.assertAlmostEqual(ffmpeg.media_info(out).fps, 15.0, delta=0.3)

    # -- transitions ------------------------------------------------------

    def test_xfade_shortens_output(self) -> None:
        segments = self._detect()
        hard = self._request()
        trans = self._request(
            transition=TransitionSettings(kind="xfade", xfade="fade", duration=0.3)
        )
        out_hard = self.tmp / "t_hard.mp4"
        out_soft = self.tmp / "t_soft.mp4"
        for req, out in ((hard, out_hard), (trans, out_soft)):
            plan = graph.build(req, self.info, segments, self.info.duration)
            encoder.render(req, plan, self.source, out)

        d_hard = ffmpeg.media_info(out_hard).duration
        d_soft = ffmpeg.media_info(out_soft).duration
        self.assertLess(d_soft, d_hard, "xfade should overlap and shorten the cut")

    def test_fade_kind_crossfades_like_xfade(self) -> None:
        # "fade" is the friendly label for an overlapping dissolve. It must not
        # silently degrade to a hard cut.
        segments = self._detect()
        cut = self._request(transition=TransitionSettings(kind="cut"))
        fade = self._request(
            transition=TransitionSettings(kind="fade", xfade="fade", duration=0.3)
        )
        out_cut = self.tmp / "k_cut.mp4"
        out_fade = self.tmp / "k_fade.mp4"
        for req, out in ((cut, out_cut), (fade, out_fade)):
            plan = graph.build(req, self.info, segments, self.info.duration)
            encoder.render(req, plan, self.source, out)

        self.assertIn("xfade=", plan.filter_complex)
        self.assertLess(
            ffmpeg.media_info(out_fade).duration,
            ffmpeg.media_info(out_cut).duration,
        )

    def test_every_xfade_preset(self) -> None:
        segments = self._detect()
        for name in graph._XFADE:
            with self.subTest(xfade=name):
                out = self.tmp / f"xf_{name}.mp4"
                req = self._request(
                    transition=TransitionSettings(kind="xfade", xfade=name, duration=0.2)
                )
                plan = graph.build(req, self.info, segments, self.info.duration)
                encoder.render(req, plan, self.source, out)
                self.assertTrue(out.exists() and out.stat().st_size > 500)
                self.assertTrue(ffmpeg.media_info(out).has_video)

    # -- speed ------------------------------------------------------------

    def test_speed_changes_output_duration(self) -> None:
        segments = self._detect()
        fast = self._request(
            auto_cut=AutoCutSettings(
                vad=VadSettings(min_silence=0.3, min_segment=0.15),
                speed="normal", speed_value=2.0,
            )
        )
        out = self._render(fast, segments)
        got = ffmpeg.media_info(out).duration
        expected = sum(e - s for s, e in segments) / 2.0
        self.assertAlmostEqual(got, expected, delta=0.4)

    def test_ramp_up_progressively_speeds_up(self) -> None:
        segments = self._detect()
        speeds = [graph._speed_for("rampUp", 2.0, i, len(segments))
                  for i in range(len(segments))]
        self.assertAlmostEqual(speeds[0], 1.0)
        self.assertAlmostEqual(speeds[-1], 2.0)
        self.assertEqual(speeds, sorted(speeds))
        req = self._request(
            auto_cut=AutoCutSettings(
                vad=VadSettings(min_silence=0.3, min_segment=0.15),
                speed="rampUp", speed_value=2.0,
            )
        )
        out = self._render(req, segments)
        self.assertTrue(ffmpeg.media_info(out).duration < SPEECH_SECONDS)

    # -- audio ------------------------------------------------------------

    def test_mute_removes_audio(self) -> None:
        req = self._request(audio=AudioSettings(mute=True, normalize=False))
        out = self._render(req)
        self.assertFalse(ffmpeg.media_info(out).has_audio)

    def test_audio_cleanup_chain(self) -> None:
        req = self._request(audio=AudioSettings(
            highpass_hz=80, lowpass_hz=12000, denoise=0.4,
            gain_db=-3.0, compress=True, deesser=True, normalize=True,
        ))
        out = self._render(req)
        self.assertTrue(ffmpeg.media_info(out).has_audio)

    def test_cut_fade(self) -> None:
        req = self._request(audio=AudioSettings(cut_fade=0.08, normalize=False))
        out = self._render(req)
        self.assertTrue(ffmpeg.media_info(out).has_audio)

    def test_silent_source_gets_silent_track(self) -> None:
        silent = _make_silent_video(self.tmp / "silent.mp4")
        info = ffmpeg.media_info(silent)
        self.assertFalse(info.has_audio)
        req = self._request()
        plan = graph.build(req, info, [(0.0, info.duration)], info.duration)
        self.assertTrue(plan.warnings)
        encoder.render(req, plan, silent, self.out)
        result = ffmpeg.media_info(self.out)
        self.assertTrue(result.has_audio, "a silent track should be synthesised")
        self.assertAlmostEqual(result.duration, info.duration, delta=0.3)

    # -- overlays ---------------------------------------------------------

    def test_drawtext_title(self) -> None:
        req = self._request(
            text=TextSettings(
                enabled=True, kind="title", text="Chapter One: 50% off",
                font_size=40, box=True, align="bottom", margin_v=40,
            )
        )
        out = self._render(req)
        self.assertTrue(ffmpeg.media_info(out).has_video)

    def test_drawtext_special_characters(self) -> None:
        """Commas, colons, percent and apostrophes all break naive graphs."""
        req = self._request(text=TextSettings(
            enabled=True, kind="title",
            text="a,b: c; 100% [x] 'quoted' \\ back",
            font_size=32, box=True, shadow=0,
        ))
        out = self._render(req)
        self.assertTrue(ffmpeg.media_info(out).has_video)

    def test_burned_in_captions(self) -> None:
        srt = _make_srt(self.tmp / "subs.srt")
        req = self._request(
            text=TextSettings(enabled=True, kind="captions", subtitle_file=str(srt))
        )
        out = self._render(req)
        self.assertTrue(ffmpeg.media_info(out).has_video)

    def test_watermark_overlay(self) -> None:
        mark = _make_watermark(self.tmp / "mark.png")
        req = self._request(text=TextSettings(
            enabled=True, kind="title", text="", watermark_file=str(mark),
            watermark_position="bottomRight", watermark_opacity=0.5,
        ))
        out = self._render(req)
        self.assertTrue(ffmpeg.media_info(out).has_video)

    # -- draggable overlays ------------------------------------------------

    def test_image_overlay_renders(self) -> None:
        photo = _make_overlay_photo(self.tmp / "ov.png", "red")
        image_id = self._store_image(photo)
        req = self._request(overlays=[Overlay(
            kind="image", image_id=image_id, x=0.5, y=0.5,
            width_pct=0.4, height_pct=0.4, start=0.0, duration=2.0,
        )])
        out = self._render(req)
        info = ffmpeg.media_info(out)
        self.assertTrue(info.has_video)
        self.assertGreater(info.duration, 0.1)

    def test_image_overlay_position_moves_the_pixels(self) -> None:
        """The whole point of the drag handle: position has to actually move."""
        photo = _make_overlay_photo(self.tmp / "ov_pos.png", "red")
        image_id = self._store_image(photo)

        def corner_red(x: float, y: float) -> float:
            """Red channel of the top-left corner when the photo is at (x, y)."""
            req = self._request(overlays=[Overlay(
                kind="image", image_id=image_id, x=x, y=y,
                width_pct=0.5, height_pct=0.5, start=0.0, duration=2.0,
            )])
            out = self._render(req, segments=[(0.0, 1.0)])
            return self._red_score(out)

        # The source is 0x203040 (red 32), so a red corner means the photo is
        # there. Compare the two placements rather than using a hard cutoff:
        # H.264 bleeds a little chroma at the frame edge.
        top_left = corner_red(0.25, 0.25)
        bottom_right = corner_red(0.75, 0.75)
        self.assertGreater(top_left, 100.0, f"photo missing from top-left: {top_left}")
        self.assertLess(
            bottom_right, 90.0, f"photo still in the top-left corner: {bottom_right}"
        )
        self.assertGreater(
            top_left - bottom_right, 60.0,
            f"moving the photo barely changed the corner: {top_left} vs {bottom_right}",
        )

    def test_image_overlay_opacity(self) -> None:
        photo = _make_overlay_photo(self.tmp / "ov_op.png", "red")
        image_id = self._store_image(photo)
        req = self._request(overlays=[Overlay(
            kind="image", image_id=image_id, x=0.5, y=0.5,
            width_pct=0.8, height_pct=0.8, start=0.0, duration=2.0,
            opacity=0.3,
        )])
        out = self._render(req, segments=[(0.0, 1.0)])
        self.assertTrue(ffmpeg.media_info(out).has_video)

    def test_image_overlay_rotation(self) -> None:
        photo = _make_overlay_photo(self.tmp / "ov_rot.png", "red")
        image_id = self._store_image(photo)
        req = self._request(overlays=[Overlay(
            kind="image", image_id=image_id, x=0.5, y=0.5,
            width_pct=0.5, height_pct=0.5, start=0.0, duration=2.0,
            rotation=35.0,
        )])
        out = self._render(req, segments=[(0.0, 1.0)])
        self.assertTrue(ffmpeg.media_info(out).has_video)

    def test_image_overlay_time_window(self) -> None:
        """A late overlay must not touch the first frame."""
        photo = _make_overlay_photo(self.tmp / "ov_late.png", "red")
        image_id = self._store_image(photo)
        req = self._request(overlays=[Overlay(
            kind="image", image_id=image_id, x=0.5, y=0.5,
            width_pct=0.9, height_pct=0.9, start=3.0, duration=2.0,
        )])
        out = self._render(req, segments=[(0.0, 4.0)])
        self.assertTrue(ffmpeg.media_info(out).has_video)

    def test_text_overlay_renders(self) -> None:
        if not FONT.exists():
            self.skipTest("arial.ttf not available")
        req = self._request(overlays=[Overlay(
            kind="text", text="Dragged title", x=0.5, y=0.5,
            start=0.0, duration=2.0, font_size=60,
        )])
        out = self._render(req)
        self.assertTrue(ffmpeg.media_info(out).has_video)

    def test_text_overlay_special_characters(self) -> None:
        if not FONT.exists():
            self.skipTest("arial.ttf not available")
        req = self._request(overlays=[Overlay(
            kind="text", text='a,b: 100% [x] \'q\' \\ end', x=0.5, y=0.4,
            start=0.0, duration=2.0, font_size=36,
        )])
        out = self._render(req)
        self.assertTrue(ffmpeg.media_info(out).has_video)

    def test_overlay_missing_image_is_a_warning_not_a_failure(self) -> None:
        req = self._request(overlays=[Overlay(
            kind="image", image_id="deadbeefdeadbeef", x=0.5, y=0.5,
        )])
        plan = graph.build(req, self.info, [(0.0, 1.0)], self.info.duration)
        self.assertTrue(any("no longer exists" in w for w in plan.warnings))

    def test_overlay_empty_list_changes_nothing(self) -> None:
        req = self._request(overlays=[])
        out = self._render(req)
        self.assertTrue(ffmpeg.media_info(out).has_video)

    def test_overlay_text_temp_file_is_cleaned(self) -> None:
        if not FONT.exists():
            self.skipTest("arial.ttf not available")
        req = self._request(overlays=[Overlay(
            kind="text", text="cleanup", font_size=40,
        )])
        plan = graph.build(req, self.info, [(0.0, 1.0)], self.info.duration)
        self.assertTrue(plan.cleanup_paths)
        created = [Path(p) for p in plan.cleanup_paths]
        self.assertTrue(all(p.exists() for p in created))
        graph.cleanup(plan)
        self.assertFalse(any(p.exists() for p in created))

    def test_image_overlay_and_text_overlay_together(self) -> None:
        if not FONT.exists():
            self.skipTest("arial.ttf not available")
        photo = _make_overlay_photo(self.tmp / "ov_both.png", "blue")
        image_id = self._store_image(photo)
        req = self._request(overlays=[
            Overlay(kind="image", image_id=image_id, x=0.3, y=0.3,
                    width_pct=0.25, height_pct=0.25, duration=2.0),
            Overlay(kind="text", text="Both at once", x=0.5, y=0.8,
                    duration=2.0, font_size=48),
        ])
        out = self._render(req)
        self.assertTrue(ffmpeg.media_info(out).has_video)

    def test_overlay_survives_a_vertical_reframe(self) -> None:
        """x/y must address the delivery frame, not the source frame."""
        photo = _make_overlay_photo(self.tmp / "ov_916.png", "red")
        image_id = self._store_image(photo)
        req = self._request(
            video=VideoSettings(aspect="9:16", fit="fit"),
            overlays=[Overlay(
                kind="image", image_id=image_id, x=0.5, y=0.5,
                width_pct=0.6, height_pct=0.2, start=0.0, duration=2.0,
            )],
        )
        plan = graph.build(req, self.info, [(0.0, 1.0)], self.info.duration)
        self.assertEqual((plan.width, plan.height), (1080, 1920))
        # 60% of the 1080-wide delivery frame, not of the source width.
        self.assertIn("648", plan.filter_complex)

    def _store_image(self, path: Path) -> str:
        """Copy a photo into the image store and return its id.

        Ids must be alphanumeric, so a counter is used rather than the
        filename, which carries underscores.
        """
        image_root = graph.IMAGE_ROOT
        image_root.mkdir(parents=True, exist_ok=True)
        self._image_seq = getattr(self, "_image_seq", 0) + 1
        image_id = f"testimg{self._image_seq:04d}"
        target = image_root / f"{image_id}{path.suffix}"
        shutil.copy2(path, target)
        self.addCleanup(target.unlink, missing_ok=True)
        return image_id

    def _red_score(self, video: Path) -> float:
        """Red channel of one frame's top-left corner, as a 'is the photo there' probe.

        Raw bytes go to a file rather than stdout: `ffmpeg.run` hands back text,
        which mangles anything that is not valid utf-8.
        """
        raw = self.tmp / f"probe_{self._testMethodName}.raw"
        ffmpeg.run([
            "-i", str(video), "-frames:v", "1",
            "-vf", "crop=iw/4:ih/4:0:0,scale=1:1",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-y", str(raw),
        ], capture_stdout=False)
        data = raw.read_bytes()
        raw.unlink(missing_ok=True)
        return float(data[0]) if len(data) >= 3 else 0.0

    # -- colour -----------------------------------------------------------

    def test_colour_grading(self) -> None:
        req = self._request(video=VideoSettings(
            brightness=0.08, contrast=1.15, saturation=1.3, gamma=1.1,
            sharpen=0.4, vignette=0.3,
        ))
        out = self._render(req)
        self.assertTrue(ffmpeg.media_info(out).has_video)

    def test_global_fades(self) -> None:
        req = self._request(video=VideoSettings(fade_in=0.2, fade_out=0.2))
        out = self._render(req)
        self.assertTrue(ffmpeg.media_info(out).has_video)

    # -- exports ----------------------------------------------------------

    def test_all_exported_formats(self) -> None:
        for codec, ext in (("h264", "mp4"), ("h265", "mp4"), ("vp9", "webm"), ("gif", "gif")):
            with self.subTest(codec=codec):
                out = self.tmp / f"exp_{codec}.{ext}"
                req = self._request(
                    export=ExportSettings(video_codec=codec, crf=34, preset="ultrafast")
                )
                plan = graph.build(req, self.info, self._detect(), self.info.duration)
                encoder.render(req, plan, self.source, out)
                self.assertTrue(out.exists(), f"{codec} produced nothing")
                self.assertGreater(out.stat().st_size, 300)

    # -- manual segments --------------------------------------------------

    def test_manual_segments_override(self) -> None:
        req = self._request(manual_segments=[[0.2, 1.0], [3.0, 4.0]])
        out = self._render(req, segments=[])
        got = ffmpeg.media_info(out).duration
        self.assertAlmostEqual(got, 1.8, delta=0.3)

    def test_segments_clamped_to_duration(self) -> None:
        segs = graph.clean_segments([(0.0, 999.0), (-5.0, 1.0)], self.info.duration)
        self.assertEqual(segs, [(0.0, round(self.info.duration, 3))])

    def test_everything_on_at_once(self) -> None:
        """The real test: all panels enabled simultaneously must still run."""
        mark = _make_watermark(self.tmp / "mark_all.png")
        srt = _make_srt(self.tmp / "all.srt")
        req = self._request(
            auto_cut=AutoCutSettings(
                vad=VadSettings(sensitivity=0.5, min_silence=0.3, min_segment=0.15),
                zoom="punchIn", zoom_every=2, zoom_amount=0.7,
            ),
            audio=AudioSettings(
                highpass_hz=80, denoise=0.3, compress=True,
                cut_fade=0.05, normalize=True,
            ),
            video=VideoSettings(
                aspect="9:16", fit="fit", fps=30,
                brightness=0.05, contrast=1.1, saturation=1.2,
                sharpen=0.3, vignette=0.2, fade_in=0.2, fade_out=0.2,
            ),
            text=TextSettings(
                enabled=True, kind="title", text="Full Stack", box=True,
                watermark_file=str(mark),
            ),
            transition=TransitionSettings(kind="xfade", xfade="dissolve", duration=0.2),
            export=ExportSettings(video_codec="h264", crf=30, preset="ultrafast",
                                  resolution="480p"),
        )
        out = self.tmp / "everything.mp4"
        plan = graph.build(req, self.info, self._detect(), self.info.duration)
        encoder.render(req, plan, self.source, out)
        result = ffmpeg.media_info(out)
        self.assertTrue(result.has_video and result.has_audio)
        self.assertEqual((result.width, result.height), (270, 480))
        self.assertLess(result.duration, SPEECH_SECONDS)
        self._ = srt  # captions are covered separately


@unittest.skipUnless(FFMPEG_OK, "ffmpeg binary not available")
class PlanUnitTests(unittest.TestCase):
    """Cheap checks that do not need a real render."""

    def setUp(self) -> None:
        self.info = ffmpeg.MediaInfo(
            duration=10.0, width=1920, height=1080, fps=30.0,
            has_video=True, has_audio=True, display_width=1920, display_height=1080,
        )

    def test_escaping(self) -> None:
        self.assertEqual(graph.esc_path("a:b\\c.mp4"), "a\\:b/c.mp4")
        self.assertEqual(graph.color("#FF0000"), "0xFF0000")
        self.assertEqual(graph.color("#FF0000FF"), "0xFF0000FF")
        self.assertEqual(graph.color("bogus", "black"), "black")

    def test_text_file_is_cleaned_up(self) -> None:
        req = EditRequest(
            asset_id="a", audio=AudioSettings(mute=True),
            text=TextSettings(enabled=True, kind="title", text="hello"),
        )
        plan = graph.build(req, self.info, [(0, 2)], 10.0)
        self.assertEqual(len(plan.cleanup_paths), 1)
        temp = Path(plan.cleanup_paths[0])
        self.assertTrue(temp.exists())
        self.assertIn("hello", temp.read_text(encoding="utf-8"))
        graph.cleanup(plan)
        self.assertFalse(temp.exists(), "temp text file was left behind")

    def test_segments_are_ordered_and_merged(self) -> None:
        got = graph.clean_segments([(5, 6), (0, 1), (1, 2)], 10.0)
        self.assertEqual(got, [(0.0, 2.0), (5.0, 6.0)])

    def test_single_segment_concat(self) -> None:
        req = EditRequest(asset_id="a", audio=AudioSettings(mute=True))
        plan = graph.build(req, self.info, [(1.0, 3.0)], 10.0)
        self.assertIn("concat=n=1:v=1:a=0", plan.filter_complex)
        self.assertAlmostEqual(plan.output_duration, 2.0)

    def test_output_size_is_even(self) -> None:
        req = EditRequest(
            asset_id="a", audio=AudioSettings(mute=True),
            video=VideoSettings(aspect="4:5", fit="fill"),
        )
        plan = graph.build(req, self.info, [(0, 2)], 10.0)
        self.assertEqual(plan.width % 2, 0)
        self.assertEqual(plan.height % 2, 0)

    def test_never_upscales_a_small_source(self) -> None:
        """A 320x240 clip asking for 16:9 keeps its content at 320x240."""
        small = ffmpeg.MediaInfo(
            duration=5.0, width=320, height=240, fps=30.0,
            has_video=True, display_width=320, display_height=240,
        )
        req = EditRequest(
            asset_id="a", audio=AudioSettings(mute=True),
            video=VideoSettings(aspect="16:9", fit="fit"),
        )
        geo = graph.geometry(small, req.video, req.export.resolution)
        self.assertEqual((geo.pre_w, geo.pre_h), (320, 240), "content was upscaled")
        self.assertIn("pad=1920:1080", geo.fit_filter or "")
        # Nothing in the chain may scale the content above its native size.
        plan = graph.build(req, small, [(0, 2)], 5.0)
        self.assertNotIn("scale=1920:1080", plan.filter_complex)

    def test_fit_letterboxes_a_wide_source_into_vertical(self) -> None:
        """1080p to 9:16 fit keeps the frame 9:16 and bars the sides."""
        req = EditRequest(
            asset_id="a", audio=AudioSettings(mute=True),
            video=VideoSettings(aspect="9:16", fit="fit"),
        )
        plan = graph.build(req, self.info, [(0, 2)], 10.0)
        self.assertEqual((plan.width, plan.height), (1080, 1920))
        self.assertIn("pad=1080:1920", plan.filter_complex)

    def test_fill_crops_to_the_target_box(self) -> None:
        req = EditRequest(
            asset_id="a", audio=AudioSettings(mute=True),
            video=VideoSettings(aspect="9:16", fit="fill"),
        )
        plan = graph.build(req, self.info, [(0, 2)], 10.0)
        self.assertEqual((plan.width, plan.height), (1080, 1920))
        self.assertIn("crop=1080:1920", plan.filter_complex)
        # Branches must oversample to the 9:16 box so the crop has pixels to
        # work with; 1920x1080 covering 1080x1920 is ~3412x1920.
        geo = graph.geometry(self.info, req.video, req.export.resolution)
        self.assertGreaterEqual(geo.pre_w, 1080)
        self.assertEqual(geo.pre_h, 1920)
        self.assertIn(f"scale={geo.pre_w}:{geo.pre_h}", plan.filter_complex)

    def test_all_silent_is_a_clear_error(self) -> None:
        req = EditRequest(asset_id="a", audio=AudioSettings(mute=True))
        with self.assertRaises(ValueError) as ctx:
            graph.build(req, self.info, [], 10.0)
        self.assertIn("silence", str(ctx.exception).lower())

    def test_xfade_offsets(self) -> None:
        # Three 2s clips with a 0.5s overlap: merge at 1.5 then 3.0.
        self.assertEqual(
            [round(o, 3) for o in graph._xfade_offsets([2.0, 2.0, 2.0], 0.5)],
            [1.5, 3.0],
        )

    def test_xfade_capped_to_shortest_segment(self) -> None:
        # Segments are 4.0s and 1.0s, so a 3.0s request must be clamped to
        # half the shortest one.
        req = EditRequest(
            asset_id="a",
            transition=TransitionSettings(kind="xfade", duration=3.0),
        )
        plan = graph.build(req, self.info, [(0, 4.0), (5, 6.0)], 10.0)
        self.assertIn("xfade=transition=fade:duration=0.500", plan.filter_complex)

    def test_mute_means_no_audio_branch(self) -> None:
        req = EditRequest(asset_id="a", audio=AudioSettings(mute=True))
        plan = graph.build(req, self.info, [(0, 2), (4, 6)], 10.0)
        self.assertNotIn("[0:a]", plan.filter_complex)
        self.assertNotIn("anullsrc", plan.filter_complex)
        self.assertFalse(plan.has_output_audio)
        self.assertEqual(plan.maps, ["-map", "[vcat]"])

    def test_speed_factors_follow_mode(self) -> None:
        self.assertAlmostEqual(graph._speed_for("normal", 1.75, 0, 4), 1.75)
        self.assertAlmostEqual(graph._speed_for("rampUp", 2.0, 0, 4), 1.0)
        self.assertAlmostEqual(graph._speed_for("rampUp", 2.0, 3, 4), 2.0)
        self.assertAlmostEqual(graph._speed_for("rampUpDown", 2.0, 0, 5), 1.0)

    def test_rotate_normalises_to_multiples_of_90(self) -> None:
        self.assertEqual(VideoSettings(rotate=100).rotate, 90)
        self.assertEqual(VideoSettings(rotate=-90).rotate, 270)


if __name__ == "__main__":
    unittest.main(verbosity=2)
