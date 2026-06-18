import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "video_translator"
sys.path.insert(0, str(PACKAGE))


def test_g20_language_matrix_contains_core_languages():
    from config.languages import get_language_capabilities

    languages = get_language_capabilities()
    for code in ["zh", "en", "es", "fr", "de", "ja", "ko", "ar", "hi", "id"]:
        assert code in languages
        assert languages[code].translate == "stable"


def test_indextts2_requires_base_url(tmp_path):
    from core.tts_indextts2 import IndexTTS2Engine

    engine = IndexTTS2Engine()
    with pytest.raises(RuntimeError, match="接口地址"):
        engine.generate("hello", tmp_path / "out.wav")


def test_settings_panel_defaults_to_edge_tts(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from ui.components.settings_panel import SettingsPanel, SettingsSections

    app = QApplication.instance() or QApplication([])
    original_loader = SettingsSections._load_edge_voices
    SettingsSections._load_edge_voices = lambda self: None
    panel = SettingsPanel(settings_path=tmp_path / "settings.json")
    SettingsSections._load_edge_voices = original_loader
    settings = panel.get_settings()

    assert settings["tts_engine"] == "edge-tts"
    assert settings["tts_voice"] == "zh-CN-XiaoxiaoNeural"
    assert panel.sections.edge_voice_combo.count() > 1
    assert not hasattr(panel.sections, "edge_voice_input")
    assert settings["indextts2_endpoint"] == "/tts"
    assert settings["api_key"] == ""
    assert settings["model"] == "gpt-4o-mini"
    assert hasattr(panel.sections, "install_demucs_btn")


def test_settings_panel_can_select_active_llm_provider():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from ui.components.settings_panel import SettingsPanel, SettingsSections

    app = QApplication.instance() or QApplication([])
    original_loader = SettingsSections._load_edge_voices
    SettingsSections._load_edge_voices = lambda self: None
    panel = SettingsPanel()
    SettingsSections._load_edge_voices = original_loader
    sections = panel.sections

    sections._add_provider()
    sections.provider_input.setText("DeepSeek")
    sections.api_key_input.setText("deepseek-key")
    sections.base_url_input.setText("https://api.deepseek.com")
    sections.model_combo.setCurrentText("deepseek-chat")
    sections._set_active_provider()

    settings = panel.get_settings()

    assert settings["provider"] == "DeepSeek"
    assert settings["api_key"] == "deepseek-key"
    assert settings["base_url"] == "https://api.deepseek.com"
    assert settings["model"] == "deepseek-chat"
    assert len(settings["llm_providers"]) == 2


def test_settings_panel_persists_llm_providers(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from ui.components.settings_panel import SettingsPanel, SettingsSections

    app = QApplication.instance() or QApplication([])
    original_loader = SettingsSections._load_edge_voices
    SettingsSections._load_edge_voices = lambda self: None
    settings_path = tmp_path / "settings.json"
    panel = SettingsPanel(settings_path=settings_path)
    sections = panel.sections

    sections._add_provider()
    sections.provider_input.setText("Custom")
    sections.api_key_input.setText("custom-key")
    sections.base_url_input.setText("https://api.example.com")
    sections.model_combo.setCurrentText("custom-model")
    sections._set_active_provider()
    sections.save_settings()

    restored = SettingsPanel(settings_path=settings_path)
    SettingsSections._load_edge_voices = original_loader
    settings = restored.get_settings()

    assert settings["provider"] == "Custom"
    assert settings["api_key"] == "custom-key"
    assert settings["base_url"] == "https://api.example.com"
    assert settings["model"] == "custom-model"
    assert len(settings["llm_providers"]) == 2


def test_llm_base_url_candidates_try_v1_fallback():
    from core.translator import _base_url_candidates

    assert _base_url_candidates("https://api.example.com") == [
        "https://api.example.com",
        "https://api.example.com/v1",
    ]
    assert _base_url_candidates("https://api.example.com/v1") == ["https://api.example.com/v1"]


def test_llm_chat_completion_retries(monkeypatch):
    from core import translator

    calls = {"count": 0}

    class DummyMessage:
        content = "OK"

    class DummyChoice:
        message = DummyMessage()

    class DummyResponse:
        choices = [DummyChoice()]

    class DummyCompletions:
        def create(self, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("network hiccup")
            return DummyResponse()

    class DummyChat:
        completions = DummyCompletions()

    class DummyOpenAI:
        def __init__(self, **kwargs):
            self.chat = DummyChat()

    monkeypatch.setitem(sys.modules, "openai", type("OpenAIModule", (), {"OpenAI": DummyOpenAI}))
    monkeypatch.setattr(translator.time, "sleep", lambda _seconds: None)

    response = translator._chat_completion(
        api_key="key",
        base_url="https://api.example.com/v1",
        model="model",
        messages=[],
    )

    assert response.choices[0].message.content == "OK"
    assert calls["count"] == 2


def test_demo_translation_mode_is_rejected():
    from core.translator import translate_segments

    segments = [{"start": 0, "end": 1, "text": "hello world"}]
    with pytest.raises(ValueError, match="真实 API Key"):
        translate_segments(segments, target_lang="zh", api_key="demo", model="demo-copy")


def test_pipeline_cancel_stops_running_process():
    from core import pipeline

    translator = pipeline.TranslationPipeline(api_key="demo")
    errors = []
    fake_processes = []

    class FakeProcess:
        def __init__(self, *args, **kwargs):
            self.returncode = None
            self.terminated = False
            fake_processes.append(self)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

        def communicate(self):
            return "", ""

    original_popen = pipeline.subprocess.Popen
    original_sleep = pipeline.time.sleep
    pipeline.subprocess.Popen = FakeProcess
    pipeline.time.sleep = lambda _seconds: original_sleep(0.01)

    def run_process():
        try:
            translator._run_process(["fake-long-process"], check=True)
        except Exception as exc:
            errors.append(exc)

    try:
        thread = threading.Thread(target=run_process)
        start = time.time()
        thread.start()
        time.sleep(0.05)
        translator.cancel()
        thread.join(timeout=5)
    finally:
        pipeline.subprocess.Popen = original_popen
        pipeline.time.sleep = original_sleep

    assert not thread.is_alive()
    assert time.time() - start < 5
    assert fake_processes and fake_processes[0].terminated
    assert any(isinstance(error, InterruptedError) for error in errors)


def test_pipeline_times_out_stuck_ffprobe():
    from core import pipeline

    translator = pipeline.TranslationPipeline(api_key="demo")
    fake_processes = []
    ticks = {"value": 0.0}

    class FakeProcess:
        def __init__(self, *args, **kwargs):
            self.returncode = None
            self.terminated = False
            fake_processes.append(self)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

        def communicate(self):
            return "", ""

    original_popen = pipeline.subprocess.Popen
    original_monotonic = pipeline.time.monotonic
    original_sleep = pipeline.time.sleep
    pipeline.subprocess.Popen = FakeProcess
    pipeline.time.monotonic = lambda: ticks["value"]

    def fake_sleep(_seconds):
        ticks["value"] += pipeline.FFPROBE_TIMEOUT_SECONDS + 1

    pipeline.time.sleep = fake_sleep

    try:
        with pytest.raises(TimeoutError, match="ffprobe"):
            translator._run_process(["ffprobe", "-version"], check=True)
    finally:
        pipeline.subprocess.Popen = original_popen
        pipeline.time.monotonic = original_monotonic
        pipeline.time.sleep = original_sleep

    assert fake_processes and fake_processes[0].terminated


def test_pipeline_drains_child_process_output():
    from core import pipeline
    from io import StringIO

    translator = pipeline.TranslationPipeline(api_key="demo")

    class FakeProcess:
        def __init__(self, *args, **kwargs):
            self.returncode = 0
            self.stdout = StringIO("stdout line\n")
            self.stderr = StringIO("stderr line\n")

        def poll(self):
            return self.returncode

    original_popen = pipeline.subprocess.Popen
    pipeline.subprocess.Popen = FakeProcess

    try:
        result = translator._run_process(["fake-process"], check=True)
    finally:
        pipeline.subprocess.Popen = original_popen

    assert result.stdout == "stdout line\n"
    assert result.stderr == "stderr line\n"


def test_pipeline_transcribe_retries_once(monkeypatch, tmp_path):
    from core import pipeline

    calls = {"count": 0}
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"fake")
    translator = pipeline.TranslationPipeline(api_key="demo")

    def fake_transcribe(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary ASR error")
        return {"segments": [], "language": "en", "text": ""}

    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    monkeypatch.setattr(pipeline.time, "sleep", lambda _seconds: None)

    result = translator._transcribe_with_retries(audio, language=None)

    assert calls["count"] == 2
    assert result["language"] == "en"


def test_tts_mix_keeps_subtitle_timeline(monkeypatch, tmp_path):
    from core import pipeline

    commands = []

    def fake_run(cmd, **kwargs):
        if cmd[0] == "ffprobe":
            class ProbeResult:
                stdout = "0.5"

            return ProbeResult()
        commands.append(cmd)

        class Result:
            stderr = ""

        return Result()

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first.write_bytes(b"fake")
    second.write_bytes(b"fake")

    pipeline._mix_timed_audio_segments(
        [
            ({"start": 1.25, "end": 2.0}, first),
            ({"start": 4.5, "end": 5.0}, second),
        ],
        tmp_path / "dubbed.wav",
        duration=10.0,
    )

    command_text = " ".join(commands[0])
    assert "anullsrc=r=44100:cl=stereo" in command_text
    assert "-t 1.250 -i anullsrc=r=44100:cl=stereo" in command_text
    assert "-t 4.500 -i anullsrc=r=44100:cl=stereo" in command_text
    assert "concat=n=2:v=0:a=1" in command_text
    assert "atrim=0:10.0" in command_text
    assert "amix=inputs=3" in command_text


def test_tts_segment_failure_falls_back_to_silence(monkeypatch, tmp_path):
    from core import tts

    class FailingEngine(tts.TTSEngine):
        def __init__(self):
            self.calls = 0

        def generate(self, text, output_path, voice=None, speed=1.0, **kwargs):
            self.calls += 1
            Path(output_path).write_bytes(b"")
            raise RuntimeError("No audio was received")

        def get_voices(self):
            return []

        def get_supported_languages(self):
            return []

    engine = FailingEngine()
    monkeypatch.setattr(tts.time, "sleep", lambda _seconds: None)
    audio_files = tts.generate_subtitle_audio(
        [{"start": 1.0, "end": 2.5, "translated_text": "hello"}],
        tmp_path,
        engine,
    )

    assert engine.calls == tts.TTS_SEGMENT_RETRIES
    assert len(audio_files) == 1
    silence_path = audio_files[0][1]
    assert silence_path.name == "seg_0000_silence.wav"
    assert silence_path.exists()
    assert silence_path.stat().st_size > 44


def test_tts_retries_empty_audio_file(monkeypatch, tmp_path):
    from core import tts

    class EmptyThenOkEngine(tts.TTSEngine):
        def __init__(self):
            self.calls = 0

        def generate(self, text, output_path, voice=None, speed=1.0, **kwargs):
            self.calls += 1
            Path(output_path).write_bytes(b"" if self.calls == 1 else b"audio")
            return Path(output_path)

        def get_voices(self):
            return []

        def get_supported_languages(self):
            return []

    engine = EmptyThenOkEngine()
    monkeypatch.setattr(tts.time, "sleep", lambda _seconds: None)
    audio_files = tts.generate_subtitle_audio(
        [{"start": 0.0, "end": 1.0, "translated_text": "hello"}],
        tmp_path,
        engine,
    )

    assert engine.calls == 2
    assert audio_files[0][1].name == "seg_0000.wav"
    assert audio_files[0][1].read_bytes() == b"audio"


def test_tts_cache_reuses_only_matching_segment_metadata(monkeypatch, tmp_path):
    from core import tts

    class RecordingEngine(tts.TTSEngine):
        def __init__(self):
            self.calls = 0

        def generate(self, text, output_path, voice=None, speed=1.0, **kwargs):
            self.calls += 1
            Path(output_path).write_text(f"audio:{text}:{self.calls}", encoding="utf-8")
            return Path(output_path)

        def get_voices(self):
            return []

        def get_supported_languages(self):
            return []

    engine = RecordingEngine()
    monkeypatch.setattr(tts.time, "sleep", lambda _seconds: None)

    first = tts.generate_subtitle_audio(
        [{"start": 0.0, "end": 1.0, "translated_text": "first line"}],
        tmp_path,
        engine,
        voice="voice-a",
        target_lang="en",
    )
    assert engine.calls == 1
    assert first[0][1].read_text(encoding="utf-8") == "audio:first line:1"

    second = tts.generate_subtitle_audio(
        [{"start": 0.0, "end": 1.0, "translated_text": "first line"}],
        tmp_path,
        engine,
        voice="voice-a",
        target_lang="en",
    )
    assert engine.calls == 1
    assert second[0][1] == first[0][1]

    third = tts.generate_subtitle_audio(
        [{"start": 0.0, "end": 1.0, "translated_text": "different line"}],
        tmp_path,
        engine,
        voice="voice-a",
        target_lang="en",
    )
    assert engine.calls == 2
    assert third[0][1].read_text(encoding="utf-8") == "audio:different line:2"
    assert (tmp_path / "seg_0000.json").exists()


def test_pipeline_creates_unique_output_folder_per_video(monkeypatch, tmp_path):
    from core import pipeline

    class FixedDateTime:
        @classmethod
        def now(cls):
            class FixedNow:
                def strftime(self, _format):
                    return "20260614_153000"

            return FixedNow()

    monkeypatch.setattr(pipeline, "datetime", FixedDateTime)
    translator = pipeline.TranslationPipeline(api_key="demo", output_dir=str(tmp_path))

    first = translator._create_job_output_dir(Path("My <bad>|video?.mp4"))
    second = translator._create_job_output_dir(Path("My <bad>|video?.mp4"))

    assert first.exists()
    assert second.exists()
    assert first.parent == tmp_path
    assert second.parent == tmp_path
    assert first != second
    assert first.name.endswith("20260614_153000")
    assert second.name.endswith("20260614_153000_2")
    assert not any(char in first.name for char in '<>:"/\\|?*')


def test_pipeline_parses_srt_for_retranslate(tmp_path):
    from core import pipeline

    srt = tmp_path / "original.srt"
    srt.write_text(
        "1\n00:00:01,250 --> 00:00:02,500\nHello world\n\n"
        "2\n00:00:03,000 --> 00:00:04,125\nSecond line\n",
        encoding="utf-8",
    )

    segments = pipeline._parse_srt_segments(srt)

    assert segments == [
        {"start": 1.25, "end": 2.5, "text": "Hello world"},
        {"start": 3.0, "end": 4.125, "text": "Second line"},
    ]


def test_tts_mix_speeds_up_overlong_segments(monkeypatch, tmp_path):
    from core import pipeline

    commands = []

    def fake_run(cmd, **kwargs):
        if cmd[0] == "ffprobe":
            class ProbeResult:
                stdout = "4.0"

            return ProbeResult()
        commands.append(cmd)

        class Result:
            stderr = ""

        return Result()

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    audio = tmp_path / "long.wav"
    audio.write_bytes(b"fake")

    pipeline._mix_timed_audio_segments(
        [({"start": 2.0, "end": 4.0}, audio)],
        tmp_path / "dubbed.wav",
        duration=8.0,
    )

    command_text = " ".join(commands[0])
    # A-tier alignment: a 4.0s clip in a 2.0s slot first borrows 1.2s from the
    # 2.0s leading gap (0.6 ratio), leaving a 3.2s slot, then applies a capped
    # atempo (4.0/3.2 = 1.25, well under MAX_SPEED=1.30). The clip must NEVER be
    # sped up to the old uncapped 2.0x.
    assert "atempo=1.2500" in command_text
    assert "atempo=2.0000" not in command_text
    assert "atrim=0:3.200" in command_text
    assert "-t 0.800 -i anullsrc=r=44100:cl=stereo" in command_text
    assert "concat=n=2:v=0:a=1" in command_text


def test_tts_mix_splits_batches_when_command_too_long(monkeypatch, tmp_path):
    from core import pipeline

    ffmpeg_commands = []
    created_files = []

    def fake_run(cmd, **kwargs):
        if cmd[0] == "ffprobe":
            class ProbeResult:
                stdout = "0.5"
            return ProbeResult()
        ffmpeg_commands.append(cmd)
        # Simulate ffmpeg writing each requested output file.
        out = cmd[-1]
        Path(out).write_bytes(b"audio")
        created_files.append(Path(out))
        class Result:
            stderr = ""
            stdout = ""
        return Result()

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    # Force a tiny limit so even a handful of segments split into batches.
    monkeypatch.setattr(pipeline, "MAX_MIX_COMMAND_CHARS", 800)

    segments = []
    for i in range(6):
        audio = tmp_path / f"seg_{i:04d}.mp3"
        audio.write_bytes(b"fake")
        segments.append(({"start": float(i), "end": float(i + 1)}, audio))

    output = tmp_path / "dubbed.wav"
    pipeline._mix_timed_audio_segments(segments, output, duration=10.0)

    # Multiple batch commands plus one final sum command.
    assert len(ffmpeg_commands) >= 3
    sum_cmd = ffmpeg_commands[-1]
    assert "amix=inputs=" in " ".join(sum_cmd)
    # Final output must be written and per-batch parts cleaned up.
    assert output.exists()
    mix_dir = tmp_path / "dubbed_mix_parts"
    assert not mix_dir.exists()


def test_separate_speech_requires_demucs(monkeypatch, tmp_path):
    from core import video_processor

    monkeypatch.setattr(video_processor.shutil, "which", lambda _name: None)
    monkeypatch.setattr(video_processor.importlib.util, "find_spec", lambda _name: None)

    with pytest.raises(RuntimeError, match="Demucs"):
        video_processor.separate_speech_and_background(tmp_path / "audio.wav", tmp_path / "out")


def test_demucs_helpers_use_current_python(monkeypatch):
    from core import video_processor

    monkeypatch.setattr(video_processor.shutil, "which", lambda _name: None)
    monkeypatch.setattr(video_processor.importlib.util, "find_spec", lambda _name: None)

    assert not video_processor.is_demucs_available()
    assert video_processor.demucs_install_command()[:3] == [
        video_processor.sys.executable,
        "-m",
        "pip",
    ]


def test_demucs_command_uses_python_module_when_available(monkeypatch):
    from core import video_processor

    monkeypatch.setattr(video_processor.shutil, "which", lambda _name: None)
    monkeypatch.setattr(video_processor.importlib.util, "find_spec", lambda _name: object())

    assert video_processor._demucs_command() == [video_processor.sys.executable, "-m", "demucs"]


def test_separate_speech_uses_demucs_mp3_output(monkeypatch, tmp_path):
    from core import video_processor

    commands = []
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"fake")

    def fake_run_process(cmd, check=False):
        commands.append(cmd)
        stem_dir = tmp_path / "out" / "htdemucs" / "input"
        stem_dir.mkdir(parents=True)
        (stem_dir / "vocals.mp3").write_bytes(b"vocals")
        (stem_dir / "no_vocals.mp3").write_bytes(b"background")

        class Result:
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(video_processor, "_demucs_command", lambda: ["demucs"])
    monkeypatch.setattr(video_processor, "_run_process", fake_run_process)

    vocals, background = video_processor.separate_speech_and_background(audio, tmp_path / "out")

    assert "--mp3" in commands[0]
    assert "--mp3-bitrate" in commands[0]
    assert vocals.name == "vocals.mp3"
    assert background.name == "no_vocals.mp3"


def test_separate_speech_retries_empty_demucs_output(monkeypatch, tmp_path):
    from core import video_processor

    calls = {"count": 0}
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"fake")

    def fake_run_process(cmd, check=False):
        calls["count"] += 1
        stem_dir = tmp_path / "out" / "htdemucs" / "input"
        stem_dir.mkdir(parents=True, exist_ok=True)
        if calls["count"] == 1:
            (stem_dir / "vocals.mp3").write_bytes(b"")
            (stem_dir / "no_vocals.mp3").write_bytes(b"")
        else:
            (stem_dir / "vocals.mp3").write_bytes(b"vocals")
            (stem_dir / "no_vocals.mp3").write_bytes(b"background")

        class Result:
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(video_processor, "_demucs_command", lambda: ["demucs"])
    monkeypatch.setattr(video_processor, "_run_process", fake_run_process)
    monkeypatch.setattr(video_processor.time, "sleep", lambda _seconds: None)

    vocals, background = video_processor.separate_speech_and_background(audio, tmp_path / "out")

    assert calls["count"] == 2
    assert vocals.read_bytes() == b"vocals"
    assert background.read_bytes() == b"background"


def test_ffmpeg_output_retries_empty_file(monkeypatch, tmp_path):
    from core import video_processor

    calls = {"count": 0}
    output = tmp_path / "out.wav"

    def fake_run_ffmpeg(cmd, process_runner=None):
        calls["count"] += 1
        output.write_bytes(b"" if calls["count"] == 1 else b"audio")

    monkeypatch.setattr(video_processor, "_run_ffmpeg", fake_run_ffmpeg)
    monkeypatch.setattr(video_processor.time, "sleep", lambda _seconds: None)

    video_processor._run_ffmpeg_for_output(["ffmpeg"], output)

    assert calls["count"] == 2
    assert output.read_bytes() == b"audio"


def test_separate_speech_reports_demucs_stderr(monkeypatch, tmp_path):
    from core import video_processor

    def fake_run_process(cmd, check=False):
        raise subprocess.CalledProcessError(
            1,
            cmd,
            output="stdout line",
            stderr="missing codec detail",
        )

    monkeypatch.setattr(video_processor, "_demucs_command", lambda: ["demucs"])
    monkeypatch.setattr(video_processor, "_run_process", fake_run_process)

    with pytest.raises(RuntimeError, match="missing codec detail"):
        video_processor.separate_speech_and_background(tmp_path / "input.wav", tmp_path / "out")


def test_mix_background_with_dubbing_uses_both_tracks(monkeypatch, tmp_path):
    from core import video_processor

    commands = []

    def fake_run_ffmpeg(cmd):
        commands.append(cmd)
        (tmp_path / "mixed.wav").write_bytes(b"mixed")

    monkeypatch.setattr(video_processor, "_run_ffmpeg", fake_run_ffmpeg)

    video_processor.mix_background_with_dubbing(
        tmp_path / "background.wav",
        tmp_path / "dubbed.wav",
        tmp_path / "mixed.wav",
        duration=12.0,
    )

    command_text = " ".join(commands[0])
    assert str(tmp_path / "background.wav") in command_text
    assert str(tmp_path / "dubbed.wav") in command_text
    assert "amix=inputs=2" in command_text
    assert "atrim=0:12.0" in command_text


def test_generate_dubbing_replaces_video_with_mixed_audio(monkeypatch, tmp_path):
    from core import pipeline

    calls = {}
    translator = pipeline.TranslationPipeline(api_key="demo", output_dir=str(tmp_path), temp_dir=str(tmp_path / "temp"))
    video = tmp_path / "video.mp4"
    background = tmp_path / "background.wav"
    segment_audio = tmp_path / "segment.wav"
    video.write_bytes(b"fake")
    background.write_bytes(b"fake")
    segment_audio.write_bytes(b"fake")

    class DummyEngine:
        pass

    monkeypatch.setattr(pipeline.TTSEngine, "create", lambda *args, **kwargs: DummyEngine())
    monkeypatch.setattr(
        pipeline,
        "generate_subtitle_audio",
        lambda *args, **kwargs: [({"start": 0.0, "end": 1.0, "translated_text": "hello"}, segment_audio)],
    )

    def fake_timed_mix(_audio_files, output_path, duration):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"dubbed")

    def fake_mix_background(bg, dubbing, output_path, duration):
        calls["background"] = bg
        calls["dubbing"] = dubbing
        calls["mixed"] = output_path
        output_path.write_bytes(b"mixed")

    def fake_replace(video_path, audio_path, output_path):
        calls["replace_audio"] = audio_path
        output_path.write_bytes(b"video")
        return output_path

    monkeypatch.setattr(pipeline, "_mix_timed_audio_segments", fake_timed_mix)
    monkeypatch.setattr(pipeline, "mix_background_with_dubbing", fake_mix_background)
    monkeypatch.setattr(pipeline, "replace_audio", fake_replace)

    result = translator._generate_dubbing(
        translated_segments=[{"start": 0.0, "end": 1.0, "translated_text": "hello"}],
        video_path=video,
        tts_voice="voice",
        tts_reference_audio=None,
        tts_emotion="natural",
        tts_timing_strategy="strict_subtitle_duration",
        background_audio=background,
        output_dir=tmp_path,
        results={"video_info": {"duration": 10.0}},
    )

    assert result == tmp_path / "video_dubbed.mp4"
    assert calls["background"] == background
    assert calls["replace_audio"] == tmp_path / "video_mixed_audio.wav"
    assert calls["replace_audio"] != tmp_path / "video_dubbed.wav"


def test_generate_dubbing_uses_job_scoped_tts_cache_and_target_lang(monkeypatch, tmp_path):
    from core import pipeline

    captured = {}
    translator = pipeline.TranslationPipeline(api_key="demo", output_dir=str(tmp_path), temp_dir=str(tmp_path / "temp"))
    video = tmp_path / "video.mp4"
    background = tmp_path / "background.wav"
    segment_audio = tmp_path / "segment.wav"
    video.write_bytes(b"fake")
    background.write_bytes(b"fake")
    segment_audio.write_bytes(b"fake")

    class DummyEngine:
        pass

    monkeypatch.setattr(pipeline.TTSEngine, "create", lambda *args, **kwargs: DummyEngine())

    def fake_generate_subtitle_audio(segments, output_path, engine, **kwargs):
        captured["segments"] = segments
        captured["output_path"] = Path(output_path)
        captured["target_lang"] = kwargs.get("target_lang")
        return [({"start": 0.0, "end": 1.0, "translated_text": "hello"}, segment_audio)]

    monkeypatch.setattr(pipeline, "generate_subtitle_audio", fake_generate_subtitle_audio)

    def fake_timed_mix(_audio_files, output_path, duration):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"dubbed")

    monkeypatch.setattr(pipeline, "_mix_timed_audio_segments", fake_timed_mix)
    monkeypatch.setattr(pipeline, "mix_background_with_dubbing", lambda bg, dubbing, output_path, duration: output_path.write_bytes(b"mixed"))
    monkeypatch.setattr(pipeline, "replace_audio", lambda video_path, audio_path, output_path: output_path.write_bytes(b"video") or output_path)

    translator._generate_dubbing(
        translated_segments=[{"start": 0.0, "end": 1.0, "translated_text": "hello"}],
        video_path=video,
        tts_voice="voice",
        tts_reference_audio=None,
        tts_emotion="natural",
        tts_timing_strategy="strict_subtitle_duration",
        background_audio=background,
        output_dir=tmp_path,
        results={"video_info": {"duration": 10.0}},
        target_lang="en",
    )

    assert captured["output_path"] == tmp_path / "_tts_segments"
    assert captured["target_lang"] == "en"


def test_pipeline_resume_reuses_completed_transcribe_and_translation(monkeypatch, tmp_path):
    from core import pipeline

    input_video = tmp_path / "video.mp4"
    input_video.write_bytes(b"video")
    job_dir = tmp_path / "video_20260618_120000"
    job_dir.mkdir()
    original_srt = job_dir / "video_original.srt"
    original_srt.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nhello\n",
        encoding="utf-8",
    )
    translated_json = job_dir / pipeline.TRANSLATED_SEGMENTS_JSON_NAME
    translated_json.write_text(
        json.dumps([{"start": 0.0, "end": 1.0, "text": "hello", "translated_text": "你好"}], ensure_ascii=False),
        encoding="utf-8",
    )
    manifest_path = job_dir / pipeline.JOB_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps({
            "schema": 1,
            "status": "running",
            "stage": "translated",
            "input_path": str(input_video),
            "input_video": str(input_video),
            "output_dir": str(job_dir),
            "settings": {
                "target_lang": "zh",
                "source_lang": "en",
                "subtitle_mode": "soft",
                "tts_enabled": False,
                "process_mode": "full",
            },
            "results": {
                "input_video": str(input_video),
                "output_dir": str(job_dir),
                "video_info": {"duration": 1.0},
                "original_audio": str(job_dir / "video_original_audio.wav"),
                "asr_audio": str(job_dir / "video_speech_for_asr.wav"),
                "original_subtitle": str(original_srt),
                "translated_segments": str(translated_json),
                "source_lang": "en",
            },
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    (job_dir / "video_original_audio.wav").write_bytes(b"audio")
    (job_dir / "video_speech_for_asr.wav").write_bytes(b"asr")

    def forbidden(*args, **kwargs):
        raise AssertionError("resume should not call completed expensive stages")

    monkeypatch.setattr(pipeline, "transcribe", forbidden)
    monkeypatch.setattr(pipeline, "translate_segments", forbidden)

    translator = pipeline.TranslationPipeline(api_key="real-key", output_dir=str(tmp_path), temp_dir=str(tmp_path / "temp"))
    results = translator.resume(manifest_path)

    assert results["soft_subtitle"] == str(job_dir / "video_translated.srt")
    assert Path(results["soft_subtitle"]).exists()
    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved_manifest["status"] == "completed"
    assert saved_manifest["stage"] == "completed"


def test_discover_resumable_jobs_ignores_completed(tmp_path):
    from core import pipeline

    active_dir = tmp_path / "active"
    done_dir = tmp_path / "done"
    active_dir.mkdir()
    done_dir.mkdir()
    (active_dir / pipeline.JOB_MANIFEST_NAME).write_text(
        json.dumps({"status": "cancelled", "updated_at": "2026-06-18T01:00:00", "input_path": "a.mp4"}),
        encoding="utf-8",
    )
    (done_dir / pipeline.JOB_MANIFEST_NAME).write_text(
        json.dumps({"status": "completed", "updated_at": "2026-06-18T02:00:00", "input_path": "b.mp4"}),
        encoding="utf-8",
    )

    jobs = pipeline.discover_resumable_jobs(tmp_path)

    assert len(jobs) == 1
    assert jobs[0]["input_path"] == "a.mp4"
    assert jobs[0]["manifest_path"] == str(active_dir / pipeline.JOB_MANIFEST_NAME)


def test_generate_dubbing_reads_missing_video_info_with_runner(monkeypatch, tmp_path):
    from core import pipeline

    calls = {}
    translator = pipeline.TranslationPipeline(api_key="demo", output_dir=str(tmp_path), temp_dir=str(tmp_path / "temp"))
    video = tmp_path / "video.mp4"
    background = tmp_path / "background.wav"
    segment_audio = tmp_path / "segment.wav"
    video.write_bytes(b"fake")
    background.write_bytes(b"fake")
    segment_audio.write_bytes(b"fake")

    class DummyEngine:
        pass

    monkeypatch.setattr(pipeline.TTSEngine, "create", lambda *args, **kwargs: DummyEngine())
    monkeypatch.setattr(
        pipeline,
        "generate_subtitle_audio",
        lambda *args, **kwargs: [({"start": 0.0, "end": 1.0, "translated_text": "hello"}, segment_audio)],
    )

    def fake_get_video_info(video_path, process_runner=None):
        calls["process_runner"] = process_runner
        return {"duration": 10.0}

    monkeypatch.setattr(pipeline, "get_video_info", fake_get_video_info)
    monkeypatch.setattr(pipeline, "_call_with_process_runner", lambda func, runner, *args, **kwargs: func(*args, **kwargs))
    def fake_timed_mix(_audio_files, output_path, duration):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"dubbed")

    monkeypatch.setattr(pipeline, "_mix_timed_audio_segments", fake_timed_mix)
    monkeypatch.setattr(pipeline, "mix_background_with_dubbing", lambda bg, dubbing, output_path, duration: output_path.write_bytes(b"mixed"))
    monkeypatch.setattr(pipeline, "replace_audio", lambda video_path, audio_path, output_path: output_path.write_bytes(b"video") or output_path)

    translator._generate_dubbing(
        translated_segments=[{"start": 0.0, "end": 1.0, "translated_text": "hello"}],
        video_path=video,
        tts_voice="voice",
        tts_reference_audio=None,
        tts_emotion="natural",
        tts_timing_strategy="strict_subtitle_duration",
        background_audio=background,
        output_dir=tmp_path,
        results={"video_info": {}},
    )

    assert calls["process_runner"] == translator._run_process


def test_replace_audio_does_not_cut_video(monkeypatch, tmp_path):
    from core import video_processor

    commands = []

    def fake_run_ffmpeg(cmd):
        commands.append(cmd)
        (tmp_path / "out.mp4").write_bytes(b"video")

    monkeypatch.setattr(video_processor, "_run_ffmpeg", fake_run_ffmpeg)

    video_processor.replace_audio(tmp_path / "video.mp4", tmp_path / "dubbed.wav", tmp_path / "out.mp4")

    assert "-shortest" not in commands[0]
