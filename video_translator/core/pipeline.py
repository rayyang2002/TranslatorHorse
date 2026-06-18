"""End-to-end video translation pipeline."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from inspect import signature
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from core.asr import save_srt, transcribe
from core.subtitle_renderer import generate_bilingual_subtitle, generate_soft_subtitle, render_hard_subtitle
from core.translator import translate_segments
from core.tts import TTSEngine, generate_subtitle_audio
from core.tts_aligner import (
    MAX_SPEED as ALIGN_MAX_SPEED,
    MIN_SPEED as ALIGN_MIN_SPEED,
    GAP_BORROW_RATIO as ALIGN_GAP_BORROW_RATIO,
    SegmentPlan,
    merge_short_segments,
    plan_final_placement,
)
from core.video_processor import (
    convert_audio,
    extract_audio,
    get_video_info,
    mix_background_with_dubbing,
    replace_audio,
    separate_speech_and_background,
)
from utils.downloader import download_video, is_url
from utils.helpers import ensure_dir
from utils.logger import logger


FFPROBE_TIMEOUT_SECONDS = 20
PROCESS_POLL_INTERVAL_SECONDS = 0.2
PROCESS_RETRY_DELAY_SECONDS = 2
ASR_RETRIES = 2
ASR_RETRY_DELAY_SECONDS = 2
MAX_MIX_COMMAND_CHARS = 14000
JOB_MANIFEST_NAME = ".video_translator_job.json"
TRANSLATED_SEGMENTS_JSON_NAME = "_translated_segments.json"
# Upper bound for a single ffmpeg dubbing-mix command line. Windows
# CreateProcess rejects command lines longer than 32,767 characters
# (WinError 206 "文件名或扩展名太长"). A 13-minute video can produce 500+
# subtitle segments, and a single -filter_complex graph for all of them
# blows past that limit, so the mix is split into bounded batches.


class TranslationPipeline:
    """Coordinates media extraction, ASR, translation, subtitles, TTS, and render."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        whisper_model: str = "base",
        tts_engine: str = "edge-tts",
        tts_config: Optional[dict] = None,
        output_dir: str = "./output",
        temp_dir: str = "./temp",
        on_progress: Optional[Callable[[str, float], None]] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.whisper_model = whisper_model
        self.tts_engine_name = tts_engine
        self.tts_config = tts_config or {}
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.on_progress = on_progress
        self._cancelled = False
        self._process_lock = threading.Lock()
        self._current_process: subprocess.Popen | None = None
        self._manifest_path: Path | None = None
        self._manifest: dict = {}

    def cancel(self):
        self._cancelled = True
        self._terminate_current_process()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def process(
        self,
        input_path: str,
        target_lang: str = "zh",
        source_lang: str = "auto",
        subtitle_mode: str = "both",
        tts_voice: Optional[str] = None,
        tts_enabled: bool = False,
        tts_reference_audio: Optional[str] = None,
        tts_emotion: str = "natural",
        tts_timing_strategy: str = "natural_with_light_stretch",
        process_mode: str = "full",
        original_subtitle_path: Optional[str] = None,
        resume_manifest_path: Optional[str | Path] = None,
    ) -> dict:
        """Run the full processing pipeline and return generated artifact paths."""
        ensure_dir(self.output_dir)
        ensure_dir(self.temp_dir)
        self._cancelled = False
        resume_manifest = self._load_resume_manifest(resume_manifest_path)
        if resume_manifest:
            saved_settings = resume_manifest.get("settings", {})
            target_lang = saved_settings.get("target_lang", target_lang)
            source_lang = saved_settings.get("source_lang", source_lang)
            subtitle_mode = saved_settings.get("subtitle_mode", subtitle_mode)
            tts_voice = saved_settings.get("tts_voice", tts_voice)
            tts_enabled = saved_settings.get("tts_enabled", tts_enabled)
            tts_reference_audio = saved_settings.get("tts_reference_audio", tts_reference_audio)
            tts_emotion = saved_settings.get("tts_emotion", tts_emotion)
            tts_timing_strategy = saved_settings.get("tts_timing_strategy", tts_timing_strategy)
            process_mode = saved_settings.get("process_mode", process_mode)
            original_subtitle_path = saved_settings.get("original_subtitle_path", original_subtitle_path)
            input_path = resume_manifest.get("input_path") or input_path

        results: dict = dict(resume_manifest.get("results", {}) if resume_manifest else {})

        if process_mode == "dubbing":
            tts_enabled = True
            subtitle_mode = "none"
        elif process_mode in {"transcribe", "retranslate"}:
            tts_enabled = False

        if resume_manifest:
            job_output_dir = Path(resume_manifest["output_dir"])
            video_path = Path(results.get("input_video") or resume_manifest.get("input_video") or input_path)
            if not video_path.exists() and not is_url(str(input_path)):
                raise FileNotFoundError(f"视频文件不存在: {video_path}")
        else:
            video_path = self._resolve_input(input_path)
            job_output_dir = self._create_job_output_dir(video_path)

        self._begin_manifest(
            manifest_path=job_output_dir / JOB_MANIFEST_NAME,
            input_path=input_path,
            video_path=video_path,
            output_dir=job_output_dir,
            settings={
                "target_lang": target_lang,
                "source_lang": source_lang,
                "subtitle_mode": subtitle_mode,
                "tts_enabled": tts_enabled,
                "tts_voice": tts_voice,
                "tts_reference_audio": tts_reference_audio,
                "tts_emotion": tts_emotion,
                "tts_timing_strategy": tts_timing_strategy,
                "process_mode": process_mode,
                "original_subtitle_path": original_subtitle_path,
                "whisper_model": self.whisper_model,
                "tts_engine": self.tts_engine_name,
            },
            existing=resume_manifest,
        )
        results["input_video"] = str(video_path)
        results["output_dir"] = str(job_output_dir)
        self._save_manifest(results=results, stage="started", status="running")

        try:
            self._check_cancelled()

            self._progress("读取媒体信息...", 5)
            if not results.get("video_info"):
                try:
                    results["video_info"] = get_video_info(video_path, process_runner=self._run_process)
                except Exception as exc:
                    logger.warning("Could not read video info: %s", exc)
                    results["video_info"] = {}
                self._save_manifest(results=results, stage="video_info", status="running")
            else:
                self._progress("已恢复媒体信息", 5)
            self._check_cancelled()

            if process_mode == "retranslate" and original_subtitle_path:
                srt_path = Path(original_subtitle_path)
                if not srt_path.exists():
                    raise FileNotFoundError(f"原文字幕不存在: {srt_path}")

                self._progress("读取原文字幕...", 15)
                segments = _parse_srt_segments(srt_path)
                if not segments:
                    raise RuntimeError("原文字幕为空，无法重新翻译。")
                results["original_subtitle"] = str(srt_path)
                self._save_manifest(results=results, stage="original_subtitle", status="running")
                self._check_cancelled()
                final_results = self._translate_and_render(
                    segments=segments,
                    video_path=video_path,
                    target_lang=target_lang,
                    source_lang=source_lang,
                    subtitle_mode=subtitle_mode,
                    tts_enabled=False,
                    tts_voice=tts_voice,
                    tts_reference_audio=tts_reference_audio,
                    tts_emotion=tts_emotion,
                    tts_timing_strategy=tts_timing_strategy,
                    background_audio=None,
                    output_dir=job_output_dir,
                    results=results,
                )
                self._save_manifest(results=final_results, stage="completed", status="completed")
                return final_results

            original_audio_path = job_output_dir / f"{video_path.stem}_original_audio.wav"
            if _valid_resume_file(results.get("original_audio")):
                original_audio_path = Path(results["original_audio"])
                self._progress("已恢复原始音轨", 10)
            else:
                self._progress("提取原始音轨...", 10)
                extract_audio(video_path, original_audio_path, sample_rate=44100, channels=2, process_runner=self._run_process)
                results["original_audio"] = str(original_audio_path)
                self._save_manifest(results=results, stage="original_audio", status="running")
            self._check_cancelled()

            speech_audio_path = Path(results.get("asr_audio") or job_output_dir / f"{video_path.stem}_speech_for_asr.wav")
            background_audio_path = Path(results.get("background_audio") or original_audio_path)
            if tts_enabled:
                if (
                    _valid_resume_file(results.get("speech_audio"))
                    and _valid_resume_file(results.get("background_audio"))
                    and _valid_resume_file(results.get("asr_audio"))
                ):
                    self._progress("已恢复人声分离结果", 15)
                    speech_audio_path = Path(results["asr_audio"])
                    background_audio_path = Path(results["background_audio"])
                else:
                    self._progress("分离人声和背景音乐...", 15)
                    speech_source, background_audio_path = separate_speech_and_background(
                        original_audio_path,
                        job_output_dir / "separated",
                        process_runner=self._run_process,
                    )
                    speech_audio_path = job_output_dir / f"{video_path.stem}_speech_for_asr.wav"
                    convert_audio(speech_source, speech_audio_path, sample_rate=16000, channels=1, process_runner=self._run_process)
                    results["speech_audio"] = str(speech_source)
                    results["background_audio"] = str(background_audio_path)
                    results["asr_audio"] = str(speech_audio_path)
                    self._save_manifest(results=results, stage="speech_separated", status="running")
                self._check_cancelled()
            else:
                if _valid_resume_file(results.get("asr_audio")):
                    speech_audio_path = Path(results["asr_audio"])
                    self._progress("已恢复识别音频", 15)
                else:
                    speech_audio_path = job_output_dir / f"{video_path.stem}_speech_for_asr.wav"
                    convert_audio(original_audio_path, speech_audio_path, sample_rate=16000, channels=1, process_runner=self._run_process)
                    results["speech_audio"] = str(speech_audio_path)
                    results["asr_audio"] = str(speech_audio_path)
                    self._save_manifest(results=results, stage="asr_audio", status="running")
                self._check_cancelled()

            raw_srt = job_output_dir / f"{video_path.stem}_original.srt"
            if _valid_resume_file(results.get("original_subtitle")):
                self._progress("已恢复原文字幕", 20)
                raw_srt = Path(results["original_subtitle"])
                segments = _parse_srt_segments(raw_srt)
                detected_lang = results.get("source_lang", source_lang if source_lang != "auto" else "unknown")
            else:
                self._progress("语音识别中...", 20)
                lang_param = None if source_lang == "auto" else source_lang
                asr_result = self._transcribe_with_retries(speech_audio_path, language=lang_param)
                detected_lang = asr_result["language"]
                segments = asr_result["segments"]
                save_srt(segments, raw_srt)
                results["source_lang"] = detected_lang
                results["original_subtitle"] = str(raw_srt)
                self._save_manifest(results=results, stage="transcribed", status="running")
            source_lang = detected_lang if source_lang == "auto" else source_lang
            results["source_lang"] = detected_lang
            self._check_cancelled()

            if process_mode == "transcribe":
                self._progress("转录完成", 100)
                self._save_manifest(results=results, stage="completed", status="completed")
                return results

            final_results = self._translate_and_render(
                segments=segments,
                video_path=video_path,
                target_lang=target_lang,
                source_lang=source_lang,
                subtitle_mode=subtitle_mode,
                tts_enabled=tts_enabled,
                tts_voice=tts_voice,
                tts_reference_audio=tts_reference_audio,
                tts_emotion=tts_emotion,
                tts_timing_strategy=tts_timing_strategy,
                background_audio=background_audio_path,
                output_dir=job_output_dir,
                results=results,
            )
            self._save_manifest(results=final_results, stage="completed", status="completed")
            return final_results
        except InterruptedError:
            self._save_manifest(results=results, stage=self._manifest.get("stage", "cancelled"), status="cancelled")
            raise
        except Exception as exc:
            self._save_manifest(
                results=results,
                stage=self._manifest.get("stage", "failed"),
                status="failed",
                error=str(exc),
            )
            raise

    def resume(self, manifest_path: str | Path) -> dict:
        manifest = self._load_resume_manifest(manifest_path)
        if not manifest:
            raise FileNotFoundError(f"断点任务不存在: {manifest_path}")
        return self.process(
            manifest.get("input_path") or manifest.get("input_video") or "",
            resume_manifest_path=manifest_path,
        )

    def _load_resume_manifest(self, manifest_path: Optional[str | Path]) -> dict:
        if not manifest_path:
            return {}
        path = Path(manifest_path)
        if path.is_dir():
            path = path / JOB_MANIFEST_NAME
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        data["manifest_path"] = str(path)
        return data

    def _begin_manifest(
        self,
        *,
        manifest_path: Path,
        input_path: str,
        video_path: Path,
        output_dir: Path,
        settings: dict,
        existing: dict | None = None,
    ) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        self._manifest_path = manifest_path
        self._manifest = dict(existing or {})
        self._manifest.update({
            "schema": 1,
            "status": "running",
            "input_path": str(input_path),
            "input_video": str(video_path),
            "output_dir": str(output_dir),
            "settings": settings,
            "updated_at": now,
        })
        self._manifest.setdefault("created_at", now)
        self._manifest.setdefault("results", {})
        self._write_manifest()

    def _save_manifest(
        self,
        *,
        results: dict,
        stage: str,
        status: str,
        error: str | None = None,
    ) -> None:
        if self._manifest_path is None:
            return
        self._manifest["results"] = dict(results)
        self._manifest["stage"] = stage
        self._manifest["status"] = status
        self._manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
        if error:
            self._manifest["error"] = error
        elif status != "failed":
            self._manifest.pop("error", None)
        self._write_manifest()

    def _write_manifest(self) -> None:
        if self._manifest_path is None:
            return
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._manifest_path.with_suffix(self._manifest_path.suffix + ".tmp")
        temp_path.write_text(json.dumps(self._manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self._manifest_path)

    def _translate_and_render(
        self,
        segments: list[dict],
        video_path: Path,
        target_lang: str,
        source_lang: str,
        subtitle_mode: str,
        tts_enabled: bool,
        tts_voice: Optional[str],
        tts_reference_audio: Optional[str],
        tts_emotion: str,
        tts_timing_strategy: str,
        background_audio: Optional[Path],
        output_dir: Path,
        results: dict,
    ) -> dict:

        translated_segments_path = output_dir / TRANSLATED_SEGMENTS_JSON_NAME
        if _valid_resume_file(translated_segments_path):
            self._progress("已恢复翻译字幕", 40)
            translated_segments = json.loads(translated_segments_path.read_text(encoding="utf-8"))
        else:
            self._progress("翻译字幕中...", 40)
            translated_segments = translate_segments(
                segments,
                target_lang=target_lang,
                source_lang=source_lang,
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
                cancel_check=self._check_cancelled,
            )
            translated_segments_path.write_text(
                json.dumps(translated_segments, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            results["translated_segments"] = str(translated_segments_path)
            self._save_manifest(results=results, stage="translated", status="running")
            self._check_cancelled()

        self._progress("生成译文字幕...", 60)
        translated_srt_for_render = output_dir / "_translated_for_render.srt"
        if not _valid_resume_file(translated_srt_for_render):
            generate_soft_subtitle(translated_segments, translated_srt_for_render)
        results["translated_render_subtitle"] = str(translated_srt_for_render)

        if subtitle_mode in ("soft", "both"):
            soft_srt = output_dir / f"{video_path.stem}_translated.srt"
            if not _valid_resume_file(soft_srt):
                generate_soft_subtitle(translated_segments, soft_srt)
            results["soft_subtitle"] = str(soft_srt)

            bilingual_srt = output_dir / f"{video_path.stem}_bilingual.srt"
            if not _valid_resume_file(bilingual_srt):
                generate_bilingual_subtitle(translated_segments, bilingual_srt)
            results["bilingual_subtitle"] = str(bilingual_srt)
            self._save_manifest(results=results, stage="subtitles", status="running")

        render_input = video_path
        if tts_enabled:
            render_input = self._generate_dubbing(
                translated_segments=translated_segments,
                video_path=video_path,
                target_lang=target_lang,
                tts_voice=tts_voice,
                tts_reference_audio=tts_reference_audio,
                tts_emotion=tts_emotion,
                tts_timing_strategy=tts_timing_strategy,
                background_audio=background_audio or Path(results["original_audio"]),
                output_dir=output_dir,
                results=results,
            )
            self._check_cancelled()

        if subtitle_mode in ("hard", "both"):
            self._progress("烧录字幕到视频...", 90)
            hard_video = output_dir / f"{video_path.stem}_hard_sub.mp4"
            if not _valid_resume_file(hard_video):
                _call_with_process_runner(
                    render_hard_subtitle,
                    self._run_process,
                    render_input,
                    translated_srt_for_render,
                    hard_video,
                )
            results["hard_subtitle_video"] = str(hard_video)
            self._save_manifest(results=results, stage="hard_subtitle", status="running")

        self._progress("处理完成", 100)
        return results

    def _resolve_input(self, input_path: str) -> Path:
        if is_url(input_path):
            self._progress("下载视频中...", 0)
            downloaded = download_video(input_path, self.temp_dir)
            if downloaded is None:
                raise RuntimeError("视频下载失败")
            return downloaded

        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"视频文件不存在: {path}")
        return path

    def _create_job_output_dir(self, video_path: Path) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_stem = _safe_path_name(video_path.stem) or "video"
        base_dir = self.output_dir / f"{safe_stem}_{timestamp}"
        output_dir = base_dir
        counter = 2
        while output_dir.exists():
            output_dir = self.output_dir / f"{safe_stem}_{timestamp}_{counter}"
            counter += 1
        ensure_dir(output_dir)
        return output_dir

    def _generate_dubbing(
        self,
        translated_segments: list[dict],
        video_path: Path,
        tts_voice: Optional[str],
        tts_reference_audio: Optional[str],
        tts_emotion: str,
        tts_timing_strategy: str,
        background_audio: Path,
        output_dir: Path,
        results: dict,
        target_lang: Optional[str] = None,
    ) -> Path:
        dubbed_video = output_dir / f"{video_path.stem}_dubbed.mp4"
        if _valid_resume_file(results.get("dubbed_video")):
            self._progress("已恢复配音视频", 85)
            return Path(results["dubbed_video"])
        if _valid_resume_file(dubbed_video):
            results["dubbed_video"] = str(dubbed_video)
            self._progress("已恢复配音视频", 85)
            self._save_manifest(results=results, stage="dubbed_video", status="running")
            return dubbed_video

        self._progress("生成配音...", 65)
        engine = TTSEngine.create(self.tts_engine_name, **self.tts_config)

        # Build the dubbing timeline. Short Whisper fragments (common with the
        # ``base`` model) are merged so they are not impossible slots for any
        # target language. This only affects dubbing; rendered subtitles keep
        # their original segmentation.
        dubbing_segments = merge_short_segments(translated_segments)

        # Keep TTS cache scoped to the current job. A global temp/tts_segments
        # directory can poison later videos because segment filenames are just
        # ordinal labels (seg_0000.mp3, seg_0001.mp3, ...). Per-job storage plus
        # metadata validation in core.tts prevents old spoken text from being
        # reused against a new subtitle timeline.
        audio_dir = output_dir / "_tts_segments"
        audio_files = generate_subtitle_audio(
            dubbing_segments,
            audio_dir,
            engine,
            voice=tts_voice,
            reference_audio=tts_reference_audio,
            emotion=tts_emotion,
            timing_strategy=tts_timing_strategy,
            cancel_check=self._check_cancelled,
            target_lang=target_lang,
        )
        self._check_cancelled()

        if not audio_files:
            raise RuntimeError("没有生成任何配音片段")

        video_info = results.get("video_info") or get_video_info(video_path, process_runner=self._run_process)
        video_duration = float(video_info.get("duration", 0) or 0)
        dubbed_audio = output_dir / f"{video_path.stem}_dubbed.wav"
        if _valid_resume_file(results.get("dubbed_audio")):
            dubbed_audio = Path(results["dubbed_audio"])
            self._progress("已恢复配音音频", 80)
        elif _valid_resume_file(dubbed_audio):
            self._progress("已恢复配音音频", 80)
        else:
            self._progress("合并配音音频...", 80)
            temp_dubbed_audio = output_dir / "_dubbed_work.wav"
            _call_with_process_runner(
                _mix_timed_audio_segments,
                self._run_process,
                audio_files,
                temp_dubbed_audio,
                duration=video_duration,
            )
            shutil.copy2(temp_dubbed_audio, dubbed_audio)
        results["dubbed_audio"] = str(dubbed_audio)
        self._save_manifest(results=results, stage="dubbed_audio", status="running")

        self._progress("混合背景音乐和译文配音...", 83)
        mixed_audio = output_dir / f"{video_path.stem}_mixed_audio.wav"
        if _valid_resume_file(results.get("mixed_audio")):
            mixed_audio = Path(results["mixed_audio"])
            self._progress("已恢复混合音频", 83)
        elif not _valid_resume_file(mixed_audio):
            _call_with_process_runner(
                mix_background_with_dubbing,
                self._run_process,
                background_audio,
                dubbed_audio,
                mixed_audio,
                duration=video_duration,
            )
        results["mixed_audio"] = str(mixed_audio)
        self._save_manifest(results=results, stage="mixed_audio", status="running")

        self._progress("替换视频音轨...", 85)
        if not _valid_resume_file(dubbed_video):
            _call_with_process_runner(replace_audio, self._run_process, video_path, mixed_audio, dubbed_video)
        results["dubbed_video"] = str(dubbed_video)
        self._save_manifest(results=results, stage="dubbed_video", status="running")
        return dubbed_video

    def _progress(self, message: str, pct: float = 0):
        self._check_cancelled()
        logger.info("[%s%%] %s", pct, message)
        if self.on_progress:
            self.on_progress(message, pct)

    def _transcribe_with_retries(self, speech_audio_path: Path, language: Optional[str]) -> dict:
        last_error: Exception | None = None
        for attempt in range(1, ASR_RETRIES + 1):
            try:
                self._check_cancelled()
                return transcribe(
                    speech_audio_path,
                    model_name=self.whisper_model,
                    language=language,
                    cancel_check=self._check_cancelled,
                )
            except InterruptedError:
                raise
            except Exception as exc:
                last_error = exc
                logger.warning("ASR failed attempt %s/%s: %s", attempt, ASR_RETRIES, exc)
                if attempt < ASR_RETRIES:
                    time.sleep(ASR_RETRY_DELAY_SECONDS * attempt)
        raise RuntimeError(f"语音识别失败：{last_error}") from last_error

    def _check_cancelled(self):
        if self._cancelled:
            raise InterruptedError("处理已停止")

    def _run_process(self, cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
        attempts = _process_attempts(cmd)
        for attempt in range(1, attempts + 1):
            try:
                return self._run_process_once(cmd, check=check)
            except InterruptedError:
                raise
            except (subprocess.CalledProcessError, TimeoutError) as exc:
                if attempt >= attempts:
                    raise
                logger.warning(
                    "Command failed attempt %s/%s, retrying: %s",
                    attempt,
                    attempts,
                    _format_command_for_log(cmd),
                )
                self._check_cancelled()
                time.sleep(PROCESS_RETRY_DELAY_SECONDS * attempt)

        raise RuntimeError("Process retry loop exited unexpectedly.")

    def _run_process_once(self, cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
        self._check_cancelled()
        timeout_seconds = _process_timeout_seconds(cmd)
        started_at = time.monotonic()
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform.startswith("win") else 0
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
        )
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        stdout_reader = _start_stream_reader(getattr(process, "stdout", None), stdout_chunks)
        stderr_reader = _start_stream_reader(getattr(process, "stderr", None), stderr_chunks)
        with self._process_lock:
            self._current_process = process

        try:
            while process.poll() is None:
                if self._cancelled:
                    self._terminate_process(process)
                    _join_reader(stdout_reader)
                    _join_reader(stderr_reader)
                    raise InterruptedError("处理已停止")
                if timeout_seconds and time.monotonic() - started_at > timeout_seconds:
                    self._terminate_process(process)
                    _join_reader(stdout_reader)
                    _join_reader(stderr_reader)
                    command_name = Path(cmd[0]).name if cmd else "process"
                    raise TimeoutError(f"{command_name} 超过 {timeout_seconds} 秒仍未响应，已跳过该步骤")
                time.sleep(PROCESS_POLL_INTERVAL_SECONDS)
            _join_reader(stdout_reader)
            _join_reader(stderr_reader)
            stdout = "".join(stdout_chunks)
            stderr = "".join(stderr_chunks)
            if self._cancelled:
                raise InterruptedError("处理已停止")
        finally:
            with self._process_lock:
                if self._current_process is process:
                    self._current_process = None

        result = subprocess.CompletedProcess(cmd, process.returncode, stdout=stdout, stderr=stderr)
        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd, output=stdout, stderr=stderr)
        return result

    def _terminate_current_process(self):
        with self._process_lock:
            process = self._current_process
        if process and process.poll() is None:
            self._terminate_process(process)

    def _terminate_process(self, process: subprocess.Popen):
        try:
            process.terminate()
            process.wait(timeout=3)
        except Exception:
            try:
                process.kill()
            except Exception as exc:
                logger.warning("Could not kill process during cancellation: %s", exc)


def _safe_path_name(name: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    cleaned = "".join("_" if char in invalid_chars or ord(char) < 32 else char for char in name)
    cleaned = " ".join(cleaned.split()).strip(" .")
    return cleaned[:80]


def _valid_resume_file(path: str | Path | None) -> bool:
    if not path:
        return False
    path = Path(path)
    return path.exists() and path.is_file() and path.stat().st_size > 0


def discover_resumable_jobs(output_dir: str | Path = "./output") -> list[dict]:
    """Return unfinished pipeline jobs newest-first.

    A job remains resumable after crashes, power loss, explicit cancellation, or
    recoverable failures as long as its manifest exists and status is not
    ``completed``.
    """
    output_dir = Path(output_dir)
    jobs: list[dict] = []
    if not output_dir.exists():
        return jobs

    for manifest_path in output_dir.glob(f"*/{JOB_MANIFEST_NAME}"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("status") == "completed":
            continue
        manifest["manifest_path"] = str(manifest_path)
        manifest.setdefault("output_dir", str(manifest_path.parent))
        jobs.append(manifest)

    jobs.sort(key=lambda item: item.get("updated_at") or item.get("created_at") or "", reverse=True)
    return jobs


def _parse_srt_segments(path: Path) -> list[dict]:
    content = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", content.strip())
    segments: list[dict] = []

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        timing_index = 0
        if "-->" not in lines[0] and len(lines) > 1:
            timing_index = 1
        if "-->" not in lines[timing_index]:
            continue

        start_text, end_text = [part.strip() for part in lines[timing_index].split("-->", 1)]
        text = "\n".join(lines[timing_index + 1:]).strip()
        if not text:
            continue

        segments.append({
            "start": _srt_time_to_seconds(start_text),
            "end": _srt_time_to_seconds(end_text),
            "text": text,
        })

    return segments


def _srt_time_to_seconds(value: str) -> float:
    match = re.match(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})", value.strip())
    if not match:
        return 0.0
    hours, minutes, seconds, millis = match.groups()
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis.ljust(3, "0")[:3]) / 1000
    )


def _mix_timed_audio_segments(
    audio_files: list[tuple[dict, Path]],
    output_path: Path,
    duration: float,
    process_runner: Callable[[list[str], bool], subprocess.CompletedProcess] | None = None,
) -> None:
    """Mix TTS segment files onto the subtitle timeline (A-tier alignment).

    A single alignment plan is computed across *all* segments (gap borrowing
    needs global context), then the work is split into bounded batches so the
    ffmpeg command line stays well under the Windows CreateProcess limit
    (WinError 206); per-batch outputs are summed at the end.

    Per segment the plan decides: placement (``start``), a capped time-stretch
    (``atempo``, never outside [MIN_SPEED, MAX_SPEED]), and a hard trim. Overflow
    is absorbed first by borrowing from neighbouring gaps and only then by a
    capped speed-up, so dubs stay intelligible for every language pair.
    """
    if not audio_files:
        raise RuntimeError("没有可合并的音频片段")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_duration = max(duration, _latest_segment_end(audio_files), 0.1)

    # Probe real durations once (Phase 2 input).
    segment_durations = [
        _audio_duration_seconds(path, process_runner=process_runner) for _, path in audio_files
    ]

    segments = [segment for segment, _ in audio_files]
    paths = [path for _, path in audio_files]
    plan = plan_final_placement(segments, segment_durations)

    items = list(zip(segments, paths, plan.plans))
    batches = _split_segments_into_mix_batches(items, target_duration)
    if len(batches) == 1:
        _run_single_mix_batch(
            batches[0],
            target_duration,
            output_path,
            process_runner=process_runner,
        )
        return

    workdir = output_path.parent / f"{output_path.stem}_mix_parts"
    ensure_dir(workdir)
    part_paths: list[Path] = []
    try:
        for index, batch in enumerate(batches):
            part_path = workdir / f"part_{index:04d}.wav"
            _run_single_mix_batch(
                batch,
                target_duration,
                part_path,
                process_runner=process_runner,
            )
            part_paths.append(part_path)
        _sum_mix_parts(part_paths, output_path, target_duration, process_runner=process_runner)
    finally:
        for part_path in part_paths:
            try:
                part_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove mix part: %s", part_path)
        try:
            workdir.rmdir()
        except OSError:
            pass


def _latest_segment_end(audio_files: list[tuple[dict, Path]]) -> float:
    latest = 0.0
    for segment, _ in audio_files:
        latest = max(latest, float(segment.get("end", segment.get("start", 0)) or 0))
    return latest


def _split_segments_into_mix_batches(
    items: list[tuple[dict, Path, "SegmentPlan"]],
    target_duration: float,
) -> list[list[tuple[dict, Path, "SegmentPlan"]]]:
    """Group planned segments into batches whose ffmpeg command stays under the limit.

    Each batch becomes one ``amix`` command. A conservative per-segment cost is
    used to estimate the command length without building the whole string,
    keeping the splitter cheap and stable.
    """
    batches: list[list[tuple[dict, Path, SegmentPlan]]] = []
    current: list[tuple[dict, Path, SegmentPlan]] = []
    # Estimated chars added to the command line by one segment: the "-i path"
    # input, the silence prefix input when start>0, and its filter rows. Paths
    # in this project are short, but leave generous headroom.
    per_segment_cost = 260
    fixed_cost = 400
    running = fixed_cost
    for segment, path, seg_plan in items:
        start = float(seg_plan.start if seg_plan else segment.get("start", 0) or 0)
        cost = per_segment_cost + (per_segment_cost if start > 0 else 0)
        if running + cost > MAX_MIX_COMMAND_CHARS and current:
            batches.append(current)
            current = []
            running = fixed_cost
        current.append((segment, path, seg_plan))
        running += cost
    if current:
        batches.append(current)
    return batches


def _run_single_mix_batch(
    batch: list[tuple[dict, Path, "SegmentPlan"]],
    target_duration: float,
    output_path: Path,
    process_runner: Callable[[list[str], bool], subprocess.CompletedProcess] | None = None,
) -> None:
    """Run one ffmpeg command mixing ``batch`` onto a silent timeline.

    Placement, speed-up and trim all come from each segment's ``SegmentPlan``,
    so this function contains no language-specific or ratio logic.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-t",
        str(target_duration),
        "-i",
        "anullsrc=r=44100:cl=stereo",
    ]

    filter_parts = [f"[0:a]atrim=0:{target_duration},asetpts=N/SR/TB[base]"]
    mixed_inputs = ["[base]"]
    input_index = 1
    for index, (segment, path, seg_plan) in enumerate(batch):
        # Fall back to the raw segment times when no plan is supplied, so legacy
        # callers that bypass the aligner still get sensible behaviour.
        start = float(seg_plan.start) if seg_plan else float(segment.get("start", 0) or 0)
        atempo = float(seg_plan.atempo) if seg_plan else 1.0
        trim_to = seg_plan.trim_to if seg_plan else None

        tempo_filters = _atempo_filters(atempo) if atempo > 1.01 else []
        timing_filters = tempo_filters + (
            [f"atrim=0:{trim_to:.3f}"] if trim_to and trim_to > 0 else []
        )
        prefix_label = ""
        if start > 0:
            cmd.extend([
                "-f",
                "lavfi",
                "-t",
                f"{start:.3f}",
                "-i",
                "anullsrc=r=44100:cl=stereo",
            ])
            prefix_label = f"[{input_index}:a]"
            input_index += 1
        audio_input_index = input_index
        cmd.extend(["-i", str(path)])
        input_index += 1

        label = f"a{index}"
        processed_label = f"{label}mid"
        filters = [
            "aformat=sample_rates=44100:channel_layouts=stereo",
            *timing_filters,
            "asetpts=PTS-STARTPTS",
        ]
        filter_parts.append(
            f"[{audio_input_index}:a]{','.join(filters)}[{processed_label}]"
        )
        if prefix_label:
            filter_parts.append(
                f"{prefix_label}[{processed_label}]concat=n=2:v=0:a=1,"
                f"atrim=0:{target_duration},asetpts=N/SR/TB[{label}]"
            )
        else:
            filter_parts.append(
                f"[{processed_label}]atrim=0:{target_duration},asetpts=N/SR/TB[{label}]"
            )
        mixed_inputs.append(f"[{label}]")

    filter_parts.append(
        f"{''.join(mixed_inputs)}amix=inputs={len(mixed_inputs)}:duration=first:normalize=0,"
        f"atrim=0:{target_duration},asetpts=N/SR/TB[aout]"
    )
    cmd.extend([
        "-filter_complex",
        ";".join(filter_parts),
        "-map",
        "[aout]",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        "-ac",
        "2",
        str(output_path),
    ])
    try:
        if process_runner:
            process_runner(cmd, True)
        else:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
            )
    except subprocess.CalledProcessError as exc:
        logger.error("Audio merge failed: %s", exc.stderr)
        raise RuntimeError(f"音频合并失败: {exc.stderr}") from exc


def _sum_mix_parts(
    part_paths: list[Path],
    output_path: Path,
    target_duration: float,
    process_runner: Callable[[list[str], bool], subprocess.CompletedProcess] | None = None,
) -> None:
    """Sum the per-batch parts into the final track."""
    cmd: list[str] = ["ffmpeg", "-y"]
    for part in part_paths:
        cmd.extend(["-i", str(part)])
    inputs = "".join(f"[{i}:a]" for i in range(len(part_paths)))
    cmd.extend([
        "-filter_complex",
        f"{inputs}amix=inputs={len(part_paths)}:duration=first:normalize=0,"
        f"atrim=0:{target_duration},asetpts=N/SR/TB[aout]",
        "-map",
        "[aout]",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        "-ac",
        "2",
        str(output_path),
    ])
    try:
        if process_runner:
            process_runner(cmd, True)
        else:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
            )
    except subprocess.CalledProcessError as exc:
        logger.error("Mix parts sum failed: %s", exc.stderr)
        raise RuntimeError(f"音频合并失败: {exc.stderr}") from exc


def _audio_duration_seconds(
    audio_file: Path,
    process_runner: Callable[[list[str], bool], subprocess.CompletedProcess] | None = None,
) -> float:
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_file),
        ]
        if process_runner:
            result = process_runner(cmd, True)
        else:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
            )
        return float((result.stdout or "0").strip() or 0)
    except Exception as exc:
        logger.warning("Could not read TTS segment duration for %s: %s", audio_file, exc)
        return 0.0


def _atempo_filters(tempo: float) -> list[str]:
    """Build an atempo filter chain for ``tempo``, hard-capped at MAX_SPEED.

    This is the last-resort guarantee that dubs never become unintelligible:
    regardless of how a caller computed ``tempo``, it is clamped to the
    language-agnostic ceiling before any ffmpeg chain is emitted.
    """
    if tempo <= 1.01:
        return []

    tempo = min(tempo, ALIGN_MAX_SPEED)

    filters = []
    remaining = tempo
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    filters.append(f"atempo={remaining:.4f}")
    return filters


def _call_with_process_runner(func: Callable, process_runner: Callable, *args, **kwargs):
    try:
        parameters = signature(func).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "process_runner" in parameters:
        kwargs["process_runner"] = process_runner
    return func(*args, **kwargs)


def _process_timeout_seconds(cmd: list[str]) -> int | None:
    if not cmd:
        return None
    command_name = Path(cmd[0]).stem.lower()
    if command_name == "ffprobe":
        return FFPROBE_TIMEOUT_SECONDS
    return None


def _process_attempts(cmd: list[str]) -> int:
    if not cmd:
        return 1
    command_name = Path(cmd[0]).stem.lower()
    if command_name == "ffmpeg":
        return 3
    if command_name == "demucs" or "demucs" in [part.lower() for part in cmd[:4]]:
        return 2
    return 1


def _format_command_for_log(cmd: list[str]) -> str:
    if not cmd:
        return "<empty command>"
    return " ".join(str(part) for part in cmd[:8])


def _start_stream_reader(stream, chunks: list[str]) -> threading.Thread | None:
    if stream is None:
        return None

    def read_stream():
        try:
            for line in stream:
                chunks.append(line)
        except Exception:
            return

    thread = threading.Thread(target=read_stream, daemon=True)
    thread.start()
    return thread


def _join_reader(thread: threading.Thread | None) -> None:
    if thread is not None:
        thread.join(timeout=1)


__all__ = ["TranslationPipeline", "discover_resumable_jobs"]
