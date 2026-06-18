"""Video processing helpers backed by ffmpeg and ffprobe."""

from __future__ import annotations

import json
import importlib.util
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from utils.helpers import ensure_dir
from utils.logger import logger

ProcessRunner = Callable[[list[str], bool], subprocess.CompletedProcess]
MEDIA_COMMAND_RETRIES = 3
MEDIA_RETRY_DELAY_SECONDS = 2


def extract_audio(
    video_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 16000,
    channels: int = 1,
    process_runner: ProcessRunner | None = None,
) -> Path:
    """Extract a mono 16 kHz WAV track for speech recognition."""
    video_path = Path(video_path)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    logger.info("Extracting audio: %s -> %s", video_path.name, output_path.name)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        str(output_path),
    ]
    _run_ffmpeg_for_output(cmd, output_path, process_runner=process_runner)
    logger.info("Audio extracted: %s", output_path)
    return output_path


def convert_audio(
    input_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 16000,
    channels: int = 1,
    process_runner: ProcessRunner | None = None,
) -> Path:
    """Convert an audio file to a normalized WAV format."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        str(output_path),
    ]
    _run_ffmpeg_for_output(cmd, output_path, process_runner=process_runner)
    return output_path


def separate_speech_and_background(
    audio_path: str | Path,
    output_dir: str | Path,
    process_runner: ProcessRunner | None = None,
) -> tuple[Path, Path]:
    """Separate an audio track into speech/vocals and background using Demucs."""
    demucs_cmd = _demucs_command()
    if demucs_cmd is None:
        raise RuntimeError(
            "需要先安装 Demucs 才能生成正确的配音视频：它用于把人声和背景音乐分离。"
            "请安装 demucs 后重试；否则软件不能安全地保留背景音乐并替换原说话声音。"
        )

    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    cmd = demucs_cmd + [
        "--two-stems",
        "vocals",
        "--mp3",
        "--mp3-bitrate",
        "320",
        "--out",
        str(output_dir),
        str(audio_path),
    ]
    stem_dir = output_dir / "htdemucs" / audio_path.stem
    last_error: Exception | None = None
    for attempt in range(1, 3):
        try:
            if process_runner:
                _run_process(cmd, check=True, process_runner=process_runner)
            else:
                _run_process(cmd, check=True)
            vocals = _first_existing(stem_dir / "vocals.mp3", stem_dir / "vocals.wav")
            background = _first_existing(stem_dir / "no_vocals.mp3", stem_dir / "no_vocals.wav")
            _ensure_file_created(vocals, "Demucs vocals")
            _ensure_file_created(background, "Demucs background")
            return vocals, background
        except subprocess.CalledProcessError as exc:
            details = _format_process_error(exc)
            last_error = RuntimeError(f"Demucs 分离失败：{details}")
            logger.error("Demucs failed: %s", details)
        except Exception as exc:
            last_error = exc
            logger.warning("Demucs output validation failed attempt %s/2: %s", attempt, exc)
        if attempt < 2:
            time.sleep(MEDIA_RETRY_DELAY_SECONDS * attempt)
    raise RuntimeError(str(last_error) if last_error else f"Demucs 分离完成，但没有找到输出文件: {stem_dir}")


def is_demucs_available() -> bool:
    """Return whether Demucs can be run by the current application environment."""
    return _demucs_command() is not None


def demucs_install_command() -> list[str]:
    """Return the command used by the desktop app to install Demucs."""
    return [sys.executable, "-m", "pip", "install", "demucs"]


def _demucs_command() -> list[str] | None:
    demucs = shutil.which("demucs")
    if demucs:
        return [demucs]
    if importlib.util.find_spec("demucs") is not None:
        return [sys.executable, "-m", "demucs"]
    return None


def mix_background_with_dubbing(
    background_audio: str | Path,
    dubbing_audio: str | Path,
    output_path: str | Path,
    duration: float,
    process_runner: ProcessRunner | None = None,
) -> Path:
    """Mix preserved background audio with the translated dubbing track."""
    background_audio = Path(background_audio)
    dubbing_audio = Path(dubbing_audio)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    safe_duration = max(float(duration or 0), 0.1)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(background_audio),
        "-i",
        str(dubbing_audio),
        "-filter_complex",
        (
            f"[0:a]atrim=0:{safe_duration},asetpts=N/SR/TB,volume=0.85[bg];"
            f"[1:a]atrim=0:{safe_duration},asetpts=N/SR/TB[dub];"
            "[bg][dub]amix=inputs=2:duration=first:normalize=0[aout]"
        ),
        "-map",
        "[aout]",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        "-ac",
        "2",
        str(output_path),
    ]
    _run_ffmpeg_for_output(cmd, output_path, process_runner=process_runner)
    return output_path


def get_video_info(video_path: str | Path, process_runner: ProcessRunner | None = None) -> dict:
    """Read duration, resolution, codecs, and file size."""
    video_path = Path(video_path)
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    result = _run_process(cmd, process_runner=process_runner) if process_runner else _run_process(cmd)
    info = json.loads(result.stdout or "{}")

    video_stream = None
    audio_stream = None
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video" and video_stream is None:
            video_stream = stream
        if stream.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = stream

    duration = float(info.get("format", {}).get("duration", 0))
    return {
        "duration": duration,
        "width": int(video_stream.get("width", 0)) if video_stream else 0,
        "height": int(video_stream.get("height", 0)) if video_stream else 0,
        "fps": _eval_fps(video_stream.get("r_frame_rate", "0/1")) if video_stream else 0,
        "video_codec": video_stream.get("codec_name", "") if video_stream else "",
        "audio_codec": audio_stream.get("codec_name", "") if audio_stream else "",
        "size": int(info.get("format", {}).get("size", 0)),
    }


def replace_audio(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    process_runner: ProcessRunner | None = None,
) -> Path:
    """Replace the original audio track with a generated dubbing track."""
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    logger.info("Replacing audio track: %s", video_path.name)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        str(output_path),
    ]
    _run_ffmpeg_for_output(cmd, output_path, process_runner=process_runner)
    logger.info("Audio replaced: %s", output_path)
    return output_path


def burn_subtitles(
    video_path: str | Path,
    srt_path: str | Path,
    output_path: str | Path,
    style: Optional[dict] = None,
    process_runner: ProcessRunner | None = None,
) -> Path:
    """Burn SRT subtitles into the video frames."""
    video_path = Path(video_path)
    srt_path = Path(srt_path)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    style = style or {}
    font = style.get("font", "Microsoft YaHei")
    font_size = style.get("font_size", 24)
    font_color = style.get("color", "#FFFFFF")
    subtitle_path = _escape_filter_path(srt_path)
    font_name = _escape_force_style_value(font)

    vf = (
        f"subtitles='{subtitle_path}':"
        f"force_style='FontName={font_name},FontSize={font_size},"
        f"PrimaryColour=&H{_color_to_ass(font_color)},"
        "BackColour=&H80000000,Outline=1,Shadow=1,Alignment=2'"
    )

    logger.info("Burning subtitles: %s", srt_path)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-c:a",
        "copy",
        "-preset",
        "fast",
        str(output_path),
    ]
    _run_ffmpeg_for_output(cmd, output_path, process_runner=process_runner)
    logger.info("Subtitled video generated: %s", output_path)
    return output_path


def combine_audio_and_video(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    process_runner: ProcessRunner | None = None,
) -> Path:
    """Combine video with an alternate audio track."""
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-map",
        "0:a:0?",
        str(output_path),
    ]
    _run_ffmpeg_for_output(cmd, output_path, process_runner=process_runner)
    return output_path


def _run_ffmpeg(cmd: list[str], process_runner: ProcessRunner | None = None) -> None:
    try:
        _run_process(cmd, check=True, process_runner=process_runner)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or exc.stdout or "unknown ffmpeg error"
        logger.error("ffmpeg failed: %s", stderr)
        raise RuntimeError(f"ffmpeg error: {stderr}") from exc


def _run_ffmpeg_for_output(
    cmd: list[str],
    output_path: str | Path,
    process_runner: ProcessRunner | None = None,
) -> None:
    output_path = Path(output_path)
    last_error: Exception | None = None
    for attempt in range(1, MEDIA_COMMAND_RETRIES + 1):
        try:
            _remove_partial_output(output_path)
            if process_runner:
                _run_ffmpeg(cmd, process_runner=process_runner)
            else:
                _run_ffmpeg(cmd)
            _ensure_file_created(output_path, "ffmpeg output")
            return
        except Exception as exc:
            last_error = exc
            logger.warning(
                "ffmpeg output failed attempt %s/%s for %s: %s",
                attempt,
                MEDIA_COMMAND_RETRIES,
                output_path,
                exc,
            )
            if attempt < MEDIA_COMMAND_RETRIES:
                time.sleep(MEDIA_RETRY_DELAY_SECONDS * attempt)
    raise RuntimeError(str(last_error) if last_error else f"ffmpeg output failed: {output_path}")


def _ensure_file_created(path: str | Path, label: str) -> None:
    path = Path(path)
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"{label} was not created or is empty: {path}")


def _remove_partial_output(path: Path) -> None:
    if path.exists() and path.is_file():
        try:
            path.unlink()
        except OSError:
            logger.warning("Could not remove partial output before retry: %s", path)


def _run_process(
    cmd: list[str],
    check: bool = False,
    process_runner: ProcessRunner | None = None,
) -> subprocess.CompletedProcess:
    if process_runner:
        return process_runner(cmd, check)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
    )


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _format_process_error(exc: subprocess.CalledProcessError) -> str:
    output = "\n".join(part for part in [exc.stdout, exc.stderr] if part)
    if not output:
        return f"命令退出码 {exc.returncode}"
    lines = [line for line in output.splitlines() if line.strip()]
    tail = "\n".join(lines[-12:])
    return f"命令退出码 {exc.returncode}\n{tail}"


def _escape_filter_path(path: str | Path) -> str:
    value = str(Path(path).resolve()).replace("\\", "/")
    value = value.replace(":", r"\:")
    value = value.replace("'", r"\'")
    return value


def _escape_force_style_value(value: str) -> str:
    return value.replace("'", "")


def _eval_fps(fps_str: str) -> float:
    try:
        numerator, denominator = fps_str.split("/")
        return float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _color_to_ass(color: str) -> str:
    color = color.lstrip("#")
    if len(color) != 6:
        return "00FFFFFF"
    return f"00{color[4:6]}{color[2:4]}{color[0:2]}"
