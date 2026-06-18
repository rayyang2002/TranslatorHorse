"""字幕渲染模块 - 支持硬字幕和软字幕"""
from pathlib import Path
from typing import Callable, Optional
import subprocess

from utils.logger import logger
from utils.helpers import srt_timestamp
from core.video_processor import burn_subtitles


def generate_soft_subtitle(
    segments: list[dict],
    output_path: str | Path,
    use_translated: bool = True,
) -> Path:
    """
    生成 SRT 字幕文件 (软字幕)

    参数:
        segments: 包含 translated_text 的片段列表
        output_path: 输出 .srt 文件路径
        use_translated: True 用翻译文本, False 用原文
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for i, seg in enumerate(segments, 1):
        text = seg.get("translated_text", "") if use_translated else seg["text"]
        text = text.strip()
        if not text:
            continue

        start = srt_timestamp(seg["start"])
        end = srt_timestamp(seg["end"])
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    content = "\n".join(lines)
    output_path.write_text(content, encoding="utf-8")
    logger.info(f"软字幕已生成: {output_path} ({len(lines)//4} 条)")
    return output_path


def render_hard_subtitle(
    video_path: str | Path,
    srt_path: str | Path,
    output_path: str | Path,
    font: str = "Microsoft YaHei",
    font_size: int = 24,
    font_color: str = "#FFFFFF",
    process_runner: Callable[[list[str], bool], subprocess.CompletedProcess] | None = None,
) -> Path:
    """将字幕烧录到视频中 (硬字幕)"""
    style = {
        "font": font,
        "font_size": font_size,
        "color": font_color,
    }
    return burn_subtitles(video_path, srt_path, output_path, style, process_runner=process_runner)


def generate_bilingual_subtitle(
    segments: list[dict],
    output_path: str | Path,
) -> Path:
    """生成双语 SRT 字幕 (原文 + 翻译)"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for i, seg in enumerate(segments, 1):
        original = seg["text"].strip()
        translated = seg.get("translated_text", "").strip()
        if not original and not translated:
            continue

        start = srt_timestamp(seg["start"])
        end = srt_timestamp(seg["end"])

        if original and translated:
            text = f"{original}\n{translated}"
        elif translated:
            text = translated
        else:
            text = original

        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"双语字幕已生成: {output_path}")
    return output_path
