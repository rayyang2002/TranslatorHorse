"""URL视频下载模块"""
import re
from pathlib import Path
from typing import Optional
from utils.logger import logger


def download_video(url: str, output_dir: str | Path) -> Optional[Path]:
    """使用 yt-dlp 下载视频"""
    try:
        import yt_dlp
    except ImportError:
        logger.error("yt-dlp 未安装，请执行: pip install yt-dlp")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "format": "best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            # 处理可能的扩展名差异
            p = Path(filename)
            if not p.exists():
                # 尝试其他扩展名
                for ext in [".mp4", ".webm", ".mkv", ".mov"]:
                    candidate = p.with_suffix(ext)
                    if candidate.exists():
                        p = candidate
                        break
            logger.info(f"下载完成: {p.name}")
            return p if p.exists() else None
    except Exception as e:
        logger.error(f"下载失败: {e}")
        return None


def is_url(text: str) -> bool:
    url_pattern = re.compile(
        r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+"
        r"(?:/[-\w$.+!*\'(),;:@&=?/~#%]*)?"
    )
    return bool(url_pattern.match(text))
