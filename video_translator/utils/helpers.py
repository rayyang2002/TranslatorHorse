"""工具函数"""
import os
import json
import subprocess
from pathlib import Path
from typing import Optional


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_file_size(path: str | Path) -> int:
    return os.path.getsize(path)


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:05.2f}"
    return f"{m}:{s:05.2f}"


def srt_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    ms = int((s - int(s)) * 1000)
    return f"{h:02d}:{m:02d}:{int(s):02d},{ms:03d}"


def parse_srt_timestamp(ts: str) -> float:
    parts = ts.replace(",", ".").split(":")
    h, m = int(parts[0]), int(parts[1])
    s = float(parts[2])
    return h * 3600 + m * 60 + s


def which(cmd: str) -> Optional[str]:
    try:
        result = subprocess.run(["where", cmd], capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0]
    except FileNotFoundError:
        pass
    return shutil_which(cmd)


def shutil_which(cmd: str) -> Optional[str]:
    import shutil
    return shutil.which(cmd)
