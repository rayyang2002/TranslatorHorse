"""IndexTTS2 HTTP adapter.

This adapter deliberately does not depend on one specific IndexTTS2 package.
When the user later chooses an IndexTTS2 build, they can run it as a local HTTP
service and paste the service address into the desktop app.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests

from .tts import TTSEngine
from config.languages import get_language_capabilities
from utils.logger import logger


class IndexTTS2Engine(TTSEngine):
    """Generic IndexTTS2 HTTP engine."""

    def __init__(
        self,
        base_url: str = "",
        endpoint: str = "/tts",
        api_key: str = "",
        timeout: int = 300,
        **kwargs,
    ):
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint or "/tts"
        self.api_key = api_key
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        return bool(self.base_url)

    def generate(
        self,
        text: str,
        output_path: str | Path,
        voice: Optional[str] = None,
        speed: float = 1.0,
        reference_audio: Optional[str] = None,
        emotion: str = "natural",
        target_duration: float | None = None,
        timing_strategy: str = "natural_with_light_stretch",
        **kwargs,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.is_configured:
            raise RuntimeError(
                "你选择了 IndexTTS2，但还没有填写 IndexTTS2 接口地址。"
                "请先在软件的“IndexTTS2 接口选项”里填写地址，例如 http://127.0.0.1:7860。"
            )

        url = urljoin(f"{self.base_url}/", self.endpoint.lstrip("/"))
        payload = {
            "text": text,
            "voice": voice or "default_voice",
            "speed": speed,
            "reference_audio": reference_audio or "",
            "emotion": emotion,
            "target_duration": target_duration,
            "timing_strategy": timing_strategy,
            "format": "wav",
        }

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        logger.info("Calling IndexTTS2 endpoint: %s", url)
        response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "audio" in content_type or "octet-stream" in content_type:
            output_path.write_bytes(response.content)
            return output_path

        data = response.json()
        if "audio_base64" in data:
            output_path.write_bytes(base64.b64decode(data["audio_base64"]))
            return output_path

        if "audio_path" in data:
            source_path = Path(data["audio_path"])
            if not source_path.exists():
                raise RuntimeError(f"IndexTTS2 returned audio_path, but file does not exist: {source_path}")
            output_path.write_bytes(source_path.read_bytes())
            return output_path

        raise RuntimeError(
            "IndexTTS2 接口已响应，但没有返回可识别的音频。"
            "当前适配器支持：直接返回 audio/wav，或 JSON 返回 audio_base64/audio_path。"
        )

    def get_voices(self) -> list[str]:
        return ["default_voice", "clone_reference", "voice_library"]

    def get_supported_languages(self) -> list[str]:
        capabilities = get_language_capabilities()
        return [
            code
            for code, item in capabilities.items()
            if item.tts in {"stable", "experimental"} and item.tts_engine == "indextts2"
        ]
