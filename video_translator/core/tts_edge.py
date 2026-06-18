"""Edge-TTS 引擎实现"""
import asyncio
from pathlib import Path
from typing import Optional

from .tts import TTSEngine
from utils.logger import logger


class EdgeTTSEngine(TTSEngine):
    """基于 edge-tts 的免费 TTS 引擎"""

    def __init__(self):
        self._voices = []
        self._lang_map = {
            "zh": "zh-CN-XiaoxiaoNeural",
            "en": "en-US-JennyNeural",
            "ja": "ja-JP-NanamiNeural",
            "ko": "ko-KR-SunHiNeural",
        }

    def generate(
        self,
        text: str,
        output_path: str | Path,
        voice: Optional[str] = None,
        speed: float = 1.0,
        **kwargs,
    ) -> Path:
        import edge_tts

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if voice is None:
            voice = "zh-CN-XiaoxiaoNeural"

        # ``speed`` already encodes the A-tier Phase 1 hint (see
        # core.tts_aligner.plan_tts_speed / core.tts.generate_subtitle_audio):
        # the caller asks edge-tts to speak faster/slower at synthesis time.
        # ``target_duration`` / ``timing_strategy`` from kwargs are intentionally
        # not used here — edge-tts exposes no fine-grained stretch control, and
        # the remaining alignment is handled by the mixer (Phase 2).
        rate_pct = max(-50, min(100, int((speed - 1.0) * 100)))
        tts = edge_tts.Communicate(text, voice=voice, rate=f"{rate_pct:+d}%")

        async def _run():
            await tts.save(str(output_path))

        asyncio.run(_run())
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("Edge-TTS did not return audio for this segment.")
        logger.info(f"Edge-TTS 生成: {output_path.name}")
        return output_path

    def get_voices(self) -> list[str]:
        import edge_tts

        async def _list():
            voices = await edge_tts.list_voices()
            return [v["ShortName"] for v in voices]

        return asyncio.run(_list())

    def get_supported_languages(self) -> list[str]:
        return list(self._lang_map.keys())
