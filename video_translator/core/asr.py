"""语音识别模块 - 基于 faster-whisper (兼容新版 PyTorch)"""
from pathlib import Path
from collections.abc import Callable
from typing import Optional

from utils.logger import logger
from utils.helpers import srt_timestamp


def transcribe(
    audio_path: str | Path,
    model_name: str = "base",
    language: Optional[str] = None,
    device: str = "auto",
    cancel_check: Callable[[], None] | None = None,
) -> dict:
    """
    使用 faster-whisper 进行语音识别

    参数:
        audio_path: 音频文件路径
        model_name: 模型大小 (tiny/base/small/medium/large-v3)
        language: 指定语言代码 (None=自动检测)
        device: "auto" / "cuda" / "cpu"

    返回:
        {"segments": [...], "language": "zh", "text": "..."}
    """
    from faster_whisper import WhisperModel

    audio_path = Path(audio_path)
    logger.info(f"加载 faster-whisper 模型: {model_name}")
    if cancel_check:
        cancel_check()

    # 自动选设备
    compute_type = "default"
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        compute_type = "int8"

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    if cancel_check:
        cancel_check()

    logger.info(f"开始语音识别: {audio_path.name} (设备: {device})")

    segments_gen, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
    )

    detected_lang = info.language if info else (language or "unknown")
    duration = info.duration if info else 0

    # 转成标准格式
    segments = []
    for seg in segments_gen:
        if cancel_check:
            cancel_check()
        segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "words": [{"word": w.word, "start": w.start, "end": w.end}
                      for w in (seg.words or [])],
        })

    logger.info(f"识别完成, 检测语言: {detected_lang}, "
                f"片段数: {len(segments)}, 音频时长: {duration:.1f}s")

    return {
        "segments": segments,
        "language": detected_lang,
        "text": " ".join(s["text"] for s in segments),
    }


def save_srt(segments: list[dict], output_path: str | Path) -> Path:
    """将 segments 保存为 SRT 文件"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for i, seg in enumerate(segments, 1):
        text = seg["text"].strip()
        if not text:
            continue
        start = srt_timestamp(seg["start"])
        end = srt_timestamp(seg["end"])
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"字幕保存: {output_path} ({len(lines)//4} 条)")
    return output_path


WHISPER_LANGUAGES = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "as": "Assamese",
    "az": "Azerbaijani", "ba": "Bashkir", "be": "Belarusian", "bg": "Bulgarian",
    "bn": "Bengali", "bo": "Tibetan", "br": "Breton", "bs": "Bosnian",
    "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish",
    "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "et": "Estonian", "eu": "Basque", "fa": "Persian", "fi": "Finnish",
    "fo": "Faroese", "fr": "French", "gl": "Galician", "gu": "Gujarati",
    "ha": "Hausa", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian",
    "hu": "Hungarian", "hy": "Armenian", "id": "Indonesian", "is": "Icelandic",
    "it": "Italian", "ja": "Japanese", "jw": "Javanese", "ka": "Georgian",
    "kk": "Kazakh", "km": "Khmer", "kn": "Kannada", "ko": "Korean",
    "la": "Latin", "lb": "Luxembourgish", "lo": "Lao", "lt": "Lithuanian",
    "lv": "Latvian", "mg": "Malagasy", "mi": "Maori", "mk": "Macedonian",
    "ml": "Malayalam", "mn": "Mongolian", "mr": "Marathi", "ms": "Malay",
    "mt": "Maltese", "my": "Myanmar", "ne": "Nepali", "nl": "Dutch",
    "nn": "Nynorsk", "no": "Norwegian", "oc": "Occitan", "pa": "Punjabi",
    "pl": "Polish", "ps": "Pashto", "pt": "Portuguese", "ro": "Romanian",
    "ru": "Russian", "sa": "Sanskrit", "sd": "Sindhi", "si": "Sinhala",
    "sk": "Slovak", "sl": "Slovenian", "sn": "Shona", "so": "Somali",
    "sq": "Albanian", "sr": "Serbian", "su": "Sundanese", "sv": "Swedish",
    "sw": "Swahili", "ta": "Tamil", "te": "Telugu", "tg": "Tajik",
    "th": "Thai", "tk": "Turkmen", "tl": "Tagalog", "tr": "Turkish",
    "tt": "Tatar", "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek",
    "vi": "Vietnamese", "yi": "Yiddish", "yo": "Yoruba", "zh": "Chinese",
}
