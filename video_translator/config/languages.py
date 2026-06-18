"""Language capability registry for G20-focused video translation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Capability = Literal["stable", "experimental", "subtitle_only", "unsupported"]


CHINESE_LANGUAGE_NAMES: dict[str, str] = {
    "zh": "中文",
    "en": "英语",
    "es": "西班牙语",
    "pt": "葡萄牙语",
    "fr": "法语",
    "de": "德语",
    "it": "意大利语",
    "ja": "日语",
    "ko": "韩语",
    "ru": "俄语",
    "ar": "阿拉伯语",
    "tr": "土耳其语",
    "hi": "印地语",
    "id": "印度尼西亚语",
    "nl": "荷兰语",
    "af": "南非荷兰语",
    "zu": "祖鲁语",
    "xh": "科萨语",
    "ta": "泰米尔语",
    "te": "泰卢固语",
    "bn": "孟加拉语",
    "ur": "乌尔都语",
    "mr": "马拉地语",
    "pa": "旁遮普语",
    "vi": "越南语",
    "pl": "波兰语",
    "el": "希腊语",
    "sv": "瑞典语",
    "fi": "芬兰语",
    "da": "丹麦语",
    "cs": "捷克语",
    "hu": "匈牙利语",
    "ro": "罗马尼亚语",
}


@dataclass(frozen=True)
class LanguageCapability:
    code: str
    native_name: str
    english_name: str
    g20_tier: str
    transcribe: Capability
    translate: Capability
    tts: Capability
    tts_engine: str = "indextts2"

    @property
    def display_name(self) -> str:
        chinese_name = CHINESE_LANGUAGE_NAMES.get(self.code, self.native_name)
        return f"{chinese_name}（{self.english_name}）"

    @property
    def tts_badge(self) -> str:
        return {
            "stable": "TTS stable",
            "experimental": "TTS experimental",
            "subtitle_only": "subtitles only",
            "unsupported": "no TTS",
        }[self.tts]


G20_LANGUAGE_CAPABILITIES: dict[str, LanguageCapability] = {
    # Core G20 working languages.
    "zh": LanguageCapability("zh", "中文", "Chinese", "core", "stable", "stable", "stable"),
    "en": LanguageCapability("en", "English", "English", "core", "stable", "stable", "stable"),
    "es": LanguageCapability("es", "Español", "Spanish", "core", "stable", "stable", "experimental"),
    "pt": LanguageCapability("pt", "Português", "Portuguese", "core", "stable", "stable", "experimental"),
    "fr": LanguageCapability("fr", "Français", "French", "core", "stable", "stable", "experimental"),
    "de": LanguageCapability("de", "Deutsch", "German", "core", "stable", "stable", "experimental"),
    "it": LanguageCapability("it", "Italiano", "Italian", "core", "stable", "stable", "experimental"),
    "ja": LanguageCapability("ja", "日本語", "Japanese", "core", "stable", "stable", "experimental"),
    "ko": LanguageCapability("ko", "한국어", "Korean", "core", "stable", "stable", "experimental"),
    "ru": LanguageCapability("ru", "Русский", "Russian", "core", "stable", "stable", "experimental"),
    "ar": LanguageCapability("ar", "العربية", "Arabic", "core", "stable", "stable", "experimental"),
    "tr": LanguageCapability("tr", "Türkçe", "Turkish", "core", "stable", "stable", "experimental"),
    "hi": LanguageCapability("hi", "हिन्दी", "Hindi", "core", "stable", "stable", "experimental"),
    "id": LanguageCapability("id", "Bahasa Indonesia", "Indonesian", "core", "stable", "stable", "experimental"),

    # Extended official/common languages across G20 members, EU, AU, India, and South Africa.
    "nl": LanguageCapability("nl", "Nederlands", "Dutch", "extended", "stable", "stable", "subtitle_only"),
    "af": LanguageCapability("af", "Afrikaans", "Afrikaans", "extended", "experimental", "stable", "subtitle_only"),
    "zu": LanguageCapability("zu", "isiZulu", "Zulu", "extended", "experimental", "stable", "unsupported"),
    "xh": LanguageCapability("xh", "isiXhosa", "Xhosa", "extended", "experimental", "stable", "unsupported"),
    "ta": LanguageCapability("ta", "தமிழ்", "Tamil", "extended", "stable", "stable", "subtitle_only"),
    "te": LanguageCapability("te", "తెలుగు", "Telugu", "extended", "stable", "stable", "subtitle_only"),
    "bn": LanguageCapability("bn", "বাংলা", "Bengali", "extended", "stable", "stable", "subtitle_only"),
    "ur": LanguageCapability("ur", "اردو", "Urdu", "extended", "stable", "stable", "subtitle_only"),
    "mr": LanguageCapability("mr", "मराठी", "Marathi", "extended", "stable", "stable", "subtitle_only"),
    "pa": LanguageCapability("pa", "ਪੰਜਾਬੀ", "Punjabi", "extended", "stable", "stable", "subtitle_only"),
    "vi": LanguageCapability("vi", "Tiếng Việt", "Vietnamese", "extended", "stable", "stable", "subtitle_only"),
    "pl": LanguageCapability("pl", "Polski", "Polish", "extended", "stable", "stable", "subtitle_only"),
    "el": LanguageCapability("el", "Ελληνικά", "Greek", "extended", "stable", "stable", "subtitle_only"),
    "sv": LanguageCapability("sv", "Svenska", "Swedish", "extended", "stable", "stable", "subtitle_only"),
    "fi": LanguageCapability("fi", "Suomi", "Finnish", "extended", "stable", "stable", "subtitle_only"),
    "da": LanguageCapability("da", "Dansk", "Danish", "extended", "stable", "stable", "subtitle_only"),
    "cs": LanguageCapability("cs", "Čeština", "Czech", "extended", "stable", "stable", "subtitle_only"),
    "hu": LanguageCapability("hu", "Magyar", "Hungarian", "extended", "stable", "stable", "subtitle_only"),
    "ro": LanguageCapability("ro", "Română", "Romanian", "extended", "stable", "stable", "subtitle_only"),
}


def get_language_capabilities() -> dict[str, LanguageCapability]:
    return G20_LANGUAGE_CAPABILITIES


def get_language_display_map() -> dict[str, str]:
    return {code: item.display_name for code, item in G20_LANGUAGE_CAPABILITIES.items()}


def describe_language(code: str) -> str:
    item = G20_LANGUAGE_CAPABILITIES.get(code)
    if item is None:
        return code
    return f"{item.english_name} - transcribe: {item.transcribe}, translate: {item.translate}, TTS: {item.tts}"
