"""Video Translator command-line entry point."""

from __future__ import annotations

import argparse
import io
import os
import sys


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser(description="Video Translator")
    parser.add_argument("input", nargs="?", help="Video file path or URL")
    parser.add_argument("--api-key", "-k", help="LLM API key")
    parser.add_argument("--base-url", "-b", default="https://api.openai.com/v1", help="LLM API base URL")
    parser.add_argument("--model", "-m", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--target-lang", "-t", default="zh", help="Target language code")
    parser.add_argument("--whisper", "-w", default="base", help="Whisper model: tiny/base/small/medium/large-v3")
    parser.add_argument("--tts", action="store_true", help="Enable TTS dubbing")
    parser.add_argument("--tts-engine", choices=["edge-tts", "indextts2"], default="edge-tts", help="TTS engine")
    parser.add_argument("--tts-voice", default="zh-CN-XiaoxiaoNeural", help="Edge-TTS voice or IndexTTS2 voice mode")
    parser.add_argument("--tts-reference-audio", default="", help="Reference audio for IndexTTS2")
    parser.add_argument("--tts-emotion", default="natural", help="IndexTTS2 emotion/style")
    parser.add_argument(
        "--tts-timing",
        default="strict_subtitle_duration",
        choices=["strict_subtitle_duration", "natural_with_light_stretch", "segment_level"],
        help="IndexTTS2 subtitle duration alignment strategy",
    )
    parser.add_argument("--indextts2-url", default="", help="IndexTTS2 service base URL, for example http://127.0.0.1:7860")
    parser.add_argument("--indextts2-endpoint", default="/tts", help="IndexTTS2 service endpoint path")
    parser.add_argument("--indextts2-api-key", default="", help="Optional IndexTTS2 service API key")
    parser.add_argument("--subtitle-mode", choices=["hard", "soft", "both"], default="both", help="Subtitle mode")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument("--lang-list", action="store_true", help="List Whisper languages")
    parser.add_argument("--g20-list", action="store_true", help="List G20 language capability matrix")

    args = parser.parse_args()

    if args.lang_list:
        from core.asr import WHISPER_LANGUAGES

        print("\nWhisper supported languages:")
        for code, name in sorted(WHISPER_LANGUAGES.items()):
            print(f"  {code}: {name}")
        return

    if args.g20_list:
        from config.languages import get_language_capabilities

        print("\nG20 language capability matrix:")
        for code, item in get_language_capabilities().items():
            print(
                f"  {code:>3}: {item.display_name} | "
                f"tier={item.g20_tier} | asr={item.transcribe} | "
                f"translate={item.translate} | tts={item.tts}"
            )
        return

    if not args.input:
        parser.print_help()
        return

    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name = args.model or os.getenv("TRANSLATOR_MODEL", "gpt-4o-mini")

    if not api_key:
        print("Please provide an LLM API key with --api-key or OPENAI_API_KEY.")
        sys.exit(1)

    from core.pipeline import TranslationPipeline

    pipeline = TranslationPipeline(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        whisper_model=args.whisper or os.getenv("WHISPER_MODEL", "base"),
        tts_engine=args.tts_engine,
        tts_config={
            "base_url": args.indextts2_url,
            "endpoint": args.indextts2_endpoint,
            "api_key": args.indextts2_api_key,
        },
        output_dir=args.output,
    )

    def on_progress(msg, pct):
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"\r  {bar} {pct:.0f}%  {msg}", end="", flush=True)
        if pct >= 100:
            print()

    pipeline.on_progress = on_progress

    print("\nVideo Translator started")
    print(f"  input: {args.input}")
    print(f"  target language: {args.target_lang}")
    print(f"  subtitle mode: {args.subtitle_mode}")
    print(f"  dubbing: {args.tts} ({args.tts_engine})")
    print()

    try:
        results = pipeline.process(
            args.input,
            target_lang=args.target_lang,
            subtitle_mode=args.subtitle_mode,
            tts_enabled=args.tts,
            tts_voice=args.tts_voice,
            tts_reference_audio=args.tts_reference_audio or None,
            tts_emotion=args.tts_emotion,
            tts_timing_strategy=args.tts_timing,
        )

        print("\nDone.")
        print(f"  source language: {results.get('source_lang', '?')}")
        if "soft_subtitle" in results:
            print(f"  soft subtitle: {results['soft_subtitle']}")
        if "hard_subtitle_video" in results:
            print(f"  hard subtitle video: {results['hard_subtitle_video']}")
        if "dubbed_video" in results:
            print(f"  dubbed video: {results['dubbed_video']}")

    except Exception as exc:
        print(f"\nFailed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
