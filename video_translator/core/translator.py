"""Subtitle translation through OpenAI-compatible APIs."""

from __future__ import annotations

from collections.abc import Callable
import time

from utils.logger import logger


LANGUAGE_NAMES = {
    "auto": "auto detected language",
    "af": "Afrikaans",
    "ar": "Arabic",
    "bn": "Bengali",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mr": "Marathi",
    "nl": "Dutch",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "tr": "Turkish",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
}
LLM_REQUEST_RETRIES = 3
LLM_RETRY_DELAY_SECONDS = 2


def translate_segments(
    segments: list[dict],
    target_lang: str,
    source_lang: str = "auto",
    api_key: str = "",
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
    cancel_check: Callable[[], None] | None = None,
) -> list[dict]:
    """Translate ASR segments and write translated_text into each segment."""
    if not api_key:
        raise ValueError("请填写真实 API Key 后再翻译。")
    if _is_test_mode(api_key, model):
        raise ValueError("当前是测试模型配置，不会执行真实翻译。请填写真实 API Key 和模型。")

    target_name = LANGUAGE_NAMES.get(target_lang, target_lang)
    source_name = "auto detected language" if source_lang == "auto" else LANGUAGE_NAMES.get(source_lang, source_lang)

    texts_to_translate = [segment["text"].strip() for segment in segments if segment["text"].strip()]
    logger.info(
        "Translating %s subtitle lines from %s to %s with model %s",
        len(texts_to_translate),
        source_name,
        target_name,
        model,
    )

    translated_all: list[str] = []
    batch_size = 20
    for batch_start in range(0, len(texts_to_translate), batch_size):
        _raise_if_cancelled(cancel_check)
        batch = texts_to_translate[batch_start:batch_start + batch_size]
        prompt = _build_translation_prompt(batch, source_name, target_name)
        try:
            response = _chat_completion(
                api_key=api_key,
                base_url=base_url,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            _raise_if_cancelled(cancel_check)
            translated_text = response.choices[0].message.content.strip()
            translated_all.extend(_parse_translation_response(translated_text, len(batch)))
            logger.info("Translation progress: %s/%s", min(batch_start + batch_size, len(texts_to_translate)), len(texts_to_translate))
        except InterruptedError:
            raise
        except Exception as exc:
            logger.error("Translation batch failed: %s", exc)
            translated_all.extend(batch)

    index = 0
    for segment in segments:
        if segment["text"].strip():
            segment["translated_text"] = translated_all[index] if index < len(translated_all) else segment["text"]
            index += 1
        else:
            segment["translated_text"] = ""

    logger.info("Translation completed")
    return segments


def translate_text(
    text: str,
    target_lang: str,
    source_lang: str = "auto",
    api_key: str = "",
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
) -> str:
    """Translate one text block."""
    if not api_key:
        raise ValueError("请填写真实 API Key 后再翻译。")
    if _is_test_mode(api_key, model):
        raise ValueError("当前是测试模型配置，不会执行真实翻译。请填写真实 API Key 和模型。")

    target_name = LANGUAGE_NAMES.get(target_lang, target_lang)
    response = _chat_completion(
        api_key=api_key,
        base_url=base_url,
        model=model,
        messages=[{
            "role": "user",
            "content": f"Translate the following text to {target_name}. Return only the translation:\n\n{text}",
        }],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def test_provider_connection(
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
) -> tuple[bool, str]:
    """Check whether an OpenAI-compatible model provider can answer a tiny request."""
    if not api_key:
        return False, "Please provide an API Key before testing the provider."

    if not model.strip():
        return False, "Please provide a model name before testing the provider."

    if _is_test_mode(api_key, model):
        return False, "当前是测试模型配置，不会连接真实翻译模型。请填写真实 API Key 和模型。"

    try:
        response = _chat_completion(
            api_key=api_key,
            base_url=base_url,
            model=model.strip(),
            messages=[{"role": "user", "content": "Reply with OK."}],
            temperature=0,
            max_tokens=4,
            timeout=15,
        )
        content = (response.choices[0].message.content or "").strip()
        return True, content or "Provider responded successfully."
    except Exception as exc:
        logger.error("LLM provider connection test failed: %s", exc)
        return False, str(exc)


def _is_test_mode(api_key: str, model: str) -> bool:
    key = api_key.strip().lower()
    model_name = model.strip().lower()
    return key in {"demo", "test", "local-demo"} or model_name.startswith(("demo", "test"))


def _chat_completion(
    api_key: str,
    base_url: str,
    timeout: int = 60,
    **kwargs,
):
    from openai import OpenAI

    last_error: Exception | None = None
    for attempt in range(1, LLM_REQUEST_RETRIES + 1):
        for candidate_base_url in _base_url_candidates(base_url):
            try:
                client = OpenAI(
                    api_key=api_key,
                    base_url=candidate_base_url,
                    timeout=timeout,
                )
                return client.chat.completions.create(**kwargs)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "LLM request failed attempt %s/%s with base_url=%s: %s",
                    attempt,
                    LLM_REQUEST_RETRIES,
                    candidate_base_url,
                    exc,
                )
        if attempt < LLM_REQUEST_RETRIES:
            time.sleep(LLM_RETRY_DELAY_SECONDS * attempt)

    if last_error is not None:
        raise last_error
    raise RuntimeError("No LLM Base URL is available.")


def _base_url_candidates(base_url: str) -> list[str]:
    normalized = (base_url or "https://api.openai.com/v1").strip().rstrip("/")
    candidates = [normalized]
    if not normalized.endswith("/v1"):
        candidates.append(f"{normalized}/v1")

    unique = []
    for candidate in candidates:
        if candidate and candidate not in unique:
            unique.append(candidate)
    return unique


def _raise_if_cancelled(cancel_check: Callable[[], None] | None):
    if cancel_check:
        cancel_check()


def _build_translation_prompt(texts: list[str], source: str, target: str) -> str:
    numbered = "\n".join(f"{index + 1}. {text}" for index, text in enumerate(texts))
    return (
        f"Translate these subtitle lines from {source} to {target}.\n"
        "Keep the meaning natural and concise for subtitles.\n"
        "Reply line by line with the same numbering:\n\n"
        f"{numbered}"
    )


def _parse_translation_response(response: str, expected_count: int) -> list[str]:
    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
    results: list[str] = []

    for line in lines:
        cleaned = line
        for index in range(1, expected_count + 1):
            for prefix in (f"{index}. ", f"{index}: ", f"{index}) "):
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
                    break
        results.append(cleaned)

    while len(results) < expected_count:
        results.append("")
    return results[:expected_count]
