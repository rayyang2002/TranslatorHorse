"""Reusable settings sections for the desktop app."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QComboBox,
    QCheckBox,
    QRadioButton,
    QButtonGroup,
    QVBoxLayout,
    QWidget,
)

from config.languages import get_language_capabilities
from core.translator import test_provider_connection
from core.video_processor import demucs_install_command, is_demucs_available


MODEL_CHOICES = ["gpt-4o-mini", "gpt-4o", "gpt-4", "deepseek-chat", "qwen-plus", "custom"]
DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.json"
DEFAULT_LLM_PROVIDER = {
    "name": "OpenAI",
    "api_key": "",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o-mini",
}

EDGE_TTS_FALLBACK_VOICES = [
    {"ShortName": "zh-CN-XiaoxiaoNeural", "Locale": "zh-CN", "Gender": "Female", "FriendlyName": "晓晓"},
    {"ShortName": "zh-CN-YunxiNeural", "Locale": "zh-CN", "Gender": "Male", "FriendlyName": "云希"},
    {"ShortName": "zh-CN-YunjianNeural", "Locale": "zh-CN", "Gender": "Male", "FriendlyName": "云健"},
    {"ShortName": "zh-CN-XiaoyiNeural", "Locale": "zh-CN", "Gender": "Female", "FriendlyName": "晓伊"},
    {"ShortName": "zh-CN-YunyangNeural", "Locale": "zh-CN", "Gender": "Male", "FriendlyName": "云扬"},
    {"ShortName": "zh-CN-liaoning-XiaobeiNeural", "Locale": "zh-CN-liaoning", "Gender": "Female", "FriendlyName": "东北话 晓北"},
    {"ShortName": "zh-CN-shaanxi-XiaoniNeural", "Locale": "zh-CN-shaanxi", "Gender": "Female", "FriendlyName": "陕西话 晓妮"},
    {"ShortName": "zh-HK-HiuGaaiNeural", "Locale": "zh-HK", "Gender": "Female", "FriendlyName": "粤语 HiuGaai"},
    {"ShortName": "zh-HK-HiuMaanNeural", "Locale": "zh-HK", "Gender": "Female", "FriendlyName": "粤语 HiuMaan"},
    {"ShortName": "zh-HK-WanLungNeural", "Locale": "zh-HK", "Gender": "Male", "FriendlyName": "粤语 WanLung"},
    {"ShortName": "zh-TW-HsiaoChenNeural", "Locale": "zh-TW", "Gender": "Female", "FriendlyName": "台湾 HsiaoChen"},
    {"ShortName": "zh-TW-HsiaoYuNeural", "Locale": "zh-TW", "Gender": "Female", "FriendlyName": "台湾 HsiaoYu"},
    {"ShortName": "zh-TW-YunJheNeural", "Locale": "zh-TW", "Gender": "Male", "FriendlyName": "台湾 YunJhe"},
    {"ShortName": "en-US-JennyNeural", "Locale": "en-US", "Gender": "Female", "FriendlyName": "Jenny"},
    {"ShortName": "en-US-GuyNeural", "Locale": "en-US", "Gender": "Male", "FriendlyName": "Guy"},
    {"ShortName": "en-US-AriaNeural", "Locale": "en-US", "Gender": "Female", "FriendlyName": "Aria"},
    {"ShortName": "en-US-DavisNeural", "Locale": "en-US", "Gender": "Male", "FriendlyName": "Davis"},
    {"ShortName": "en-GB-SoniaNeural", "Locale": "en-GB", "Gender": "Female", "FriendlyName": "Sonia"},
    {"ShortName": "en-GB-RyanNeural", "Locale": "en-GB", "Gender": "Male", "FriendlyName": "Ryan"},
    {"ShortName": "ja-JP-NanamiNeural", "Locale": "ja-JP", "Gender": "Female", "FriendlyName": "Nanami"},
    {"ShortName": "ja-JP-KeitaNeural", "Locale": "ja-JP", "Gender": "Male", "FriendlyName": "Keita"},
    {"ShortName": "ko-KR-SunHiNeural", "Locale": "ko-KR", "Gender": "Female", "FriendlyName": "SunHi"},
    {"ShortName": "ko-KR-InJoonNeural", "Locale": "ko-KR", "Gender": "Male", "FriendlyName": "InJoon"},
]


class DemucsInstallThread(QThread):
    """Installs Demucs into the current Python environment."""

    finished = Signal(bool, str)

    def run(self):
        cmd = demucs_install_command()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except Exception as exc:
            self.finished.emit(False, str(exc))
            return

        output = "\n".join(part for part in [result.stdout, result.stderr] if part).strip()
        self.finished.emit(result.returncode == 0, output or "Demucs 安装完成。")


class EdgeVoicesLoadThread(QThread):
    """Loads the full Edge-TTS voice list without blocking the UI."""

    finished = Signal(bool, object)

    def run(self):
        try:
            import edge_tts

            voices = asyncio.run(edge_tts.list_voices())
        except Exception as exc:
            self.finished.emit(False, str(exc))
            return

        self.finished.emit(True, voices)


class SettingsPanel(QWidget):
    """Compatibility wrapper that contains every settings section."""

    def __init__(self, parent=None, settings_path: str | Path | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        self.sections = SettingsSections(self, settings_path=settings_path)
        for widget in self.sections.widgets():
            layout.addWidget(widget)
        layout.addStretch()

    def get_settings(self) -> dict:
        return self.sections.get_settings()


class SettingsSections:
    """Owns shared controls while letting pages place sections independently."""

    def __init__(self, parent: QWidget | None = None, settings_path: str | Path | None = None):
        self.parent = parent
        self.settings_path = Path(settings_path) if settings_path else DEFAULT_SETTINGS_PATH
        self._stored_settings = self._load_settings_file()
        self.language_capabilities = get_language_capabilities()
        self._create_controls()
        self._apply_stored_settings()

    def widgets(self) -> list[QWidget]:
        return [
            self.llm_group,
            self.subtitle_group,
            self.tts_group,
            self.advanced_group,
        ]

    def _create_controls(self):
        self._create_llm_group()
        self._create_subtitle_group()
        self._create_tts_group()
        self._create_advanced_group()

    def _create_llm_group(self):
        self.llm_group = QGroupBox("LLM 设置")
        layout = QVBoxLayout(self.llm_group)

        self.llm_providers = self._stored_llm_providers()
        self.active_provider_index = self._stored_active_provider_index()
        self.editing_provider_index = 0

        picker_layout = QHBoxLayout()
        picker_layout.addWidget(QLabel("提供商:"))
        self.provider_combo = QComboBox()
        self.provider_combo.setMaxVisibleItems(12)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_selected)
        picker_layout.addWidget(self.provider_combo, 1)

        self.use_provider_btn = QPushButton("设为翻译提供商")
        self.use_provider_btn.clicked.connect(self._set_active_provider)
        picker_layout.addWidget(self.use_provider_btn)
        layout.addLayout(picker_layout)

        actions_layout = QHBoxLayout()
        self.add_provider_btn = QPushButton("新增提供商")
        self.add_provider_btn.clicked.connect(self._add_provider)
        actions_layout.addWidget(self.add_provider_btn)

        self.save_provider_btn = QPushButton("保存当前")
        self.save_provider_btn.clicked.connect(self._save_current_provider)
        actions_layout.addWidget(self.save_provider_btn)

        self.delete_provider_btn = QPushButton("删除当前")
        self.delete_provider_btn.clicked.connect(self._delete_current_provider)
        actions_layout.addWidget(self.delete_provider_btn)

        self.test_provider_btn = QPushButton("测试连接")
        self.test_provider_btn.clicked.connect(self._test_current_provider)
        actions_layout.addWidget(self.test_provider_btn)
        actions_layout.addStretch()
        layout.addLayout(actions_layout)

        self.active_provider_label = QLabel("")
        self.active_provider_label.setStyleSheet("color: #54a0ff;")
        layout.addWidget(self.active_provider_label)

        form_layout = QFormLayout()

        self.api_key_input = QLineEdit("")
        self.api_key_input.setPlaceholderText("输入真实 API Key；demo 只用于开发测试，不会真实翻译")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        form_layout.addRow("API Key:", self.api_key_input)

        self.base_url_input = QLineEdit("https://api.openai.com/v1")
        form_layout.addRow("Base URL:", self.base_url_input)

        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_CHOICES)
        self.model_combo.setEditable(True)
        form_layout.addRow("模型:", self.model_combo)

        self.provider_input = QLineEdit("OpenAI")
        self.provider_input.setPlaceholderText("提供商名称")
        form_layout.addRow("提供商名称:", self.provider_input)
        layout.addLayout(form_layout)
        self._refresh_provider_combo()
        self._load_provider_into_form(0)

    def _provider_from_form(self) -> dict:
        name = self.provider_input.text().strip() or f"Provider {self.provider_combo.currentIndex() + 1}"
        return {
            "name": name,
            "api_key": self.api_key_input.text().strip(),
            "base_url": self.base_url_input.text().strip() or "https://api.openai.com/v1",
            "model": self.model_combo.currentText().strip() or "gpt-4o-mini",
        }

    def _save_form_to_provider(self, index: int | None = None):
        if index is None:
            index = self.provider_combo.currentIndex()
        if 0 <= index < len(self.llm_providers):
            self.llm_providers[index] = self._provider_from_form()
            self._refresh_provider_combo(index)

    def _load_provider_into_form(self, index: int):
        if not (0 <= index < len(self.llm_providers)):
            return
        provider = self.llm_providers[index]
        self.provider_input.setText(provider.get("name", ""))
        self.api_key_input.setText(provider.get("api_key", ""))
        self.base_url_input.setText(provider.get("base_url", "https://api.openai.com/v1"))
        model = provider.get("model", "gpt-4o-mini")
        if self.model_combo.findText(model) < 0:
            self.model_combo.addItem(model)
        self.model_combo.setCurrentText(model)
        self.editing_provider_index = index
        self._update_active_provider_label()

    def _refresh_provider_combo(self, selected_index: int | None = None):
        if selected_index is None:
            selected_index = self.provider_combo.currentIndex()
        selected_index = max(0, min(selected_index, len(self.llm_providers) - 1))
        self.provider_combo.blockSignals(True)
        self.provider_combo.clear()
        for provider in self.llm_providers:
            self.provider_combo.addItem(provider.get("name") or "Unnamed Provider")
        self.provider_combo.setCurrentIndex(selected_index)
        self.provider_combo.blockSignals(False)
        self._update_active_provider_label()

    def _on_provider_selected(self, index: int):
        if 0 <= self.editing_provider_index < len(self.llm_providers):
            self.llm_providers[self.editing_provider_index] = self._provider_from_form()
        self._load_provider_into_form(index)

    def _update_active_provider_label(self):
        if not hasattr(self, "active_provider_label"):
            return
        active = self._active_provider()
        self.active_provider_label.setText(f"当前翻译使用: {active.get('name', 'Unnamed Provider')}")

    def _active_provider(self) -> dict:
        if not self.llm_providers:
            return {}
        self.active_provider_index = max(0, min(self.active_provider_index, len(self.llm_providers) - 1))
        return self.llm_providers[self.active_provider_index]

    def _add_provider(self):
        self._save_form_to_provider()
        next_number = len(self.llm_providers) + 1
        self.llm_providers.append({
            "name": f"Provider {next_number}",
            "api_key": "",
            "base_url": "",
            "model": "",
        })
        new_index = len(self.llm_providers) - 1
        self._refresh_provider_combo(new_index)
        self._load_provider_into_form(new_index)

    def _save_current_provider(self):
        self._save_form_to_provider()
        self._update_active_provider_label()
        self.save_settings()
        QMessageBox.information(self.parent, "已保存", "当前 LLM 设置已保存，下次启动会自动恢复。")

    def _delete_current_provider(self):
        if len(self.llm_providers) <= 1:
            QMessageBox.information(self.parent, "提示", "至少保留一个模型提供商。")
            return
        index = self.provider_combo.currentIndex()
        if not (0 <= index < len(self.llm_providers)):
            return
        removed_provider = self.llm_providers.pop(index)
        if self.active_provider_index == index:
            self.active_provider_index = 0
        elif self.active_provider_index > index:
            self.active_provider_index -= 1
        next_index = min(index, len(self.llm_providers) - 1)
        self._refresh_provider_combo(next_index)
        self._load_provider_into_form(next_index)
        self.save_settings()
        QMessageBox.information(self.parent, "提示", f"已删除提供商：{removed_provider.get('name', '')}")

    def _set_active_provider(self):
        index = self.provider_combo.currentIndex()
        if not (0 <= index < len(self.llm_providers)):
            return
        self._save_form_to_provider(index)
        self.active_provider_index = index
        self._update_active_provider_label()

    def _test_current_provider(self):
        index = self.provider_combo.currentIndex()
        if not (0 <= index < len(self.llm_providers)):
            return
        self._save_form_to_provider(index)
        provider = self.llm_providers[index]
        self.test_provider_btn.setEnabled(False)
        self.test_provider_btn.setText("测试中...")
        try:
            ok, message = test_provider_connection(
                api_key=provider.get("api_key", ""),
                base_url=provider.get("base_url", ""),
                model=provider.get("model", ""),
            )
        finally:
            self.test_provider_btn.setEnabled(True)
            self.test_provider_btn.setText("测试连接")

        title = "连接成功" if ok else "连接失败"
        if ok:
            QMessageBox.information(self.parent, title, message)
        else:
            QMessageBox.warning(self.parent, title, message)

    def _load_settings_file(self) -> dict:
        if not self.settings_path.exists():
            return {}
        try:
            return json.loads(self.settings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _stored_llm_providers(self) -> list[dict]:
        providers = self._stored_settings.get("llm_providers")
        if not isinstance(providers, list) or not providers:
            single_provider = self._stored_settings.get("model_provider", {})
            providers = [single_provider] if isinstance(single_provider, dict) else []

        cleaned = []
        for index, provider in enumerate(providers):
            if not isinstance(provider, dict):
                continue
            cleaned.append({
                "name": str(provider.get("name") or f"Provider {index + 1}"),
                "api_key": str(provider.get("api_key") or ""),
                "base_url": str(provider.get("base_url") or ""),
                "model": str(provider.get("model") or ""),
            })
        return cleaned or [dict(DEFAULT_LLM_PROVIDER)]

    def _stored_active_provider_index(self) -> int:
        try:
            index = int(self._stored_settings.get("active_provider_index", 0))
        except (TypeError, ValueError):
            index = 0
        return max(0, min(index, len(self.llm_providers) - 1))

    def _apply_stored_settings(self):
        target_lang = self._stored_settings.get("target_lang")
        target_index = self.target_lang_combo.findData(target_lang)
        if target_index >= 0:
            self.target_lang_combo.setCurrentIndex(target_index)

        subtitle_mode = self._stored_settings.get("subtitle_mode", "both")
        if subtitle_mode == "hard":
            self.hard_radio.setChecked(True)
        elif subtitle_mode == "soft":
            self.soft_radio.setChecked(True)
        else:
            self.both_radio.setChecked(True)

        self.tts_checkbox.setChecked(bool(self._stored_settings.get("tts_enabled", False)))
        tts_engine = self._stored_settings.get("tts_engine", "edge-tts")
        self.tts_engine_combo.setCurrentIndex(1 if tts_engine == "indextts2" else 0)

        indextts = self._stored_settings.get("indextts2", {})
        if isinstance(indextts, dict):
            self.index_base_url_input.setText(str(indextts.get("base_url") or ""))
            self.index_endpoint_input.setText(str(indextts.get("endpoint") or "/tts"))
            self.index_api_key_input.setText(str(indextts.get("api_key") or ""))
            self.voice_mode_combo.setCurrentText(str(indextts.get("voice_mode") or "default_voice"))
            self.reference_audio_input.setText(str(indextts.get("reference_audio") or ""))
            self.emotion_combo.setCurrentText(str(indextts.get("emotion") or "natural"))
            self.timing_combo.setCurrentText(str(indextts.get("timing_strategy") or "natural_with_light_stretch"))

        whisper_model = self._stored_settings.get("whisper_model")
        if whisper_model:
            for index in range(self.whisper_combo.count()):
                if self.whisper_combo.itemText(index).split(" ")[0] == whisper_model:
                    self.whisper_combo.setCurrentIndex(index)
                    break

        tts_voice = self._stored_settings.get("tts_voice")
        if tts_voice:
            voice_index = self.edge_voice_combo.findData(tts_voice)
            if voice_index >= 0:
                self.edge_voice_combo.setCurrentIndex(voice_index)

    def save_settings(self):
        settings = dict(self._stored_settings)
        current = self.get_settings()
        settings.update({
            "tts_engine": current["tts_engine"],
            "tts_enabled": current["tts_enabled"],
            "tts_voice": current["tts_voice"],
            "whisper_model": current["whisper_model"],
            "subtitle_mode": current["subtitle_mode"],
            "target_lang": current["target_lang"],
            "llm_providers": current["llm_providers"],
            "active_provider_index": current["active_provider_index"],
            "model_provider": {
                "name": current["provider"],
                "api_key": current["api_key"],
                "base_url": current["base_url"],
                "model": current["model"],
            },
            "indextts2": {
                "base_url": current["indextts2_base_url"],
                "endpoint": current["indextts2_endpoint"],
                "api_key": current["indextts2_api_key"],
                "voice_mode": current["tts_voice_mode"],
                "emotion": current["tts_emotion"],
                "timing_strategy": current["tts_timing_strategy"],
                "reference_audio": current["tts_reference_audio"],
            },
        })
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
        self._stored_settings = settings

    def _create_subtitle_group(self):
        self.subtitle_group = QGroupBox("字幕设置")
        layout = QVBoxLayout(self.subtitle_group)

        lang_layout = QFormLayout()
        self.target_lang_combo = QComboBox()
        for code, item in self.language_capabilities.items():
            self.target_lang_combo.addItem(f"{item.display_name}-{item.tts_badge}", code)
        zh_index = self.target_lang_combo.findData("zh")
        if zh_index >= 0:
            self.target_lang_combo.setCurrentIndex(zh_index)
        self.target_lang_combo.currentIndexChanged.connect(self._update_language_hint)
        lang_layout.addRow("目标语言:", self.target_lang_combo)

        self.source_label = QLabel("自动检测")
        self.source_label.setStyleSheet("color: #54a0ff;")
        lang_layout.addRow("源语言:", self.source_label)

        self.language_hint = QLabel("")
        self.language_hint.setWordWrap(True)
        self.language_hint.setStyleSheet("color: #8fb3d9;")
        lang_layout.addRow("能力:", self.language_hint)
        layout.addLayout(lang_layout)

        self.subtitle_mode_group = QButtonGroup(self.parent)
        self.subtitle_mode_group.setExclusive(True)
        self.both_radio = QRadioButton("硬字幕 + 软字幕 (SRT)")
        self.both_radio.setChecked(True)
        self.hard_radio = QRadioButton("仅硬字幕 (烧录到画面)")
        self.soft_radio = QRadioButton("仅软字幕 (外挂 SRT 文件)")
        self.subtitle_mode_group.addButton(self.both_radio, 0)
        self.subtitle_mode_group.addButton(self.hard_radio, 1)
        self.subtitle_mode_group.addButton(self.soft_radio, 2)
        layout.addWidget(self.both_radio)
        layout.addWidget(self.hard_radio)
        layout.addWidget(self.soft_radio)
        self._update_language_hint()

    def _create_tts_group(self):
        self.tts_group = QGroupBox("配音设置")
        layout = QVBoxLayout(self.tts_group)

        self.tts_checkbox = QCheckBox("启用 AI 配音 (TTS)")
        layout.addWidget(self.tts_checkbox)

        engine_layout = QHBoxLayout()
        engine_layout.addWidget(QLabel("TTS 引擎:"))
        self.tts_engine_combo = QComboBox()
        self.tts_engine_combo.addItems(["edge-tts (可直接使用)", "IndexTTS2 (接口模式，预留)"])
        self.tts_engine_combo.currentIndexChanged.connect(self._update_tts_hint)
        engine_layout.addWidget(self.tts_engine_combo)
        engine_layout.addStretch()
        layout.addLayout(engine_layout)

        self.tts_hint = QLabel("")
        self.tts_hint.setWordWrap(True)
        self.tts_hint.setStyleSheet("color: #8fb3d9;")
        layout.addWidget(self.tts_hint)

        separation_group = QGroupBox("人声分离")
        separation_layout = QVBoxLayout(separation_group)
        self.demucs_status_label = QLabel("")
        self.demucs_status_label.setWordWrap(True)
        separation_layout.addWidget(self.demucs_status_label)

        demucs_actions = QHBoxLayout()
        self.check_demucs_btn = QPushButton("检测 Demucs")
        self.check_demucs_btn.clicked.connect(self._refresh_demucs_status)
        demucs_actions.addWidget(self.check_demucs_btn)

        self.install_demucs_btn = QPushButton("安装 Demucs")
        self.install_demucs_btn.clicked.connect(self._install_demucs)
        demucs_actions.addWidget(self.install_demucs_btn)
        demucs_actions.addStretch()
        separation_layout.addLayout(demucs_actions)
        layout.addWidget(separation_group)

        edge_group = QGroupBox("Edge-TTS 选项")
        edge_layout = QFormLayout(edge_group)
        edge_voice_row = QHBoxLayout()
        self.edge_voice_combo = QComboBox()
        self.edge_voice_combo.setMinimumWidth(360)
        self.edge_voice_combo.setMaxVisibleItems(20)
        edge_voice_row.addWidget(self.edge_voice_combo, 1)

        self.refresh_edge_voices_btn = QPushButton("刷新音色")
        self.refresh_edge_voices_btn.clicked.connect(self._load_edge_voices)
        edge_voice_row.addWidget(self.refresh_edge_voices_btn)
        edge_layout.addRow("音色:", edge_voice_row)

        self.edge_voice_status_label = QLabel("")
        self.edge_voice_status_label.setWordWrap(True)
        self.edge_voice_status_label.setStyleSheet("color: #8fb3d9;")
        edge_layout.addRow("", self.edge_voice_status_label)
        layout.addWidget(edge_group)

        index_group = QGroupBox("IndexTTS2 接口选项")
        index_layout = QFormLayout(index_group)
        self.index_base_url_input = QLineEdit()
        self.index_base_url_input.setPlaceholderText("例如 http://127.0.0.1:7860")
        index_layout.addRow("接口地址:", self.index_base_url_input)

        self.index_endpoint_input = QLineEdit("/tts")
        self.index_endpoint_input.setPlaceholderText("例如 /tts 或 /api/tts")
        index_layout.addRow("接口路径:", self.index_endpoint_input)

        self.index_api_key_input = QLineEdit()
        self.index_api_key_input.setPlaceholderText("没有就留空")
        self.index_api_key_input.setEchoMode(QLineEdit.Password)
        index_layout.addRow("接口密钥:", self.index_api_key_input)

        self.voice_mode_combo = QComboBox()
        self.voice_mode_combo.addItems(["default_voice", "clone_reference", "voice_library"])
        index_layout.addRow("音色模式:", self.voice_mode_combo)

        reference_layout = QHBoxLayout()
        self.reference_audio_input = QLineEdit()
        self.reference_audio_input.setPlaceholderText("参考音频，可选")
        reference_layout.addWidget(self.reference_audio_input, 1)
        self.reference_audio_btn = QPushButton("浏览")
        self.reference_audio_btn.clicked.connect(self._browse_reference_audio)
        reference_layout.addWidget(self.reference_audio_btn)
        index_layout.addRow("参考音频:", reference_layout)

        self.emotion_combo = QComboBox()
        self.emotion_combo.addItems(["natural", "narration", "serious", "warm", "energetic", "documentary"])
        index_layout.addRow("情绪风格:", self.emotion_combo)

        self.timing_combo = QComboBox()
        self.timing_combo.addItems(["strict_subtitle_duration", "natural_with_light_stretch", "segment_level"])
        index_layout.addRow("对齐策略:", self.timing_combo)
        layout.addWidget(index_group)
        self._populate_edge_voice_combo(EDGE_TTS_FALLBACK_VOICES, "zh-CN-XiaoxiaoNeural")
        self._load_edge_voices()
        self._refresh_demucs_status()
        self._update_tts_hint()

    def _create_advanced_group(self):
        self.advanced_group = QGroupBox("高级功能")
        layout = QFormLayout(self.advanced_group)

        self.whisper_combo = QComboBox()
        self.whisper_combo.addItems(["tiny (fastest)", "base", "small", "medium", "large-v3 (most accurate)"])
        self.whisper_combo.setCurrentText("base")
        layout.addRow("Whisper 模型:", self.whisper_combo)

        self.output_dir_label = QLabel("output")
        layout.addRow("输出目录:", self.output_dir_label)

        self.temp_dir_label = QLabel("temp")
        layout.addRow("临时目录:", self.temp_dir_label)

    def _current_tts_engine(self) -> str:
        return "indextts2" if "IndexTTS2" in self.tts_engine_combo.currentText() else "edge-tts"

    def _update_language_hint(self):
        code = self.target_lang_combo.currentData()
        item = self.language_capabilities.get(code)
        if item is None:
            self.language_hint.setText("未知语言能力。")
            return
        self.language_hint.setText(
            f"G20 tier: {item.g20_tier} | ASR: {item.transcribe} | "
            f"翻译: {item.translate} | IndexTTS2: {item.tts}"
        )

    def _update_tts_hint(self):
        if self._current_tts_engine() == "edge-tts":
            self.tts_hint.setText("当前选择 Edge-TTS：可以直接用，适合先跑通完整视频翻译流程。")
        else:
            self.tts_hint.setText("当前选择 IndexTTS2：填写接口地址和路径后即可接入你未来确定的版本。")

    def _load_edge_voices(self):
        if hasattr(self, "edge_voices_thread") and self.edge_voices_thread.isRunning():
            return
        self.refresh_edge_voices_btn.setEnabled(False)
        self.edge_voice_status_label.setText("正在加载 Edge-TTS 音色列表...")
        self.edge_voices_thread = EdgeVoicesLoadThread()
        self.edge_voices_thread.finished.connect(self._on_edge_voices_loaded)
        self.edge_voices_thread.finished.connect(self.edge_voices_thread.deleteLater)
        self.edge_voices_thread.start()

    def _on_edge_voices_loaded(self, ok: bool, payload: object):
        self.refresh_edge_voices_btn.setEnabled(True)
        if ok and isinstance(payload, list) and payload:
            current = self.edge_voice_combo.currentData() or "zh-CN-XiaoxiaoNeural"
            self._populate_edge_voice_combo(payload, str(current))
            self.edge_voice_status_label.setText(f"已加载 {len(payload)} 个 Edge-TTS 音色。")
            return
        self.edge_voice_status_label.setText(f"音色列表加载失败，正在使用内置常用音色。{payload}")

    def _populate_edge_voice_combo(self, voices: list[dict], selected_voice: str = "zh-CN-XiaoxiaoNeural"):
        grouped = sorted(voices, key=lambda item: (item.get("Locale", ""), item.get("Gender", ""), item.get("ShortName", "")))
        self.edge_voice_combo.blockSignals(True)
        self.edge_voice_combo.clear()
        for voice in grouped:
            short_name = voice.get("ShortName", "")
            if not short_name:
                continue
            locale = voice.get("Locale", "")
            gender = voice.get("Gender", "")
            friendly = _short_friendly_voice_name(voice.get("FriendlyName", ""))
            label = f"{locale} | {gender} | {friendly} | {short_name}".strip(" |")
            self.edge_voice_combo.addItem(label, short_name)
        index = self.edge_voice_combo.findData(selected_voice)
        if index < 0:
            index = self.edge_voice_combo.findData("zh-CN-XiaoxiaoNeural")
        self.edge_voice_combo.setCurrentIndex(max(index, 0))
        self.edge_voice_combo.blockSignals(False)

    def _refresh_demucs_status(self):
        available = is_demucs_available()
        if available:
            self.demucs_status_label.setText("Demucs 已可用：配音时会自动分离人声和背景音乐。")
            self.demucs_status_label.setStyleSheet("color: #76e6a1;")
        else:
            self.demucs_status_label.setText("Demucs 未安装：配音需要它来保留背景音乐并替换原人声。")
            self.demucs_status_label.setStyleSheet("color: #ffcf66;")
        self.install_demucs_btn.setEnabled(not available)

    def _install_demucs(self):
        self.install_demucs_btn.setEnabled(False)
        self.check_demucs_btn.setEnabled(False)
        self.demucs_status_label.setText("正在安装 Demucs，请稍候...")
        self.demucs_status_label.setStyleSheet("color: #54a0ff;")
        self.demucs_install_thread = DemucsInstallThread()
        self.demucs_install_thread.finished.connect(self._on_demucs_install_finished)
        self.demucs_install_thread.finished.connect(self.demucs_install_thread.deleteLater)
        self.demucs_install_thread.start()

    def _on_demucs_install_finished(self, ok: bool, output: str):
        self.check_demucs_btn.setEnabled(True)
        self._refresh_demucs_status()
        if ok and is_demucs_available():
            QMessageBox.information(self.parent, "安装完成", "Demucs 已安装完成，可以直接进行配音翻译。")
        else:
            self.install_demucs_btn.setEnabled(True)
            QMessageBox.warning(self.parent, "安装失败", output[-2000:] if output else "Demucs 安装失败。")

    def _browse_reference_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self.parent,
            "选择参考音频",
            "",
            "Audio files (*.wav *.mp3 *.m4a *.flac);;All files (*.*)",
        )
        if path:
            self.reference_audio_input.setText(path)

    def get_settings(self) -> dict:
        mode_map = {0: "both", 1: "hard", 2: "soft"}
        if hasattr(self, "llm_providers"):
            self._save_form_to_provider(self.provider_combo.currentIndex())
        provider = self._active_provider() if hasattr(self, "llm_providers") else {
            "api_key": self.api_key_input.text().strip(),
            "base_url": self.base_url_input.text().strip(),
            "model": self.model_combo.currentText().strip(),
            "name": self.provider_input.text().strip(),
        }
        return {
            "api_key": provider.get("api_key", ""),
            "base_url": provider.get("base_url", "https://api.openai.com/v1"),
            "model": provider.get("model", "gpt-4o-mini"),
            "provider": provider.get("name", ""),
            "llm_providers": list(self.llm_providers) if hasattr(self, "llm_providers") else [provider],
            "active_provider_index": self.active_provider_index if hasattr(self, "active_provider_index") else 0,
            "target_lang": self.target_lang_combo.currentData(),
            "subtitle_mode": mode_map.get(self.subtitle_mode_group.checkedId(), "both"),
            "tts_enabled": self.tts_checkbox.isChecked(),
            "tts_engine": self._current_tts_engine(),
            "tts_voice_mode": self.voice_mode_combo.currentText(),
            "tts_voice": self.edge_voice_combo.currentData() or "zh-CN-XiaoxiaoNeural",
            "tts_reference_audio": self.reference_audio_input.text().strip(),
            "tts_emotion": self.emotion_combo.currentText(),
            "tts_timing_strategy": self.timing_combo.currentText(),
            "indextts2_base_url": self.index_base_url_input.text().strip(),
            "indextts2_endpoint": self.index_endpoint_input.text().strip() or "/tts",
            "indextts2_api_key": self.index_api_key_input.text().strip(),
            "whisper_model": self.whisper_combo.currentText().split(" ")[0],
        }


def _short_friendly_voice_name(value: str) -> str:
    if not value:
        return ""
    for prefix in ("Microsoft ", "Online (Natural) - "):
        value = value.replace(prefix, "")
    return value.strip()
