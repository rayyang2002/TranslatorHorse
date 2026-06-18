"""Main desktop window for Video Translator."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QThread, Qt, Signal, QUrl
from PySide6.QtGui import QDesktopServices, QDragEnterEvent, QDropEvent
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ui.components.progress_panel import ProgressPanel
from ui.components.settings_panel import SettingsSections
from utils.downloader import is_url


MEDIA_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".mp3", ".wav", ".m4a"}


class ProcessingThread(QThread):
    """Runs the translation pipeline outside the UI thread."""

    progress = Signal(str, float)
    finished = Signal(object)
    error = Signal(str)
    cancelled = Signal(str)

    def __init__(self, pipeline, input_path: str | list[str], settings: dict, resume_manifest_path: str | None = None):
        super().__init__()
        self.pipeline = pipeline
        self.input_paths = input_path if isinstance(input_path, list) else [input_path]
        self.settings = settings
        self.resume_manifest_path = resume_manifest_path

    def run(self):
        try:
            self.pipeline.on_progress = lambda msg, pct: self.progress.emit(msg, pct)
            if self.resume_manifest_path:
                self.progress.emit("继续断点任务...", 0)
                self.finished.emit(self.pipeline.resume(self.resume_manifest_path))
                return

            batch_results = []
            total = len(self.input_paths)
            for index, input_path in enumerate(self.input_paths, 1):
                if self.pipeline.is_cancelled:
                    raise InterruptedError("处理已停止")
                if total > 1:
                    self.progress.emit(f"批量处理 {index}/{total}: {Path(input_path).name}", 0)
                results = self.pipeline.process(
                    input_path,
                    target_lang=self.settings["target_lang"],
                    subtitle_mode=self.settings["subtitle_mode"],
                    tts_enabled=self.settings["tts_enabled"],
                    tts_voice=self.settings.get("tts_voice") or self.settings.get("tts_voice_mode"),
                    tts_reference_audio=self.settings.get("tts_reference_audio") or None,
                    tts_emotion=self.settings.get("tts_emotion", "natural"),
                    tts_timing_strategy=self.settings.get("tts_timing_strategy", "natural_with_light_stretch"),
                    process_mode=self.settings.get("process_mode", "full"),
                    original_subtitle_path=self.settings.get("original_subtitle_path"),
                )
                batch_results.append(results)
                if self.pipeline.is_cancelled:
                    raise InterruptedError("处理已停止")

            if total == 1:
                self.finished.emit(batch_results[0])
            else:
                self.finished.emit({
                    "batch_results": batch_results,
                    "input_videos": self.input_paths,
                    "last_result": batch_results[-1] if batch_results else {},
                })
        except InterruptedError as exc:
            self.cancelled.emit(str(exc) or "处理已停止")
        except Exception as exc:
            self.error.emit(str(exc))


class OutputRow(QFrame):
    """Small status row for an output artifact."""

    def __init__(self, title: str):
        super().__init__()
        self.path: str | None = None
        self.setObjectName("outputRow")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 7, 10, 7)

        self.title_label = QLabel(title)
        self.status_label = QLabel("未生成")
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        layout.addWidget(self.title_label, 1)
        layout.addWidget(self.status_label)

    def reset(self):
        self.path = None
        self.status_label.setText("未生成")
        self.status_label.setStyleSheet("color: #8fa3bd;")
        self.setCursor(Qt.ArrowCursor)

    def set_generated(self, path: str):
        self.path = path
        status_text = "已生成"
        file_path = Path(path)
        if file_path.exists():
            status_text = f"{file_path.stat().st_size / 1024 / 1024:.2f} MB"
        self.status_label.setText(status_text)
        self.status_label.setStyleSheet("color: #76e6a1;")
        self.setCursor(Qt.PointingHandCursor)

    def set_failed(self):
        self.status_label.setText("失败")
        self.status_label.setStyleSheet("color: #ff6b81;")
        self.setCursor(Qt.ArrowCursor)

    def open_path(self):
        if self.path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.path).resolve())))

    def mousePressEvent(self, event):
        if self.path:
            self.open_path()
        super().mousePressEvent(event)


class MaterialCard(QFrame):
    """Material panel that accepts dropped media files."""

    filesDropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setObjectName("workCard")
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if _media_paths_from_urls(event.mimeData().urls()):
            event.acceptProposedAction()
            self.setProperty("dragActive", True)
            self.style().unpolish(self)
            self.style().polish(self)
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setProperty("dragActive", False)
        self.style().unpolish(self)
        self.style().polish(self)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent):
        self.setProperty("dragActive", False)
        self.style().unpolish(self)
        self.style().polish(self)
        paths = _media_paths_from_urls(event.mimeData().urls())
        if paths:
            self.filesDropped.emit(paths)
            event.acceptProposedAction()


class MainWindow(QMainWindow):
    """Video translation workbench."""

    def __init__(self):
        super().__init__()
        self.pipeline = None
        self.processing_thread: ProcessingThread | None = None
        self._current_input = ""
        self._current_inputs: list[str] = []
        self._last_results: dict = {}
        self._resumable_jobs: list[dict] = []
        self.output_dir = Path("./output")
        self.history_path = self.output_dir / "history.json"
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("Video Translator")
        self.setMinimumSize(1180, 760)
        self._load_styles()

        self.settings_sections = SettingsSections(self)

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.setCentralWidget(central)

        root.addWidget(self._create_sidebar())

        self.stack = QStackedWidget()
        self.stack.addWidget(self._create_video_page())
        self.stack.addWidget(self._wrap_page(self._create_subtitle_page()))
        self.stack.addWidget(self._wrap_page(self._create_dubbing_page()))
        self.stack.addWidget(self._wrap_page(self._create_llm_page()))
        self.stack.addWidget(self._wrap_page(self._create_advanced_page()))
        self.stack.addWidget(self._wrap_page(self._create_logs_page()))
        root.addWidget(self.stack, 1)

        self.statusBar().showMessage("空闲")
        self._switch_page(0)
        self._reset_outputs()
        self._refresh_resumable_jobs()

    def _load_styles(self):
        style_path = Path(__file__).parent / "assets" / "styles.qss"
        if style_path.exists():
            self.setStyleSheet(style_path.read_text(encoding="utf-8"))

    def _create_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(230)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(18, 22, 18, 22)
        layout.setSpacing(12)

        title = QLabel("VideoFish")
        title.setObjectName("brandTitle")
        subtitle = QLabel("智能视频翻译")
        subtitle.setObjectName("brandSubtitle")
        version = QLabel("版本 v0.2")
        version.setObjectName("mutedLabel")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(version)
        layout.addSpacing(20)

        self.nav_buttons: list[QPushButton] = []
        nav_items = [
            "01  视频处理",
            "02  字幕设置",
            "03  配音设置",
            "04  LLM 设置",
            "05  高级功能",
            "06  任务日志",
        ]
        for index, text in enumerate(nav_items):
            btn = QPushButton(text)
            btn.setObjectName("navButton")
            btn.clicked.connect(lambda checked=False, i=index: self._switch_page(i))
            layout.addWidget(btn)
            self.nav_buttons.append(btn)

        layout.addStretch()

        self.sidebar_refresh_btn = QPushButton("刷新状态")
        self.sidebar_refresh_btn.setObjectName("sidebarRefreshButton")
        self.sidebar_refresh_btn.clicked.connect(self._refresh_status)
        layout.addWidget(self.sidebar_refresh_btn)
        return sidebar

    def _wrap_page(self, child: QWidget) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(child)
        return scroll

    def _create_page_shell(self, title_text: str, subtitle_text: str = "") -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 22, 28, 22)
        layout.setSpacing(18)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel(title_text)
        title.setObjectName("pageTitle")
        title_box.addWidget(title)
        if subtitle_text:
            sub = QLabel(subtitle_text)
            sub.setObjectName("mutedLabel")
            title_box.addWidget(sub)
        header.addLayout(title_box)
        header.addStretch()
        layout.addLayout(header)
        return page, layout

    def _create_video_page(self) -> QWidget:
        page, layout = self._create_page_shell("总览", "本地桌面工作台")
        self.state_badge = QLabel("空闲")
        self.state_badge.setObjectName("stateBadge")
        layout.itemAt(0).layout().addWidget(self.state_badge)

        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(18)
        grid.addWidget(self._create_material_card(), 0, 0)
        grid.addWidget(self._create_output_card(), 0, 1)
        grid.addWidget(self._create_run_card(), 1, 0, 1, 2)
        grid.setColumnStretch(0, 11)
        grid.setColumnStretch(1, 10)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 0)
        layout.addLayout(grid, 1)
        return page

    def _create_subtitle_page(self) -> QWidget:
        page, layout = self._create_page_shell("字幕设置", "语言、字幕输出模式和字幕能力")
        layout.addWidget(self.settings_sections.subtitle_group)
        layout.addStretch()
        return page

    def _create_dubbing_page(self) -> QWidget:
        page, layout = self._create_page_shell("配音设置", "Edge-TTS 可直接使用，IndexTTS2 保留接口")
        layout.addWidget(self.settings_sections.tts_group)
        layout.addStretch()
        return page

    def _create_llm_page(self) -> QWidget:
        page, layout = self._create_page_shell("LLM 设置", "翻译模型、API Key 和兼容接口")
        layout.addWidget(self.settings_sections.llm_group)
        layout.addStretch()
        return page

    def _create_advanced_page(self) -> QWidget:
        page, layout = self._create_page_shell("高级功能", "模型、目录、归档和清理")
        layout.addWidget(self.settings_sections.advanced_group)
        layout.addStretch()
        return page

    def _create_logs_page(self) -> QWidget:
        page, layout = self._create_page_shell("任务日志", "最近任务记录和输出路径")
        self.history_output = QPlainTextEdit()
        self.history_output.setReadOnly(True)
        self.history_output.setObjectName("logOutput")
        layout.addWidget(self.history_output)
        self._refresh_history_view()
        return page

    def _create_material_card(self) -> QFrame:
        card = MaterialCard()
        card.filesDropped.connect(self._on_files_selected)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("素材")
        title.setObjectName("cardTitle")
        header.addWidget(title)
        header.addStretch()

        self.reselect_btn = QPushButton("删除所有文件")
        self.reselect_btn.setObjectName("runToolButton")
        self.reselect_btn.clicked.connect(self._delete_all_material_files)
        header.addWidget(self.reselect_btn)
        layout.addLayout(header)

        self.material_file_row = QFrame()
        self.material_file_row.setObjectName("materialFileRow")
        row_layout = QHBoxLayout(self.material_file_row)
        row_layout.setContentsMargins(10, 7, 10, 7)
        self.file_label = QLabel("未选择文件")
        self.file_label.setObjectName("materialFileName")
        self.file_size_label = QLabel("")
        self.file_size_label.setObjectName("materialFileSize")
        self.file_size_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row_layout.addWidget(self.file_label, 1)
        row_layout.addWidget(self.file_size_label)
        layout.addWidget(self.material_file_row)

        self.file_path_label = QLabel("请选择视频或音频文件")
        self.file_path_label.setObjectName("materialPathLabel")
        layout.addWidget(self.file_path_label)

        self.material_file_list = QListWidget()
        self.material_file_list.setObjectName("materialFileList")
        self.material_file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.material_file_list.setMaximumHeight(128)
        self.material_file_list.setVisible(False)
        self.material_file_list.itemClicked.connect(
            lambda item: self._select_material_file_by_index(self.material_file_list.row(item))
        )
        self.material_file_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.material_file_list.customContextMenuRequested.connect(
            lambda pos: self._show_material_file_context_menu(pos, self.material_file_list)
        )
        layout.addWidget(self.material_file_list)

        self.preview_checkbox = QCheckBox("预览当前视频")
        self.preview_checkbox.setObjectName("materialPreviewCheck")
        self.preview_checkbox.setChecked(True)
        self._sync_preview_checkbox_text(True)
        self.preview_checkbox.toggled.connect(self._toggle_media_preview)
        layout.addWidget(self.preview_checkbox)

        self.video_preview = QVideoWidget()
        self.video_preview.setObjectName("videoPreview")
        self.video_preview.setMinimumHeight(210)
        self.video_preview.setVisible(False)
        layout.addWidget(self.video_preview)

        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_preview)

        upload_row = QHBoxLayout()
        self.upload_media_btn = QPushButton("上传视频或音频")
        self.upload_media_btn.setObjectName("uploadMediaButton")
        self.upload_media_btn.clicked.connect(self._select_media_file)
        upload_row.addWidget(self.upload_media_btn, 0, Qt.AlignLeft)
        upload_row.addStretch()
        layout.addLayout(upload_row)
        return card

    def _create_output_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("outputCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(8)
        title = QLabel("输出")
        title.setObjectName("outputTitle")
        header.addWidget(title)

        for text, handler, object_name in [
            ("output", lambda: self._open_folder(self.output_dir), "outputToolButton"),
            ("history", self._refresh_status, "outputToolButton"),
            ("项目归档", self._archive_current_project, "outputToolButton"),
            ("删除配音", self._delete_dubbing_outputs, "outputDangerButton"),
        ]:
            btn = QPushButton(text)
            btn.setObjectName(object_name)
            btn.clicked.connect(handler)
            header.addWidget(btn)
        header.addStretch()
        layout.addLayout(header)
        layout.addSpacing(6)

        self.output_rows = {
            "hard_subtitle_video": OutputRow("字幕视频"),
            "dubbed_video": OutputRow("配音视频"),
            "dubbed_audio": OutputRow("配音音频"),
            "soft_subtitle": OutputRow("译文字幕"),
            "bilingual_subtitle": OutputRow("双语字幕"),
            "original_subtitle": OutputRow("原文字幕"),
        }
        for row in self.output_rows.values():
            layout.addWidget(row)

        self.preview_label = QLabel("完成结果预览：还没有可预览的完成结果")
        self.preview_label.setObjectName("mutedLabel")
        layout.addWidget(self.preview_label)
        layout.addStretch()
        return card

    def _create_run_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("workCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel("运行")
        title.setObjectName("cardTitle")
        header.addWidget(title)
        header.addStretch()

        self.logs_folder_btn = QPushButton("日志文件夹")
        self.logs_folder_btn.setObjectName("runToolButton")
        self.logs_folder_btn.clicked.connect(lambda: self._open_folder(self.output_dir))
        header.addWidget(self.logs_folder_btn)
        layout.addLayout(header)

        separator = QFrame()
        separator.setObjectName("runSeparator")
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Plain)
        layout.addWidget(separator)

        resume_layout = QVBoxLayout()
        resume_header = QHBoxLayout()
        resume_title = QLabel("可继续任务")
        resume_title.setObjectName("mutedLabel")
        resume_header.addWidget(resume_title)
        resume_header.addStretch()

        self.refresh_resume_btn = QPushButton("刷新断点")
        self.refresh_resume_btn.setObjectName("runToolButton")
        self.refresh_resume_btn.clicked.connect(self._refresh_resumable_jobs)
        resume_header.addWidget(self.refresh_resume_btn)

        self.continue_resume_btn = QPushButton("继续任务")
        self.continue_resume_btn.setObjectName("runToolButton")
        self.continue_resume_btn.clicked.connect(self._continue_selected_resume_task)
        self.continue_resume_btn.setEnabled(False)
        resume_header.addWidget(self.continue_resume_btn)
        resume_layout.addLayout(resume_header)

        self.resume_job_list = QListWidget()
        self.resume_job_list.setObjectName("materialFileList")
        self.resume_job_list.setMaximumHeight(86)
        self.resume_job_list.itemDoubleClicked.connect(lambda _item: self._continue_selected_resume_task())
        resume_layout.addWidget(self.resume_job_list)
        layout.addLayout(resume_layout)

        btn_layout = QHBoxLayout()
        btn_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        btn_layout.addStretch()
        self.start_btn = QPushButton("一键翻译并配音")
        self.start_btn.setObjectName("startButton")
        self.start_btn.clicked.connect(lambda: self._start_translation("full"))
        btn_layout.addWidget(self.start_btn, 0, Qt.AlignLeft)

        self.subtitle_btn = QPushButton("只处理字幕")
        self.subtitle_btn.setObjectName("runToolButton")
        self.subtitle_btn.clicked.connect(lambda: self._start_translation("subtitles"))
        btn_layout.addWidget(self.subtitle_btn, 0, Qt.AlignLeft)

        self.dubbing_btn = QPushButton("只处理配音")
        self.dubbing_btn.setObjectName("runToolButton")
        self.dubbing_btn.clicked.connect(lambda: self._start_translation("dubbing"))
        btn_layout.addWidget(self.dubbing_btn, 0, Qt.AlignLeft)

        self.transcribe_btn = QPushButton("只转录")
        self.transcribe_btn.setObjectName("runToolButton")
        self.transcribe_btn.clicked.connect(lambda: self._start_translation("transcribe"))
        btn_layout.addWidget(self.transcribe_btn, 0, Qt.AlignLeft)

        self.retranslate_btn = QPushButton("重新翻译")
        self.retranslate_btn.setObjectName("runToolButton")
        self.retranslate_btn.clicked.connect(lambda: self._start_translation("retranslate"))
        btn_layout.addWidget(self.retranslate_btn, 0, Qt.AlignLeft)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("runToolButton")
        self.stop_btn.clicked.connect(self._cancel_processing)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setVisible(False)
        btn_layout.addWidget(self.stop_btn, 0, Qt.AlignLeft)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.progress_panel = ProgressPanel()
        self.progress_panel.setVisible(False)
        layout.addWidget(self.progress_panel)
        return card

    def _card(self, title_text: str) -> QFrame:
        card = QFrame()
        card.setObjectName("workCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)
        title = QLabel(title_text)
        title.setObjectName("cardTitle")
        layout.addWidget(title)
        return card

    def _switch_page(self, index: int):
        if hasattr(self, "stack"):
            self.stack.setCurrentIndex(index)
        for i, btn in enumerate(getattr(self, "nav_buttons", [])):
            btn.setProperty("active", i == index)
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _get_settings(self) -> dict:
        return self.settings_sections.get_settings()

    def _select_media_file(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择视频或音频文件，可多选",
            "",
            "Media files (*.mp4 *.mov *.avi *.webm *.mkv *.mp3 *.wav *.m4a);;All files (*.*)",
        )
        if paths:
            self._on_files_selected(paths)

    def _reselect_media_file(self):
        self._clear_selected_media()
        self._select_media_file()

    def _delete_all_material_files(self):
        if not self._current_inputs:
            return
        self._clear_selected_media()
        self.progress_panel.append_log("已删除全部素材文件")

    def _clear_selected_media(self):
        self._current_input = ""
        self._current_inputs = []
        self._last_results = {}
        self.media_player.stop()
        self.media_player.setSource(QUrl())
        self.video_preview.setVisible(False)
        self.file_label.setText("未选择文件")
        self.file_size_label.setText("")
        self.file_path_label.setText("请选择视频或音频文件")
        self.material_file_list.clear()
        self.material_file_list.setVisible(False)
        self._set_state("空闲")
        self._reset_outputs()

    def _on_files_selected(self, paths: list[str]):
        media_paths = [str(Path(path)) for path in paths if _is_supported_media_path(path)]
        if not media_paths:
            QMessageBox.warning(self, "提示", "请选择或拖入视频/音频文件。")
            return

        self._current_inputs = media_paths
        self._current_input = media_paths[0]
        self._last_results = {}
        self._update_material_summary()
        if len(media_paths) == 1:
            self.progress_panel.append_log(f"已选择文件: {media_paths[0]}")
        else:
            self.progress_panel.append_log(f"已选择 {len(media_paths)} 个文件")
        self._set_state("空闲")
        self._reset_outputs()
        self._toggle_media_preview(self.preview_checkbox.isChecked())

    def _toggle_media_preview(self, checked: bool):
        self._sync_preview_checkbox_text(checked)
        self._sync_material_file_list_visibility()
        if not checked or not self._current_input or is_url(self._current_input):
            self.media_player.stop()
            self.video_preview.setVisible(False)
            return

        file_path = Path(self._current_input)
        if not file_path.exists():
            self.video_preview.setVisible(False)
            return
        self.video_preview.setVisible(True)
        self.media_player.setSource(QUrl.fromLocalFile(str(file_path.resolve())))
        self.media_player.play()

    def _sync_preview_checkbox_text(self, checked: bool):
        self.preview_checkbox.setText("预览当前视频")

    def _update_material_summary(self):
        paths = [Path(path) for path in self._current_inputs]
        existing = [path for path in paths if path.exists()]
        total_size = sum(path.stat().st_size for path in existing)
        total_size_text = f"{total_size / 1024 / 1024:.2f} MB" if total_size else ""

        if len(paths) == 1:
            self.file_label.setText(paths[0].name)
            self.file_path_label.setText(str(paths[0]))
        else:
            self.file_label.setText(f"已选择 {len(paths)} 个文件")
            first = str(paths[0]) if paths else ""
            self.file_path_label.setText(f"{first} 等 {len(paths)} 个文件")
        self.file_size_label.setText(total_size_text)

        current_index = 0
        if self._current_input in self._current_inputs:
            current_index = self._current_inputs.index(self._current_input)

        self._refresh_material_file_list(paths, current_index)
        self._sync_material_file_list_visibility()

    def _select_material_file_by_index(self, index: int):
        if index < 0 or index >= len(self._current_inputs):
            return
        path = self._current_inputs[index]
        self._current_input = path
        self.file_path_label.setText(str(path))
        self.material_file_list.setCurrentRow(index)
        self._toggle_media_preview(self.preview_checkbox.isChecked())

    def _refresh_material_file_list(self, paths: list[Path], current_index: int):
        self.material_file_list.clear()
        for index, path in enumerate(paths):
            path_text = str(path)
            path = Path(path_text)
            item = QListWidgetItem(self._material_file_item_text(index, path))
            item.setData(Qt.UserRole, path_text)
            self.material_file_list.addItem(item)

        if paths:
            self.material_file_list.setCurrentRow(current_index)

    def _sync_material_file_list_visibility(self):
        self.material_file_list.setVisible(
            bool(self._current_inputs) and not self.preview_checkbox.isChecked()
        )

    def _show_material_file_context_menu(self, pos, list_widget: QListWidget):
        clicked_item = list_widget.itemAt(pos)
        selected_rows = self._selected_material_file_rows(list_widget)
        if clicked_item and list_widget.row(clicked_item) not in selected_rows:
            list_widget.clearSelection()
            clicked_item.setSelected(True)
            list_widget.setCurrentItem(clicked_item)
            selected_rows = [list_widget.row(clicked_item)]

        if not selected_rows:
            return

        menu = QMenu(self)
        label = "删除选中文件" if len(selected_rows) > 1 else "删除此文件"
        delete_action = menu.addAction(label)
        delete_action.triggered.connect(lambda: self._delete_selected_material_files(selected_rows))
        menu.exec(list_widget.viewport().mapToGlobal(pos))

    def _selected_material_file_rows(self, list_widget: QListWidget) -> list[int]:
        return sorted({list_widget.row(item) for item in list_widget.selectedItems()})

    def _material_file_item_text(self, index: int, path: Path) -> str:
        size_text = ""
        if path.exists():
            size_text = f" · {path.stat().st_size / 1024 / 1024:.2f} MB"
        return f"{index + 1}. {path.name}{size_text}"

    def _delete_selected_material_files(self, rows: list[int]):
        if not rows:
            return

        delete_rows = {row for row in rows if 0 <= row < len(self._current_inputs)}
        if not delete_rows:
            return

        remaining = [
            path for index, path in enumerate(self._current_inputs)
            if index not in delete_rows
        ]
        if not remaining:
            self._clear_selected_media()
            self.progress_panel.append_log("已从素材列表删除全部文件")
            return

        next_index = min(min(delete_rows), len(remaining) - 1)
        self._current_inputs = remaining
        self._current_input = remaining[next_index]
        self._last_results = {}
        self._update_material_summary()
        self.file_path_label.setText(self._current_input)
        self._toggle_media_preview(self.preview_checkbox.isChecked())
        self._reset_outputs()
        self.progress_panel.append_log(f"已从素材列表删除 {len(delete_rows)} 个文件")

    def _refresh_resumable_jobs(self):
        if not hasattr(self, "resume_job_list"):
            return
        try:
            from core.pipeline import discover_resumable_jobs

            self._resumable_jobs = discover_resumable_jobs(self.output_dir)
        except Exception as exc:
            self._resumable_jobs = []
            self.progress_panel.append_log(f"断点任务扫描失败: {exc}")

        self.resume_job_list.clear()
        if not self._resumable_jobs:
            item = QListWidgetItem("暂无可继续任务")
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            self.resume_job_list.addItem(item)
            self.continue_resume_btn.setEnabled(False)
            return

        for index, job in enumerate(self._resumable_jobs, 1):
            item = QListWidgetItem(self._resume_job_item_text(index, job))
            item.setData(Qt.UserRole, job.get("manifest_path", ""))
            self.resume_job_list.addItem(item)
        self.resume_job_list.setCurrentRow(0)
        busy = self.processing_thread is not None and self.processing_thread.isRunning()
        self.continue_resume_btn.setEnabled(not busy)

    def _resume_job_item_text(self, index: int, job: dict) -> str:
        input_path = Path(job.get("input_path") or job.get("input_video") or "")
        name = input_path.name or "未知素材"
        status = job.get("status", "unknown")
        stage = job.get("stage", "started")
        updated = job.get("updated_at") or job.get("created_at") or "-"
        return f"{index}. {name} · {status}/{stage} · {updated}"

    def _continue_selected_resume_task(self):
        if not self._resumable_jobs:
            return
        current = self.resume_job_list.currentItem()
        manifest_path = current.data(Qt.UserRole) if current else ""
        if not manifest_path:
            return

        job = next((item for item in self._resumable_jobs if item.get("manifest_path") == manifest_path), None)
        if not job:
            QMessageBox.warning(self, "提示", "断点任务不存在，请刷新后重试。")
            self._refresh_resumable_jobs()
            return

        settings = self._get_settings()
        job_settings = job.get("settings", {})
        results = job.get("results", {})
        needs_translation = (
            job_settings.get("process_mode", "full") != "transcribe"
            and not results.get("translated_segments")
            and not (Path(job.get("output_dir", "")) / "_translated_segments.json").exists()
        )
        if needs_translation and not settings["api_key"]:
            QMessageBox.warning(self, "提示", "该断点任务还没有完成翻译阶段，请先在 LLM 设置中填写 API Key。")
            self._switch_page(3)
            return

        input_path = job.get("input_path") or job.get("input_video") or ""
        if input_path and not is_url(input_path):
            self._current_inputs = [input_path]
            self._current_input = input_path
            if Path(input_path).exists():
                self._update_material_summary()
                self._toggle_media_preview(self.preview_checkbox.isChecked())
        self._last_results = dict(results)
        self._reset_outputs()
        self._update_outputs(self._last_results)
        self.progress_panel.clear()
        self.progress_panel.append_log(f"继续断点任务: {manifest_path}")
        self._set_busy(True)
        self._set_state("继续中")

        from core.pipeline import TranslationPipeline

        self.pipeline = TranslationPipeline(
            api_key=settings["api_key"],
            base_url=settings["base_url"],
            model=settings["model"],
            whisper_model=job_settings.get("whisper_model") or settings["whisper_model"],
            tts_engine=job_settings.get("tts_engine") or settings["tts_engine"],
            tts_config={
                "base_url": settings.get("indextts2_base_url", ""),
                "endpoint": settings.get("indextts2_endpoint", "/tts"),
                "api_key": settings.get("indextts2_api_key", ""),
            },
            output_dir=str(self.output_dir),
            temp_dir="./temp",
        )

        self.processing_thread = ProcessingThread(
            self.pipeline,
            input_path or [],
            settings,
            resume_manifest_path=manifest_path,
        )
        self.processing_thread.progress.connect(self._on_progress)
        self.processing_thread.finished.connect(self._on_finished)
        self.processing_thread.error.connect(self._on_error)
        self.processing_thread.cancelled.connect(self._on_cancelled)
        self.processing_thread.start()

    def _start_translation(self, run_mode: str = "full"):
        if not self._current_inputs:
            QMessageBox.warning(self, "提示", "请先选择视频文件或输入 URL")
            return

        settings = self._get_settings()
        settings["process_mode"] = run_mode
        if run_mode == "full":
            settings["tts_enabled"] = True
        elif run_mode == "subtitles":
            settings["tts_enabled"] = False
        elif run_mode == "dubbing":
            settings["tts_enabled"] = True
            settings["subtitle_mode"] = "none"
        elif run_mode == "transcribe":
            settings["tts_enabled"] = False
            settings["subtitle_mode"] = "none"
        elif run_mode == "retranslate":
            settings["tts_enabled"] = False
            original_subtitle = self._last_results.get("original_subtitle")
            if len(self._current_inputs) == 1 and original_subtitle and Path(original_subtitle).exists():
                settings["original_subtitle_path"] = original_subtitle

        needs_translation = run_mode != "transcribe"
        if needs_translation and not settings["api_key"]:
            QMessageBox.warning(self, "提示", "请先在 LLM 设置中填写 API Key")
            self._switch_page(3)
            return

        api_key = settings["api_key"].strip().lower()
        model = settings["model"].strip().lower()
        if needs_translation and (api_key in {"demo", "test", "local-demo"} or model.startswith("demo")):
            QMessageBox.warning(
                self,
                "需要真实翻译模型",
                "当前 LLM 设置是 demo 测试模式，它只会复制原文，不会把英文翻译成中文。请在 LLM 设置中选择真实提供商、填写 API Key 和模型后再运行。",
            )
            self._switch_page(3)
            return

        if settings["tts_enabled"] and settings["tts_engine"] == "indextts2" and not settings["indextts2_base_url"]:
            QMessageBox.warning(self, "提示", "你选择了 IndexTTS2，请先在配音设置中填写接口地址，或改用 Edge-TTS。")
            self._switch_page(2)
            return

        if settings["tts_enabled"]:
            from core.video_processor import is_demucs_available

            if not is_demucs_available():
                QMessageBox.warning(
                    self,
                    "需要安装 Demucs",
                    "配音翻译需要 Demucs 来分离人声和背景音乐。请先在配音设置中点击“安装 Demucs”。",
                )
                self._switch_page(2)
                return

        self._reset_outputs()
        self.progress_panel.clear()
        mode_labels = {
            "full": "一键翻译并配音",
            "subtitles": "只处理字幕",
            "dubbing": "只处理配音",
            "transcribe": "只转录",
            "retranslate": "重新翻译",
        }
        self.progress_panel.append_log(f"任务启动: {mode_labels.get(run_mode, run_mode)}")
        self._set_busy(True)
        self._set_state("处理中")

        from core.pipeline import TranslationPipeline

        self.pipeline = TranslationPipeline(
            api_key=settings["api_key"],
            base_url=settings["base_url"],
            model=settings["model"],
            whisper_model=settings["whisper_model"],
            tts_engine=settings["tts_engine"],
            tts_config={
                "base_url": settings.get("indextts2_base_url", ""),
                "endpoint": settings.get("indextts2_endpoint", "/tts"),
                "api_key": settings.get("indextts2_api_key", ""),
            },
            output_dir=str(self.output_dir),
            temp_dir="./temp",
        )

        self.processing_thread = ProcessingThread(self.pipeline, self._current_inputs, settings)
        self.processing_thread.progress.connect(self._on_progress)
        self.processing_thread.finished.connect(self._on_finished)
        self.processing_thread.error.connect(self._on_error)
        self.processing_thread.cancelled.connect(self._on_cancelled)
        self.processing_thread.start()

    def _cancel_processing(self):
        if self.pipeline:
            self.pipeline.cancel()
        self.progress_panel.append_log("正在停止任务...")
        self._set_state("停止中")
        self.stop_btn.setEnabled(False)

    def _on_progress(self, message: str, pct: float):
        self.progress_panel.set_progress(int(pct))
        self.progress_panel.set_status(message)
        self.progress_panel.append_log(message)
        self.statusBar().showMessage(message)

    def _on_finished(self, results: dict):
        display_results = results.get("last_result", results)
        self._last_results = display_results
        self._set_busy(False)
        self._set_state("完成")
        if results.get("batch_results"):
            self.progress_panel.append_log(f"批量任务完成: {len(results['batch_results'])} 个文件")
        else:
            self.progress_panel.append_log("任务完成")
        self._update_outputs(display_results)
        self._save_history(results, status="completed")
        self._refresh_resumable_jobs()
        QMessageBox.information(self, "完成", "视频翻译处理完成。")

    def _on_error(self, error_msg: str):
        self._set_busy(False)
        self._set_state("出错")
        self.progress_panel.append_log(f"错误: {error_msg}")
        for row in self.output_rows.values():
            if row.path is None:
                row.set_failed()
        self._save_history({"input_video": self._current_inputs or self._current_input, "error": error_msg}, status="failed")
        self._refresh_resumable_jobs()
        QMessageBox.critical(self, "错误", error_msg)

    def _on_cancelled(self, message: str):
        self._set_busy(False)
        self._set_state("已停止")
        self.progress_panel.append_log(message or "处理已停止")
        self._save_history({"input_video": self._current_inputs or self._current_input}, status="cancelled")
        self._refresh_resumable_jobs()

    def _update_outputs(self, results: dict):
        for key, row in self.output_rows.items():
            if key in results and results[key]:
                row.set_generated(results[key])
        preview = results.get("hard_subtitle_video") or results.get("dubbed_video")
        if preview:
            self.preview_label.setText(f"完成结果预览：{preview}")

    def _reset_outputs(self):
        if not hasattr(self, "output_rows"):
            return
        for row in self.output_rows.values():
            row.reset()
        self.preview_label.setText("完成结果预览：还没有可预览的完成结果")

    def _set_busy(self, busy: bool):
        self.start_btn.setEnabled(not busy)
        self.subtitle_btn.setEnabled(not busy)
        self.dubbing_btn.setEnabled(not busy)
        self.transcribe_btn.setEnabled(not busy)
        self.retranslate_btn.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)
        self.stop_btn.setVisible(busy)
        self.upload_media_btn.setEnabled(not busy)
        self.reselect_btn.setEnabled(not busy)
        self.preview_checkbox.setEnabled(not busy)
        self.material_file_list.setEnabled(not busy)
        self.logs_folder_btn.setEnabled(not busy)
        if hasattr(self, "refresh_resume_btn"):
            self.refresh_resume_btn.setEnabled(not busy)
        if hasattr(self, "continue_resume_btn"):
            self.continue_resume_btn.setEnabled(not busy and bool(self._resumable_jobs))
        if hasattr(self, "resume_job_list"):
            self.resume_job_list.setEnabled(not busy)
        self.progress_panel.setVisible(busy or self.progress_panel.log_output.blockCount() > 1)

    def _set_state(self, text: str):
        self.state_badge.setText(text)
        self.statusBar().showMessage(text)

    def _refresh_status(self):
        self.progress_panel.append_log("状态已刷新")
        self._update_outputs(self._last_results)
        self._refresh_history_view()
        self._refresh_resumable_jobs()

    def _open_folder(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder.resolve())))

    def _save_history(self, results: dict, status: str):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        history = []
        if self.history_path.exists():
            try:
                history = json.loads(self.history_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                history = []
        history.insert(0, {
            "time": datetime.now().isoformat(timespec="seconds"),
            "status": status,
            "input": self._current_inputs if len(self._current_inputs) > 1 else self._current_input,
            "results": results,
        })
        self.history_path.write_text(json.dumps(history[:100], ensure_ascii=False, indent=2), encoding="utf-8")
        self._refresh_history_view()

    def _refresh_history_view(self):
        if not hasattr(self, "history_output"):
            return
        if not self.history_path.exists():
            self.history_output.setPlainText("暂无历史记录。")
            return
        try:
            history = json.loads(self.history_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self.history_output.setPlainText("历史记录文件损坏，无法读取。")
            return

        lines = []
        for item in history[:50]:
            lines.append(f"[{item.get('time', '-')}] {item.get('status', '-')}")
            lines.append(f"输入: {item.get('input', '-')}")
            results = item.get("results", {})
            for key in ["soft_subtitle", "bilingual_subtitle", "dubbed_audio", "dubbed_video", "hard_subtitle_video"]:
                if results.get(key):
                    lines.append(f"  {key}: {results[key]}")
            if results.get("error"):
                lines.append(f"  error: {results['error']}")
            lines.append("")
        self.history_output.setPlainText("\n".join(lines) or "暂无历史记录。")

    def _archive_current_project(self):
        if not self._last_results:
            QMessageBox.information(self, "提示", "当前还没有可归档的项目。")
            return

        archives_dir = self.output_dir / "archives"
        archives_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_root = archives_dir / f"project_{timestamp}"
        archive_root.mkdir(parents=True, exist_ok=True)

        manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "input": self._current_inputs if len(self._current_inputs) > 1 else self._current_input,
            "results": self._last_results,
        }
        (archive_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        for value in self._last_results.values():
            if not isinstance(value, str):
                continue
            path = Path(value)
            if path.exists() and path.is_file():
                shutil.copy2(path, archive_root / path.name)

        zip_path = shutil.make_archive(str(archive_root), "zip", archive_root)
        self.progress_panel.append_log(f"项目已归档: {zip_path}")
        QMessageBox.information(self, "完成", f"项目已归档:\n{zip_path}")

    def _delete_dubbing_outputs(self):
        removed = []
        for key in ["dubbed_audio", "dubbed_video"]:
            path_text = self._last_results.get(key)
            if not path_text:
                continue
            path = Path(path_text)
            if path.exists() and path.is_file():
                path.unlink()
                removed.append(str(path))
            self._last_results.pop(key, None)

        self._reset_outputs()
        self._update_outputs(self._last_results)
        if removed:
            self.progress_panel.append_log("已删除配音输出: " + ", ".join(removed))
        else:
            self.progress_panel.append_log("没有可删除的配音输出")

    def closeEvent(self, event):
        self.settings_sections.save_settings()
        if self.processing_thread and self.processing_thread.isRunning():
            if self.pipeline:
                self.pipeline.cancel()
            if not self.processing_thread.wait(3000):
                QMessageBox.information(self, "正在停止", "任务正在停止中，请稍后再关闭软件。")
                event.ignore()
                return
        event.accept()


def _is_supported_media_path(path: str) -> bool:
    return Path(path).suffix.lower() in MEDIA_EXTENSIONS


def _media_paths_from_urls(urls: list[QUrl]) -> list[str]:
    paths = []
    for url in urls:
        path = url.toLocalFile()
        if path and _is_supported_media_path(path):
            paths.append(path)
    return paths
