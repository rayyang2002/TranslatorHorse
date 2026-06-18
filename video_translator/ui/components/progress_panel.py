"""Progress and task log panel."""

from datetime import datetime

from PySide6.QtWidgets import QLabel, QPlainTextEdit, QProgressBar, QVBoxLayout, QWidget


class ProgressPanel(QWidget):
    """Displays processing progress and log output."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("等待处理...")
        self.status_label.setStyleSheet("color: #8fa3bd; font-size: 12px;")
        layout.addWidget(self.status_label)

        self.log_output = QPlainTextEdit()
        self.log_output.setObjectName("logOutput")
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumBlockCount(800)
        self.log_output.setMinimumHeight(150)
        layout.addWidget(self.log_output)

    def set_progress(self, value: int):
        self.progress_bar.setValue(value)

    def set_status(self, text: str):
        self.status_label.setText(text)

    def append_log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_output.appendPlainText(f"[{timestamp}] {message}")
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear(self):
        self.progress_bar.setValue(0)
        self.status_label.setText("等待处理...")
        self.log_output.clear()
