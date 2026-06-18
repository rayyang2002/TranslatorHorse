"""配置管理模块"""

import json
import os
from pathlib import Path
from typing import Any


class Config:
    """应用配置管理器"""
    
    _instance: "Config" = None
    _config: dict[str, Any] = {}
    
    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load(self, config_path: str | Path | None = None) -> None:
        """加载配置文件"""
        if config_path is None:
            config_path = Path(__file__).parent / "settings.json"
        else:
            config_path = Path(config_path)
        
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self._config = json.load(f)
        else:
            self._config = self._get_defaults()
    
    def _get_defaults(self) -> dict[str, Any]:
        """默认配置"""
        return {
            "app_name": "Video Translator",
            "version": "0.1.0",
            "tts_engine": "edge-tts",
            "source_lang": "auto",
            "target_lang": "zh",
            "whisper_model": "base",
            "output_dir": "./output",
            "temp_dir": "./temp",
            "subtitle_style": {
                "font": "Microsoft YaHei",
                "font_size": 24,
                "color": "#FFFFFF",
                "background": "#00000080",
                "position": "bottom"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """设置配置项"""
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def save(self, config_path: str | Path | None = None) -> None:
        """保存配置文件"""
        if config_path is None:
            config_path = Path(__file__).parent / "settings.json"
        else:
            config_path = Path(config_path)
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)


# 全局配置实例
config = Config()
