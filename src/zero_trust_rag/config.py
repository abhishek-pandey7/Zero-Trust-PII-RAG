import os
import yaml
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).parent / "config.yaml"


class Config:
    _instance = None
    _data: dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                self._data = yaml.safe_load(f) or {}
        else:
            self._data = {}

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        result = self.get(key)
        if result is None:
            raise KeyError(f"Config key '{key}' not found")
        return result

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None