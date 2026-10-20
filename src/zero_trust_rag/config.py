import os
from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigurationError

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _config_path() -> Path:
    """The config file to load: ZT_CONFIG_PATH if set, else the packaged config.yaml."""
    override = os.getenv("ZT_CONFIG_PATH")
    return Path(override) if override else DEFAULT_CONFIG_PATH


class Config:
    """Read-only access to config.yaml with dotted keys, e.g. config.get("presidio.score_threshold").

    A missing or unreadable file is an error rather than an empty config: silently falling back to
    defaults would mean a tightened security setting in the wrong file never takes effect.
    """

    _instance = None
    _data: dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        path = _config_path()
        if not path.is_file():
            raise ConfigurationError(f"Config file not found: {path}")
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ConfigurationError(f"Config file is empty or not a mapping: {path}")
        self._data = data

    @classmethod
    def reload(cls) -> "Config":
        """Drop the cached instance and read the file again (used by tests)."""
        cls._instance = None
        return cls()

    def get(self, key: str, default: Any = None) -> Any:
        value: Any = self._data
        for k in key.split("."):
            if not isinstance(value, dict) or k not in value:
                return default
            value = value[k]
        return value

    def __getitem__(self, key: str) -> Any:
        result = self.get(key)
        if result is None:
            raise KeyError(f"Config key '{key}' not found")
        return result

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None
