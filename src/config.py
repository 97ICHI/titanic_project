from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Загружает YAML-конфиг и возвращает обычный словарь.
    """

    config_path = Path(path)

    with open(config_path, "r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)

    if not isinstance(config_data, dict):
        raise ValueError("Config must be a YAML mapping.")

    return config_data