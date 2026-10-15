"""RefGrade standalone config - JSON in %APPDATA%/RefGrade/config.json"""
import json
import os
from pathlib import Path
from typing import Optional

CONFIG_DIR = Path(os.environ.get("APPDATA", "")) / "RefGrade"
CONFIG_FILE = CONFIG_DIR / "config.json"


def load_config() -> dict:
    """Load config. Returns {} if file missing or corrupt."""
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def save_config(cfg: dict) -> None:
    """Save config atomically (temp + replace)."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    tmp.replace(CONFIG_FILE)


def get_torch_path() -> Optional[Path]:
    """Get saved torch path from config."""
    cfg = load_config()
    p = cfg.get("torch_path")
    return Path(p) if p else None


def set_torch_path(path: Path) -> None:
    """Save torch path to config."""
    cfg = load_config()
    cfg["torch_path"] = str(path)
    save_config(cfg)


def clear_torch_path() -> None:
    """Remove torch path from config."""
    cfg = load_config()
    cfg.pop("torch_path", None)
    save_config(cfg)
