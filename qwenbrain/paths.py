"""Configure model storage before importing Hugging Face or PyTorch."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configure_storage():
    home = ROOT / ".cache" / "huggingface"
    paths = {
        "HF_HOME": home,
        "HF_HUB_CACHE": ROOT / "models" / "hub",
        "HUGGINGFACE_HUB_CACHE": ROOT / "models" / "hub",
        "HF_XET_CACHE": home / "xet",
        "HF_ASSETS_CACHE": home / "assets",
        "HF_DATASETS_CACHE": ROOT / "data" / "cache",
        "TORCH_HOME": ROOT / ".cache" / "torch",
        "TEMP": ROOT / ".cache" / "tmp",
        "TMP": ROOT / ".cache" / "tmp",
    }
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ.pop("TRANSFORMERS_CACHE", None)
    return paths
