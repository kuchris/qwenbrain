"""Download the pinned model into this project and record content hashes."""

import hashlib
import json
from pathlib import Path

from .paths import ROOT, configure_storage

configure_storage()

from huggingface_hub import snapshot_download


def main():
    config = json.loads((ROOT / "configs/chat.json").read_text(encoding="utf-8"))
    print(f"Downloading {config['model_id']} at {config['model_revision']} into {ROOT / 'models'}", flush=True)
    snapshot = Path(snapshot_download(
        repo_id=config["model_id"], revision=config["model_revision"],
        cache_dir=ROOT / "models/hub",
        allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "LICENSE", "README.md"],
    ))
    files = []
    for path in sorted(snapshot.iterdir()):
        if not path.is_file():
            continue
        if not path.resolve().is_relative_to(ROOT):
            raise RuntimeError(f"Model file resolves outside the project: {path.name}")
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest})
    manifest = {"model_id": config["model_id"], "revision": config["model_revision"],
                "snapshot": snapshot.relative_to(ROOT).as_posix(), "files": files,
                "total_file_bytes": sum(f["bytes"] for f in files)}
    (ROOT / "models/download-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Ready: {len(files)} files, {manifest['total_file_bytes'] / 2**30:.2f} GiB. Manifest: models/download-manifest.json")


if __name__ == "__main__":
    main()
