#!/usr/bin/env python3
"""
Install SeSMap model assets declared in model_requirements.json.

Usage:
  python scripts/install_models.py

The BGE encoder is downloaded from Hugging Face. Private/checkpoint files can be
provided by URL through .env, for example:
  SESMAP_MAPPER_CKPT_URL=https://...
  SESMAP_MAPPER_CKPT_SHA256=<optional sha256>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = BACKEND_DIR / "model_requirements.json"


def load_dotenv_light(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sha256(path: Path, expected: str | None) -> None:
    if not expected:
        return
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"SHA256 mismatch for {path}: expected {expected}, got {actual}"
        )


def install_huggingface_snapshot(model: dict, force: bool) -> None:
    local_dir = BACKEND_DIR / model["local_dir"]
    if local_dir.exists() and any(local_dir.iterdir()) and not force:
        print(f"[ok] {model['name']} already exists at {local_dir}")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency huggingface_hub. Run: python -m pip install -r requirements.txt"
        ) from exc

    local_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {model['repo_id']} -> {local_dir}")
    snapshot_download(
        repo_id=model["repo_id"],
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=model.get("ignore_patterns") or None,
    )
    print(f"[ok] installed {model['name']}")


def install_file(model: dict, force: bool) -> bool:
    local_path = BACKEND_DIR / model["local_path"]
    expected_hash = os.getenv(model.get("sha256_env", "") or "")

    if local_path.exists() and not force:
        verify_sha256(local_path, expected_hash)
        print(f"[ok] {model['name']} already exists at {local_path}")
        return True

    url = os.getenv(model.get("url_env", "") or "")
    if not url:
        print(
            f"[missing] {model['name']} is not in GitHub and no download URL was provided.\n"
            f"          Put the file at: {local_path}\n"
            f"          Or set {model.get('url_env')} in SeSMap-backend/.env and rerun this script."
        )
        return not model.get("required", False)

    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_suffix(local_path.suffix + ".download")
    print(f"[download] {model['name']} -> {local_path}")
    urllib.request.urlretrieve(url, tmp_path)
    verify_sha256(tmp_path, expected_hash)
    tmp_path.replace(local_path)
    print(f"[ok] installed {model['name']}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--force", action="store_true", help="Re-download existing model assets.")
    args = parser.parse_args()

    load_dotenv_light(BACKEND_DIR / ".env")

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []

    for model in manifest.get("models", []):
        try:
            kind = model.get("kind")
            if kind == "huggingface_snapshot":
                install_huggingface_snapshot(model, args.force)
            elif kind == "file":
                ok = install_file(model, args.force)
                if not ok:
                    failures.append(model["name"])
            else:
                raise RuntimeError(f"Unknown model kind: {kind}")
        except Exception as exc:
            failures.append(model.get("name", "<unknown>"))
            print(f"[error] {model.get('name', '<unknown>')}: {exc}", file=sys.stderr)

    if failures:
        print("\nModel installation incomplete:", ", ".join(failures), file=sys.stderr)
        return 2

    print("\nAll model assets are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
