"""Download the public fruit detection dataset into data/raw/.

Source: GitHub mirror lightly-ai/dataset_fruits_detection (CC0-1.0), a copy of the
Kaggle dataset "lakshaytyagi01/fruit-detection". We download a ZIP of a pinned commit
(set in config.yaml) so every team member and the Kaggle notebook get identical files.

Usage (from the project root):
    python -m src.data.download_dataset
    python -m src.data.download_dataset --force     # re-download
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from src.utils.helpers import load_config, resolve_path


def download_file(url: str, dest: Path) -> None:
    """Stream a URL to disk while printing simple progress."""
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as response, open(dest, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1 << 20):  # 1 MB chunks
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
            else:
                print(f"\r  {done / 1e6:7.1f} MB", end="", flush=True)
    print()


def download_dataset(force: bool = False, config_path: str | None = None) -> Path:
    cfg = load_config(config_path)["dataset"]
    raw_dir = resolve_path(cfg["raw_dir"])

    if raw_dir.exists() and (raw_dir / "data.yaml").is_file() and not force:
        print(f"Dataset already present at {raw_dir} (use --force to re-download).")
        return raw_dir

    repo, commit = cfg["github_repo"], cfg["github_commit"]
    url = f"https://github.com/{repo}/archive/{commit}.zip"
    raw_dir.parent.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir.parent / "_dataset_download.zip"

    try:
        download_file(url, zip_path)
        print("Extracting ...")
        with zipfile.ZipFile(zip_path) as zf:
            top_level = {name.split("/")[0] for name in zf.namelist()}
            if len(top_level) != 1:
                raise RuntimeError(f"Unexpected archive layout: {sorted(top_level)[:5]}")
            # zipfile.extractall() strips absolute paths and '..' components (safe extraction).
            zf.extractall(raw_dir.parent)
        extracted = raw_dir.parent / top_level.pop()
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        extracted.rename(raw_dir)
    finally:
        if zip_path.exists():
            zip_path.unlink()  # free ~500 MB of disk space

    # Record exactly what was downloaded.
    (raw_dir / "SOURCE.txt").write_text(f"repo: {repo}\ncommit: {commit}\nurl: {url}\n", encoding="utf-8")
    print(f"Dataset ready at {raw_dir}")
    return raw_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download even if the dataset exists")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args()
    try:
        download_dataset(force=args.force, config_path=args.config)
    except Exception as exc:  # report clearly instead of a long traceback
        print(f"ERROR: dataset download failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
