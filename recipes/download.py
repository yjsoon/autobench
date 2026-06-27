#!/usr/bin/env python3
"""Download the BF16 REAP source checkpoint to local NVMe.

Run:  source config.env && uv run python download.py
Resumable: re-running skips already-complete shards.
"""
import os
import sys

from huggingface_hub import snapshot_download

REPO = os.environ["SRC_REPO"]
DST = os.environ["SRC_DIR"]


def main() -> int:
    os.makedirs(DST, exist_ok=True)
    print(f"Downloading {REPO}  ->  {DST}", flush=True)
    path = snapshot_download(
        repo_id=REPO,
        local_dir=DST,
        # weights + the config/tokenizer files needed to load and re-export.
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "*.txt",
            "*.py",
            "tokenizer*",
            "*.model",
            "*.jinja",
        ],
        max_workers=8,
    )
    print(f"\nDone. Local snapshot at: {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
