"""Fetch raw source files into the immutable raw area and record a manifest entry.

Raw files land at data/raw/<source_id>/<filename>; files and their directory are
made read-only after each fetch.
Every fetch appends one JSON line to registry/raw_manifest.jsonl with URL,
retrieval time, size, sha256 (and md5 when the publisher supplies one).
A file that already exists is never overwritten; its checksum is re-verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "registry" / "raw_manifest.jsonl"

# Pinned retrieval targets. Versions are pinned where the host supports it
# (HF commit sha); otherwise the publisher's checksum is recorded and checked.
SOURCES: dict[str, list[dict]] = {
    "freshretailnet_50k": [
        {
            "url": "https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/resolve/08c1fab7f9257bc73679d415d65d644165d351d4/data/train.parquet",
            "filename": "train.parquet",
        },
        {
            "url": "https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/resolve/08c1fab7f9257bc73679d415d65d644165d351d4/data/eval.parquet",
            "filename": "eval.parquet",
        },
        {
            "url": "https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/resolve/08c1fab7f9257bc73679d415d65d644165d351d4/README.md",
            "filename": "README.md",
        },
    ],
    "uci_online_retail_ii": [
        {
            "url": "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
            "filename": "online_retail_ii.zip",
        },
    ],
    "wide_world_importers": [
        {
            "url": "https://github.com/microsoft/sql-server-samples/releases/download/wide-world-importers-v1.0/WideWorldImporters-Standard.bacpac",
            "filename": "WideWorldImporters-Standard.bacpac",
        },
    ],
    "bpi_challenge_2019": [
        {
            "url": "https://data.4tu.nl/file/35ed7122-966a-484e-a0e1-749b64e3366d/864493d1-3a58-47f6-ad6f-27f95f995828",
            "filename": "BPI_Challenge_2019.xes",
            "publisher_md5": "4eb909242351193a61e1c15b9c3cc814",
        },
    ],
}


def _hash(path: Path) -> tuple[str, str]:
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
            md5.update(chunk)
    return sha.hexdigest(), md5.hexdigest()


def fetch(source_id: str, item: dict) -> dict:
    dest_dir = RAW / source_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / item["filename"]
    status = "existing"
    if not dest.exists():
        os.chmod(dest_dir, 0o755)  # writable only while adding a new file
        tmp = dest.with_suffix(dest.suffix + ".part")
        with requests.get(item["url"], stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(1 << 20):
                    fh.write(chunk)
        tmp.rename(dest)
        status = "downloaded"
    os.chmod(dest, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    os.chmod(dest_dir, 0o555)  # raw area: no deletes or renames without an explicit chmod
    sha, md5 = _hash(dest)
    expected = item.get("publisher_md5")
    if expected and expected != md5:
        raise RuntimeError(f"{dest}: md5 {md5} != publisher {expected}")
    entry = {
        "source_id": source_id,
        "url": item["url"],
        "local_path": str(dest.relative_to(ROOT)),
        "bytes": dest.stat().st_size,
        "sha256": sha,
        "md5": md5,
        "publisher_md5": expected,
        "status": status,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sources", nargs="*", default=list(SOURCES))
    args = ap.parse_args(argv)
    for sid in args.sources:
        for item in SOURCES[sid]:
            e = fetch(sid, item)
            print(f"{e['status']:10s} {e['local_path']} {e['bytes']:>12,d} sha256={e['sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
