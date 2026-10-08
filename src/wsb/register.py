"""Build registry/source_register.csv from sources.yaml + raw_manifest.jsonl."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict

import yaml

from wsb.lineage import ROOT

FIELDS = [
    "source_id", "publisher", "url", "version", "retrieval_time_utc", "license", "license_evidence",
    "local_path", "n_files", "bytes", "sha256", "format", "source_grain", "coverage", "provenance",
    "inspection_level", "counts_basis", "role", "access_note",
]


def build() -> list[dict]:
    meta = yaml.safe_load((ROOT / "registry" / "sources.yaml").read_text())
    files: dict[str, dict[str, dict]] = defaultdict(dict)
    first_seen: dict[str, str] = {}
    for line in (ROOT / "registry" / "raw_manifest.jsonl").read_text().splitlines():
        e = json.loads(line)
        files[e["source_id"]][e["local_path"]] = e  # last check wins; checksum must not change
        if e.get("status") == "downloaded":
            first_seen.setdefault(e["local_path"], e["checked_at"])
    rows = []
    for m in meta:
        fs = files.get(m["source_id"], {})
        times = sorted(first_seen.get(p, e["checked_at"]) for p, e in fs.items())
        row = {k: m.get(k) for k in FIELDS}
        row["n_files"] = len(fs)
        row["bytes"] = sum(e["bytes"] for e in fs.values())
        row["local_path"] = f"data/raw/{m['source_id']}/" if fs else None
        row["sha256"] = next(iter(fs.values()))["sha256"] if len(fs) == 1 else (
            f"{len(fs)} files; see registry/raw_manifest.jsonl" if fs else None)
        row["retrieval_time_utc"] = (times[0] if len(times) == 1 else f"{times[0]}..{times[-1]}") if times else None
        rows.append(row)
    with (ROOT / "registry" / "source_register.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


if __name__ == "__main__":
    for r in build():
        print(r["source_id"], r["inspection_level"], r["n_files"], r["bytes"])
    sys.exit(0)
