"""Rebuild every data artifact from public sources: download, stage, clean, profile, calibrate, register.

    uv run python -m wsb.pipeline              # everything except the slow Wayback fetch
    uv run python -m wsb.pipeline --wayback    # also fetch archived USDA reports (~119 files, rate-limited)

A source that is unavailable (BPI 2019 was down for publisher maintenance on 2026-10-07) is recorded
and skipped; the run continues. Raw files are checksummed in registry/raw_manifest.jsonl and made read-only.
"""

from __future__ import annotations

import argparse
import sys

from wsb import acquire, register
from wsb.sources import freshretail, uci_retail, usda_ams, wwi_schema


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wayback", action="store_true", help="also fetch Internet Archive captures of USDA bh_fv020")
    args = ap.parse_args(argv)
    status = {}
    for sid in ("freshretailnet_50k", "uci_online_retail_ii", "wide_world_importers", "bpi_challenge_2019"):
        try:
            acquire.main([sid])
            status[sid] = "ok"
        except Exception as e:  # recorded, not fatal: e.g. publisher maintenance
            status[sid] = f"unavailable: {type(e).__name__}: {e}"[:200]
    usda_ams.acquire_legacy()
    if args.wayback:
        usda_ams.acquire_wayback()
    uci_retail.main()
    freshretail.main()
    usda_ams.parse_all()
    usda_ams.calibrate()
    wwi_schema.main()
    register.build()
    print("\nsource status:")
    for sid, st in status.items():
        print(f"  {sid}: {st}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
