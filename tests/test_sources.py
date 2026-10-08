"""Source-pipeline checks. Parser tests run anywhere; data checks skip when the
(git-ignored) raw or derived files are absent."""

from __future__ import annotations

import hashlib
import json

import pytest

from wsb.lineage import ROOT
from wsb.sources.usda_ams import _pkg_weight_lb, parse_report

SAMPLE = """
BOSTON Terminal Prices as of 03-JAN-2024

VEGETABLES
---EGGPLANT:  MARKET STEADY.  1 1/9 bushel cartons FL med 26.00-28.00 lge
23.00-24.00 GA med 24.00 30 lb cartons HD Chinese med 52.00
---GARLIC:  MARKET STEADY.  30 lb cartons CA White super col 80.00-84.00 mostly 82.00 cartons
20 1-lb plastic jars CQ Peeled 80.00
"""


def test_parser_carries_package_and_origin_forward():
    rows, bad, meta = parse_report(SAMPLE, "sample")
    assert meta["market"] == "Boston" and meta["report_date"] == "2024-01-03"
    egg = [r for r in rows if r["commodity"] == "EGGPLANT"]
    assert [(r["package_source"], r["origin_code"], r["price_low"], r["price_high"]) for r in egg] == [
        ("1 1/9 bushel cartons", "FL", 26.0, 28.0),
        ("1 1/9 bushel cartons", "FL", 23.0, 24.0),
        ("1 1/9 bushel cartons", "GA", 24.0, 24.0),
        ("30 lb cartons", "HD", 52.0, 52.0),
    ]
    garlic = [r for r in rows if r["commodity"] == "GARLIC"]
    assert garlic[0]["mostly_low"] == 82.0 and garlic[0]["mostly_high"] == 82.0
    assert garlic[0]["package_weight_lb_derived"] == 30.0


def test_weight_only_from_outer_package():
    assert _pkg_weight_lb("30 lb cartons") == 30.0
    assert _pkg_weight_lb("5 kg/11 lb cartons") == 11.0
    assert _pkg_weight_lb("cartons 20 1-lb plastic jars") is None  # inner packs, no carton weight
    assert _pkg_weight_lb("1 1/9 bushel cartons") is None  # volume, no weight without an assumption


def _need(path):
    if not path.exists():
        pytest.skip(f"{path} not present (data is git-ignored; run the pipeline first)")
    return path


def test_raw_checksums_match_manifest():
    man = _need(ROOT / "registry" / "raw_manifest.jsonl")
    latest = {}
    for line in man.read_text().splitlines():
        e = json.loads(line)
        latest[e["local_path"]] = e["sha256"]
    checked = 0
    for rel, sha in latest.items():
        p = ROOT / rel
        if p.exists() and p.stat().st_size < 200_000_000:
            assert hashlib.sha256(p.read_bytes()).hexdigest() == sha, rel
            checked += 1
    if checked == 0:
        pytest.skip("no raw files present (run `python -m wsb.pipeline` to download them)")


def test_uci_cleaning_drops_no_rows():
    import duckdb

    staged = _need(ROOT / "data/staged/uci_online_retail_ii/lines.parquet")
    clean = _need(ROOT / "data/clean/uci_online_retail_ii/lines.parquet")
    quar = _need(ROOT / "data/quarantine/uci_online_retail_ii/lines.parquet")
    n = lambda p: duckdb.sql(f"select count(*) from '{p}'").fetchone()[0]
    assert n(clean) + n(quar) == n(staged)
    ids = duckdb.sql(f"select count(distinct row_id) from (select row_id from '{clean}' union all select row_id from '{quar}')").fetchone()[0]
    assert ids == n(staged)


def test_freshretail_profile_semantics():
    prof = json.loads(_need(ROOT / "reports/freshretailnet_50k/profile.json").read_text())
    assert prof["stock_cnt_equals_sum_slots_6_21"] == 1.0
    assert prof["daily_equals_sum_hourly_max_abs_err"] < 1e-9
