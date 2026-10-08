"""USDA AMS Specialty Crops Market News: keyless acquisition and terminal-report parsing.

Access boundary (verified 2026-10-07): the MARS API (structured history) needs
an API key, so this module uses only keyless material:
  1. legacy pre-MARS terminal-market text reports still served at
     www.ams.usda.gov/mnreports/<id>.txt. Each is a single frozen day
     (the last issue before the 2024-01-04 MARS switch);
  2. Internet Archive captures of one report (Boston vegetables, bh_fv020),
     a sparse set of dated snapshots, not a daily series;
  3. the current MARS-era PDF of the same report, kept as format evidence only.

A terminal report quotes wholesale *offering* price ranges per package at one
market on one day. It is not an executed transaction, not a volume, and not
evidence of a specific causal shock. Prices are USD per package as quoted. A
per-lb value is derived only when the package text states a weight; nothing is
pooled across package types.
"""

from __future__ import annotations

import hashlib
import re
import sys
import time
from datetime import datetime

import duckdb
import pandas as pd
import requests

from wsb.acquire import fetch
from wsb.lineage import ROOT, TransformLog, write_json

LEGACY_SID = "usda_ams_legacy_txt"
WAYBACK_SID = "usda_ams_wayback_bh_fv020"
CURRENT_SID = "usda_ams_current_pdf"
TERMINAL_IDS = (
    "aj_fv010 aj_fv020 aj_fv030 aj_fv040 as_fv010 as_fv020 as_fv030 bh_fv010 bh_fv020 bh_fv030 bh_fv040 "
    "bp_fv010 bp_fv020 bp_fv030 bp_fv040 ca_fv010 ca_fv020 ca_fv030 ca_fv040 du_fv010 du_fv020 du_fv030 "
    "du_fv040 hc_fv010 hc_fv020 hc_fv030 hc_fv040 hx_fv010 hx_fv020 hx_fv030 hx_fv040 mh_fv010 mh_fv020 "
    "mh_fv030 mh_fv040 na_fv010 na_fv020 na_fv030 na_fv040 nx_fv010 nx_fv020 nx_fv030 nx_fv040 ra_fv010 "
    "ra_fv020 ra_fv030"
).split()
CLEAN = ROOT / "data" / "clean" / "usda_ams"
QUAR = ROOT / "data" / "quarantine" / "usda_ams"
REPORT = ROOT / "reports" / "usda_ams"


# --------------------------------------------------------------------------- acquire

def acquire_legacy() -> list[dict]:
    out = []
    for rid in TERMINAL_IDS:
        try:
            out.append(fetch(LEGACY_SID, {"url": f"https://www.ams.usda.gov/mnreports/{rid}.txt", "filename": f"{rid}.txt"}))
        except requests.HTTPError as e:  # recorded, not fatal: some ids may only exist as PDF
            out.append({"source_id": LEGACY_SID, "url": rid, "error": str(e)})
    out.append(fetch(CURRENT_SID, {"url": "https://www.ams.usda.gov/mnreports/bh_fv020.pdf", "filename": "bh_fv020_current.pdf"}))
    return out


def acquire_wayback(report_id: str = "bh_fv020", pause: float = 1.5) -> list[dict]:
    target = f"www.ams.usda.gov/mnreports/{report_id}.txt"
    cdx = requests.get("https://web.archive.org/cdx/search/cdx",
                       params={"url": target, "output": "json", "filter": "statuscode:200", "collapse": "digest"},
                       timeout=120).json()[1:]
    out = []
    for row in cdx:
        ts = row[1]
        item = {"url": f"https://web.archive.org/web/{ts}id_/http://{target}", "filename": f"{report_id}_{ts}.txt"}
        for attempt in range(3):
            try:
                out.append(fetch(WAYBACK_SID, item))
                break
            except requests.RequestException as e:
                if attempt == 2:
                    out.append({"source_id": WAYBACK_SID, "url": item["url"], "error": str(e)})
                time.sleep(5 * (attempt + 1))
        time.sleep(pause)
    return out


# --------------------------------------------------------------------------- parse

HEADER = re.compile(r"^\s*(?P<market>[A-Z .'-]+?)\s+TERMINAL PRICES\s+AS OF\s+(?P<date>\d{1,2}-[A-Z]{3}-\d{2,4})", re.I | re.M)
COMMODITY = re.compile(r"^-{2,3}(?P<name>[A-Z][A-Z0-9 ,/&()'.-]*?):\s*(?P<body>.*)$")
# prices are d.dd or .dd (per-lb quotes such as ".65-.70"); a leading digit or dot must not precede them
PRICE = re.compile(r"(?<![\d.])(?P<mostly>mostly\s+)?(?P<lo>\d{0,4}\.\d{2})(?:-(?P<hi>\d{0,4}\.\d{2}))?(?![\d])")
TONE = re.compile(r"^(?P<tone>(?:MARKET|OFFERINGS|DEMAND|SUPPLIES|TRADING)\b[^.\d]*)\.\s*", re.I)
PKG_ALT = (r"cartons?|crates?|sacks?|bags?|filmbags|flmbags|flats?|containers?|cases|baskets?|lugs?|bins?|boxes|cups|"
           r"trays|bunches|RPCs?|packages|ctns|crts|cntrs|sks|bxs|flts|bskts|buctns|bucrts|bu|lugs")
PKG_WORD = re.compile(rf"^(?:{PKG_ALT})(?:/(?:{PKG_ALT}))*[,/]?$")  # lowercase only: "Flat" spinach is a variety
NUMLIKE = r"(\d+|\d+/\d+|\d+\.\d+|\d+-\d+|\d+(\.\d+)?-\d+(\.\d+)?|\d+-\d+/\d+)"
PKG_LEFT = re.compile(rf"^({NUMLIKE}|kg/\d+(\.\d+)?|\d+(\.\d+)?kg/\d+(\.\d+)?|\d+(\.\d+)?-?(lb|kg|oz|pt|qt|ct|bu|gal|gallon)s?\.?|lb|lbs|kg|oz|pt|bu|bushel|bushels|"
                      r"1/2|1/9|1/4|1/6|1/3|5/9|and|baled|film|mesh|bulk|master|container|tray|wire-bound|wirebound|plastic|"
                      r"\d+-layer|layer|1-layer|2-layer|\d+/\d+-bu|bu,|half|cell|pack|loose|bunched|wrapped|tray-pack)$", re.I)
PKG_RIGHT = re.compile(r"^(wrapped|loose|bunched|bchd|tray|pack|precooled|film|flmwrpd|filmwrapped|with|lids|tpd|bagged|"
                       r"\d+|\d+-\d+/\d+|\d+/\d+|\d+(\.\d+)?-?(lb|oz|pt|ct|kg)s?|lb|oz|pt|cups|containers|bags|trays|clamshells|jars|lyr|layer|"
                       r"\d+-lyr|bottles|sacks|mesh|plastic|clear|film-wrapped|topped)$", re.I)
WEIGHT = re.compile(r"(?<![\d.\-/])(\d+(?:\.\d+)?)\s*-?\s*(lb|lbs|kg)\b", re.I)
INNER_PACK = re.compile(r"\b\d+\s+\d+(?:\.\d+)?\s*-?\s*(lb|lbs|kg|oz|pt)\b", re.I)
BASIS = [("USD per lb", re.compile(r"\bper (lb|pound)\b", re.I)), ("USD each", re.compile(r"\beach\b", re.I)),
         ("USD per bin", re.compile(r"\bper bin\b", re.I)), ("USD per bunch", re.compile(r"\bper bunch\b", re.I))]
ANNOT = {"occas", "occasional", "occasionally", "few", "higher", "lower", "and", "&", "slightly", "one", "label",
         "(one", "label)", "best", "poorer", "mostly", "some", "fine", "appearance", "previous", "commitments",
         "low", "high", "as"}
NOT_VARIETY = {"U.S.", "Shprs", "Size", "No.", "No", "One", "Fancy", "Extra", "Choice", "Commercial", "Combination", "AIR"}
QUALITY_WORDS = {"poor", "fair", "fr", "ord", "good", "fine", "cond", "qual", "quality", "condition", "appear",
                 "fineappear", "mostly", "and", "&", "few", "occas"}
NO_OFFER = re.compile(r"(?i)no offerings|offerings insufficient[a-z ]*|insufficient offerings[a-z ]*")
REPACK = [(re.compile(r"\bLocal Repack(ed)?\b"), "LOCALREPACK"), (re.compile(r"\bRepacked Enroute\b"), "REPACKEDENROUTE"),
          (re.compile(r"\bLocal\b"), "LOCAL")]
# AMS origin abbreviations: US states plus the country codes seen in these reports.
ORIGINS = set(
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK "
    "OR PA RI SC SD TN TX UT VT VA WA WV WI WY PR "
    "MX CD CL PE HD GU DR SP NL BL IS AG CQ FR IT NZ AU CR EC CO PN NI BR SA MO TW JM EG PL TK GR VN IN TH KO JP ES".split()
) | {"NENG", "LOCAL", "LOCALREPACK", "REPACKEDENROUTE"}
SECTION = re.compile(r"^[A-Z][A-Z &/,-]{3,}$")


def _is_origin(tok: str) -> bool:
    parts = tok.strip(",").split("/")
    return all(p in ORIGINS for p in parts) and len(parts) <= 4


def _parse_date(s: str) -> str | None:
    for fmt in ("%d-%b-%Y", "%d-%b-%y"):
        try:
            return datetime.strptime(s.title(), fmt).date().isoformat()
        except ValueError:
            pass
    return None


def _pkg_weight_lb(pkg: str | None) -> float | None:
    """Carton weight in lb, only when the package states exactly one outer weight."""
    if not pkg or INNER_PACK.search(pkg) or re.search(r"\d+\s*-\s*\d+\s*(lb|kg)", pkg, re.I):
        return None  # inner packs or weight ranges: no single carton weight
    if m := re.search(r"(\d+(?:\.\d+)?)\s*kg\s*/\s*(\d+(?:\.\d+)?)\s*lb", pkg, re.I):
        return float(m.group(2))  # dual-unit statement of one carton weight: use the stated lb
    ws = WEIGHT.findall(pkg)
    if len(ws) != 1:
        return None
    v, unit = float(ws[0][0]), ws[0][1].lower()
    return v if unit.startswith("lb") else round(v * 2.20462, 3)


def _package_span(toks: list[str]) -> tuple[int, int] | None:
    """Token span of the LAST package phrase (the one nearest the price)."""
    spans = []
    i = 0
    while i < len(toks):
        if PKG_WORD.match(toks[i]):
            lo = i
            while lo > 0 and PKG_LEFT.match(toks[lo - 1]) and (not spans or lo - 1 >= spans[-1][1]):
                lo -= 1
            hi = i + 1
            while hi < len(toks) and (PKG_RIGHT.match(toks[hi]) or PKG_WORD.match(toks[hi])):
                hi += 1
            spans.append((lo, hi))
            i = hi
        else:
            i += 1
    if len(spans) >= 2 and toks[spans[-1][0]].lower() == "and" and spans[-2][1] == spans[-1][0]:
        return spans[-2][0], spans[-1][1]  # one quote for two packages: keep both, weight becomes ambiguous
    return spans[-1] if spans else None


def _split_variety(desc_toks: list[str]) -> tuple[str | None, str]:
    """Leading capitalised words are a variety/type ("Red Type", "Honeycrisp"); the rest is size/grade."""
    k = 0
    while k < len(desc_toks) and re.match(r"^[A-Z(][A-Za-z().'/-]*$", desc_toks[k]) and desc_toks[k] not in NOT_VARIETY:
        k += 1
    return (" ".join(desc_toks[:k]) or None), " ".join(desc_toks[k:])


def _blocks(text: str) -> list[dict]:
    blocks, section, cur, last = [], None, None, None
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            if cur:
                blocks.append(cur)
                last, cur = cur, None
            continue
        if m := COMMODITY.match(s):
            if cur:
                blocks.append(cur)
            cur = {"section": section, "commodity": m.group("name").strip(), "body": m.group("body").strip()}
        elif cur is not None:
            cur["body"] += " " + s
        elif SECTION.match(s) and "TERMINAL" not in s.upper():
            section, last = s, None
        elif last is not None and PRICE.search(s):
            last["body"] += " " + s  # continuation after a blank line inside a block
    if cur:
        blocks.append(cur)
    return blocks


def parse_report(text: str, file_id: str) -> tuple[list[dict], list[dict], dict]:
    text = text.replace("\r", "")
    h = HEADER.search(text)
    meta = {"file_id": file_id, "market": h.group("market").strip().title() if h else None,
            "report_date": _parse_date(h.group("date")) if h else None}
    rows, bad = [], []
    blocks = _blocks(text)
    n_price_tokens = 0
    cnt = {"quotes": 0, "mostly_attached": 0, "mostly_orphan": 0, "annotation_prices": 0}
    for b in blocks:
        body = re.sub(r"\s+", " ", b["body"])
        n_price_tokens += len(PRICE.findall(body))
        tones = []
        while m := TONE.match(body):
            tones.append(m.group("tone").strip())
            body = body[m.end():]
        tone = "; ".join(tones) or None
        pkg = origin = variety = None
        greenhouse = False
        last, quote, n_quotes = 0, None, 0
        for m in PRICE.finditer(body):
            ctx = body[last:m.start()].strip()
            last = m.end()
            ctx = NO_OFFER.split(ctx)[-1].strip()  # text of an item with no offerings belongs to no quote
            for rx, rep in REPACK:
                ctx = rx.sub(rep, ctx)
            if m.group("mostly"):
                if quote is not None:
                    quote["mostly_low"] = float(m.group("lo"))
                    quote["mostly_high"] = float(m.group("hi") or m.group("lo"))
                    cnt["mostly_attached"] += 1
                else:
                    cnt["mostly_orphan"] += 1
                continue
            toks = [t for t in ctx.split() if t != "GREENHOUSE"]
            greenhouse = greenhouse or "GREENHOUSE" in ctx
            # leading annotation words qualify the PREVIOUS quote ("occas higher", "few")
            lead = 0
            while lead < len(toks) and toks[lead].lower() in ANNOT:
                lead += 1
            annot, toks = " ".join(toks[:lead]), toks[lead:]
            if not toks and quote is not None and annot:
                quote.setdefault("annotations", []).append(f"{annot} {m.group(0)}")
                cnt["annotation_prices"] += 1
                continue
            if annot and quote is not None:
                quote.setdefault("annotations", []).append(annot)
            n_quotes += 1
            cnt["quotes"] += 1
            span = _package_span(toks)
            after = [i for i, t in enumerate(toks) if _is_origin(t) and (not span or i >= span[1])]
            before = [i for i, t in enumerate(toks) if _is_origin(t) and span and i < span[0]]
            o_idx = after[0] if after else (before[-1] if before else None)
            origin_carried = False
            if span:
                # a new package keeps the previous origin until a new one is named (flagged)
                pkg, variety, origin_carried = " ".join(toks[span[0]:span[1]]), None, origin is not None
            if o_idx is not None:
                if toks[o_idx].strip(",") != origin:
                    variety = None
                origin = toks[o_idx].strip(",")
            used = set(range(*span)) if span else set()
            if o_idx is not None:
                used.add(o_idx)
            local = [t for i, t in enumerate(toks) if i not in used]
            air = "AIR" in local
            v, spec = _split_variety([t for t in local if t != "AIR"])
            if v:
                variety = v
            if o_idx is not None:
                origin_carried = False
            spec_toks = spec.lower().replace("-", " ").split()
            if quote is not None and spec_toks and all(t in QUALITY_WORDS for t in spec_toks) and not span and o_idx is None:
                spec = f"{quote['size_grade_source'] or ''} {spec}".strip()  # condition sub-quote of the previous item
            basis = next((name for name, rx in BASIS if rx.search(ctx) or (pkg and rx.search(pkg))), "USD per package")
            quote = {
                **meta, "section": b["section"], "commodity": b["commodity"], "market_tone": tone,
                "organic": (b["section"] or "").upper().startswith("ORGANIC"), "greenhouse": greenhouse,
                "package_source": pkg, "origin_code": origin, "variety_carried": variety,
                "descriptor_source": " ".join(local) or None, "size_grade_source": spec or None,
                "price_low": float(m.group("lo")), "price_high": float(m.group("hi") or m.group("lo")),
                "mostly_low": None, "mostly_high": None, "price_basis": basis, "air_freight": air,
                "origin_carried_across_package": origin_carried, "raw_context": ctx[:200],
            }
            if pkg and origin:
                rows.append(quote)
            else:
                bad.append({**quote, "reason": "no package" if not pkg else "no origin"})
        if n_quotes == 0:
            bad.append({**meta, "section": b["section"], "commodity": b["commodity"], "market_tone": tone,
                        "raw_context": body[:200], "reason": "no price in block"})
    for r in rows + bad:
        r["annotations"] = "; ".join(r.get("annotations", [])) or None
    for r in rows:
        w = _pkg_weight_lb(r["package_source"]) if r["price_basis"] == "USD per package" else None
        r["package_weight_lb_derived"] = w
        r["usd_per_lb_mid_derived"] = round((r["price_low"] + r["price_high"]) / 2 / w, 4) if w else None
    return rows, bad, {**meta, "n_blocks": len(blocks), "n_quotes": len(rows), "n_unparsed": len(bad),
                       "n_price_tokens": n_price_tokens, **{f"n_{k}": v for k, v in cnt.items()}}


def parse_all() -> dict:
    log = TransformLog("usda_ams")
    all_rows, all_bad, metas = [], [], []
    for sid in (LEGACY_SID, WAYBACK_SID):
        for p in sorted((ROOT / "data" / "raw" / sid).glob("*.txt")):
            text = p.read_text(errors="replace")
            rows, bad, meta = parse_report(text, f"{sid}/{p.name}")
            for r in rows + bad:
                r["source_id"] = sid
            if sid == WAYBACK_SID:
                cap = p.stem.split("_")[-1]
                meta["capture_ts"] = cap
                for r in rows + bad:
                    r["available_by_utc"] = datetime.strptime(cap, "%Y%m%d%H%M%S").isoformat()
            all_rows += rows
            all_bad += bad
            metas.append(meta)
    df, bad = pd.DataFrame(all_rows), pd.DataFrame(all_bad)
    # Wayback can serve the same bytes under different capture timestamps. Keep, per identical file, the earliest
    # capture taken on or after its market date; any file captured before its own market date is excluded.
    m0 = pd.DataFrame(metas)
    m0["sha256"] = [hashlib.sha256((ROOT / "data" / "raw" / f).read_bytes()).hexdigest() for f in m0.file_id]
    m0["capture_date"] = pd.to_datetime(m0.get("capture_ts"), format="%Y%m%d%H%M%S", errors="coerce").dt.date.astype(str)
    m0["capture_before_market_date"] = m0.capture_ts.notna() & (m0.capture_date < m0.report_date)
    excluded = set(m0[m0.capture_before_market_date].file_id)
    for _, g in m0[~m0.capture_before_market_date].groupby("sha256"):
        excluded |= set(g.sort_values("file_id").file_id.iloc[1:])
    m0["excluded_file"] = m0.file_id.isin(excluded)
    metas = m0.to_dict("records")
    n_ex = int(df.file_id.isin(excluded).sum())
    df, bad = df[~df.file_id.isin(excluded)].copy(), bad[~bad.file_id.isin(excluded)].copy()
    # Re-captures of the same issue: identical quote key in a DIFFERENT file for the same market and date.
    key = ["market", "report_date", "commodity", "package_source", "origin_code", "variety_carried",
           "descriptor_source", "price_low", "price_high", "organic"]
    n0 = len(df)
    first_file = df.groupby(key, dropna=False)["file_id"].transform("first")
    df["is_duplicate_issue_quote"] = df["file_id"] != first_file
    m = pd.DataFrame(metas)
    log.add("parse_quotes", "segment commodity blocks (blank-line continuations kept); one row per price quote; "
            "package = phrase nearest the price; origin after it (else before); variety carried until package/origin changes; "
            "annotation-only prices attach to the previous quote", len(metas), n0, rows_quarantined=len(bad),
            notes="rows without package or origin are quarantined")
    log.add("exclude_duplicate_capture_files", "byte-identical Wayback files: keep earliest capture on/after market date; "
            "drop files captured before their own market date", len(m0), len(m0) - len(excluded),
            rows_flagged=n_ex, notes=f"excluded files: {sorted(excluded)}")
    log.add("flag_duplicate_quotes", "same market/date/quote key appearing in a different file (re-captured issue); kept, flagged",
            n0, n0, rows_flagged=int(df.is_duplicate_issue_quote.sum()))
    log.add("price_basis", "per lb / each / per bin / per bunch detected from text; per-lb value derived only for per-package "
            "prices with exactly one stated outer weight", n0, n0,
            rows_flagged=int((df.price_basis != "USD per package").sum()))
    CLEAN.mkdir(parents=True, exist_ok=True)
    QUAR.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN / "terminal_quotes.parquet", index=False)
    bad.to_parquet(QUAR / "terminal_unparsed.parquet", index=False)
    m.to_csv(REPORT / "report_files.csv", index=False)
    log.write(REPORT / "transform_log.json")
    return profile(df, bad, metas)


def profile(df: pd.DataFrame, bad: pd.DataFrame, metas: list[dict]) -> dict:
    con = duckdb.connect()
    con.register("q", df[~df.is_duplicate_issue_quote])
    mall = pd.DataFrame(metas)
    m = mall[~mall.excluded_file] if "excluded_file" in mall else mall
    prof = {
        "grain": "price quote (market x report date x commodity x package x origin x variety/descriptor)",
        "files_parsed": len(metas), "files_excluded_duplicate_or_misdated": int(len(mall) - len(m)),
        "files_without_header_date": int(m.report_date.isna().sum()),
        "quotes": int(len(df)), "quotes_unique": int((~df.is_duplicate_issue_quote).sum()), "unparsed_segments": int(len(bad)),
        "parse_yield": round(len(df) / max(1, len(df) + len(bad)), 3),
        "price_token_coverage": {"tokens_in_blocks": int(m.n_price_tokens.sum()),
                                 "quotes_parsed_or_quarantined": int(m.n_quotes.sum()),
                                 "mostly_attached": int(m.n_mostly_attached.sum()), "mostly_orphan": int(m.n_mostly_orphan.sum()),
                                 "annotation_prices_attached": int(m.n_annotation_prices.sum()),
                                 "unaccounted": int(m.n_price_tokens.sum() - m.n_quotes.sum() - m.n_mostly_attached.sum()
                                                    - m.n_mostly_orphan.sum() - m.n_annotation_prices.sum())},
        "legacy_report_dates": sorted(m[m.file_id.str.startswith(LEGACY_SID)].report_date.dropna().unique().tolist()),
        "wayback_report_dates_n": int(m[m.file_id.str.startswith(WAYBACK_SID)].report_date.nunique()),
        "wayback_report_date_range": [m[m.file_id.str.startswith(WAYBACK_SID)].report_date.min(), m[m.file_id.str.startswith(WAYBACK_SID)].report_date.max()],
        "markets": sorted(df.market.dropna().unique().tolist()),
        "price_basis_counts": df.price_basis.value_counts().to_dict(),
        "share_quotes_with_weight_derived": round(float(df.package_weight_lb_derived.notna().mean()), 3),
        "share_quotes_with_range": round(float((df.price_high > df.price_low).mean()), 3),
        "share_quotes_with_mostly": round(float(df.mostly_low.notna().mean()), 3),
        "top_commodities": con.sql("select commodity, count(*) n from q group by 1 order by 2 desc limit 15").fetchall(),
        "eggplant_packages": con.sql("""select package_source, count(*) n, min(price_low), max(price_high)
                                         from q where commodity='EGGPLANT' and price_basis='USD per package'
                                         group by 1 order by 2 desc limit 10""").fetchall(),
        "precision_audit": "independent hand audits: v1 42.5% strict (40 quotes, data_review.md); v2 74.0% strict "
                           "(50 quotes, Wilson 95% CI 60-84%, usda_parser_reaudit.md; price/mostly/basis/$/lb 50/50, package 45/50). "
                           "This v3 revision targets the v2 error classes and has not been re-audited.",
        "time_semantics": "report_date = market date in header ('as of'); publication time not in file; "
                          "Wayback capture time is an upper bound on public availability",
    }
    write_json(prof, REPORT / "profile.json")
    return prof


def calibrate() -> dict:
    """Eggplant 1 1/9 bushel carton offering prices: levels by month and quality spread.

    Sparse dated snapshots (one market for most dates), not a daily series; they
    bound price levels and seasonal ranges. They say nothing about the timing or
    cause of any particular price move.
    """
    q = pd.read_parquet(CLEAN / "terminal_quotes.parquet")
    e = q[(q.commodity == "EGGPLANT") & (q.price_basis == "USD per package") & ~q.is_duplicate_issue_quote
          & q.package_source.str.contains(r"1 1/9 (?:bushel|bu)\b|1 1/9 bu|1 1/9 buctns", regex=True, na=False)
          & ~q.package_source.str.contains(r"1 bu,", regex=False, na=False)].copy()
    e["mid"] = (e.price_low + e.price_high) / 2
    e["month"] = pd.to_datetime(e.report_date).dt.month
    text = (e.size_grade_source.fillna("") + " " + e.descriptor_source.fillna("")).str.lower()
    e["lower_quality"] = text.str.contains(r"fr qual|fair|poor|ord qual|ord cond|fr cond|poor cond|fr appear")
    by_date = e[~e.lower_quality].groupby("report_date").mid.median()
    months = pd.to_datetime(by_date.index).month
    out = {
        "package": "1 1/9 bushel carton (all wrappings and carton/crate variants pooled; per-package USD)",
        "n_quotes": int(len(e)), "n_report_dates": int(e.report_date.nunique()),
        "markets": sorted(e.market.unique().tolist()),
        "report_date_median_mid_usd_q10_50_90": [round(float(x), 2) for x in by_date.quantile([0.1, 0.5, 0.9])],
        "by_month_median_of_date_medians": {int(m): round(float(by_date[months == m].median()), 2) for m in sorted(set(months))},
        "by_month_n_dates": {int(m): int((months == m).sum()) for m in sorted(set(months))},
        "jan_2024_cross_section_mid_usd_q10_50_90": [round(float(x), 2) for x in
                                                     e[(e.report_date >= "2024-01-01") & ~e.lower_quality].mid.quantile([0.1, 0.5, 0.9])],
        "lower_quality_to_standard_ratio_same_date_median": None,
        "provenance": {"status": "derived", "source_ids": [LEGACY_SID, WAYBACK_SID],
                       "caveats": ["offering quotes, not transactions or volumes",
                                   "dated snapshots, mostly Boston; not a daily series; capture dates are irregular",
                                   "quality class inferred from descriptor text ('fr qual', 'poor cond', ...)",
                                   "parser precision: independent hand audits, see docs/findings.md"]},
    }
    ratios = []
    for d, g in e.groupby("report_date"):
        lo, hi = g[g.lower_quality].mid, g[~g.lower_quality].mid
        if len(lo) and len(hi):
            ratios.append(float(lo.median() / hi.median()))
    if ratios:
        out["lower_quality_to_standard_ratio_same_date_median"] = round(float(pd.Series(ratios).median()), 3)
        out["lower_quality_ratio_n_dates"] = len(ratios)
    write_json(out, ROOT / "data" / "calibration" / "usda_eggplant.json")
    return out


def main(argv: list[str]) -> int:
    if "acquire" in argv:
        res = acquire_legacy()
        print(sum("error" not in r for r in res), "legacy files ok;", [r for r in res if "error" in r])
    if "wayback" in argv:
        res = acquire_wayback()
        print(sum("error" not in r for r in res), "wayback captures ok;", [r for r in res if "error" in r][:5])
    if "parse" in argv:
        print(parse_all())
    if "calibrate" in argv:
        print(calibrate())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
