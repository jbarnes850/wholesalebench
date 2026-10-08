"""UCI Online Retail II: stage, profile, clean.

Source: UK-based non-store online retailer of giftware; many customers are
wholesalers (per the UCI page). Two sheets, 2009-12-01..2010-12-09 and
2010-12-01..2011-12-09, which overlap for 1-9 Dec 2010.

Staging is lossless: every cell is kept as text exactly as read (plus a type
tag for the mixed-type key columns) with (sheet, excel_row) lineage.
Cleaning never drops a row. It classifies each line and quarantines rows whose
meaning cannot be resolved from the data.
"""

from __future__ import annotations

import re
import sys
import zipfile

import duckdb
import openpyxl
import pandas as pd

from wsb.lineage import ROOT, TransformLog, write_json

SID = "uci_online_retail_ii"
XLSX = ROOT / "data" / "staged" / SID / "_unzip" / "online_retail_II.xlsx"
STAGED = ROOT / "data" / "staged" / SID / "lines.parquet"
CLEAN = ROOT / "data" / "clean" / SID / "lines.parquet"
QUAR = ROOT / "data" / "quarantine" / SID / "lines.parquet"
REPORT = ROOT / "reports" / SID

PRODUCT_CODE = re.compile(r"^\d{5}[A-Za-z]{0,2}$")
# Split before any distribution is fitted. Calibration ends 2011-06-30.
CALIBRATION_END = pd.Timestamp("2011-07-01")


def extract_workbook() -> None:
    """Extract the single workbook member from the raw zip (no other member is touched)."""
    if XLSX.exists():
        return
    XLSX.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ROOT / "data" / "raw" / SID / "online_retail_ii.zip") as zf:
        XLSX.write_bytes(zf.read(XLSX.name))


def stage() -> pd.DataFrame:
    extract_workbook()
    wb = openpyxl.load_workbook(XLSX, read_only=True)
    frames = []
    for ws in wb.worksheets:
        rows = ws.iter_rows(values_only=True)
        header = next(rows)
        recs = []
        for i, r in enumerate(rows, start=2):
            inv, code, desc, qty, ts, price, cust, country = r
            recs.append(
                (
                    ws.title,
                    i,
                    None if inv is None else str(inv),
                    type(inv).__name__,
                    None if code is None else str(code),
                    type(code).__name__,
                    None if desc is None else str(desc),
                    type(desc).__name__,
                    qty,
                    ts,
                    price,
                    None if cust is None else str(int(cust)) if isinstance(cust, float) else str(cust),
                    country,
                )
            )
        df = pd.DataFrame(
            recs,
            columns=[
                "sheet", "excel_row", "invoice", "invoice_type_tag", "stock_code",
                "stock_code_type_tag", "description", "description_type_tag", "quantity", "invoice_ts",
                "price", "customer_id", "country",
            ],
        )
        frames.append(df)
        assert list(header) == ["Invoice", "StockCode", "Description", "Quantity",
                                "InvoiceDate", "Price", "Customer ID", "Country"], header
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "row_id", [f"{s[5:9]}:{r}" for s, r in zip(out.sheet, out.excel_row)])
    STAGED.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(STAGED, index=False)
    return out


CHARGE_CODES = {
    "POST": "postage", "DOT": "postage", "C2": "carriage", "C3": "carriage",
    "BANK CHARGES": "fee", "AMAZONFEE": "fee", "CRUK": "commission",
    "D": "discount", "S": "samples", "PADS": "manual_misc",
}
QUARANTINE_CODES = {"TEST001", "TEST002", "M", "m", "ADJUST", "ADJUST2", "GIFT"}


def classify(con: duckdb.DuckDBPyConnection) -> None:
    """Create table `c` with one row per staged line and its classification."""
    charge_case = " ".join(f"when stock_code = '{k}' then 'non_product_{v}'" for k, v in CHARGE_CODES.items())
    quar = ", ".join(f"'{k}'" for k in QUARANTINE_CODES)
    K = "invoice, stock_code, coalesce(description,''), quantity, invoice_ts, price, coalesce(customer_id,''), country"
    con.sql(f"""
    create or replace table c as
    with base as (
      select *,
        (sheet = 'Year 2010-2011' and invoice_ts < '2010-12-10') as is_sheet_overlap_dup,
        regexp_matches(stock_code, '^\\d{{5}}[A-Za-z]{{0,2}}$') as is_standard_product_code,
        quantity * price as line_amount
      from '{STAGED}'
    ), keyed as (
      select *, count(*) over (partition by is_sheet_overlap_dup, {K}) as exact_repeat_n,
             row_number() over (partition by is_sheet_overlap_dup, {K} order by excel_row) as exact_repeat_rank
      from base
    )
    select *,
      case
        when is_sheet_overlap_dup then 'source_overlap_duplicate'
        when invoice like 'A%' then 'bad_debt_adjustment'
        when stock_code in ({quar}) then 'quarantine_ambiguous_code'
        when invoice like 'C%' and quantity >= 0 then 'quarantine_cancel_nonnegative'
        when stock_code like 'gift_0001%' then 'gift_voucher'
        {charge_case}
        when invoice like 'C%' then 'cancellation'
        when quantity < 0 and price = 0 and customer_id is null then 'inventory_writeoff'
        when quantity > 0 and price = 0 and customer_id is null then 'quarantine_unpriced_inflow'
        when quantity > 0 and price = 0 then 'sale_zero_price'
        when quantity > 0 and price > 0 then 'sale'
        else 'quarantine_other'
      end as line_class,
      exact_repeat_n > 1 as is_exact_repeat,
      customer_id is null as is_customer_unknown,
      invoice_ts < TIMESTAMP '{CALIBRATION_END}' as is_calibration_period
    from keyed
    """)


def link_cancellations(con: duckdb.DuckDBPyConnection) -> None:
    """Link cancellations to earlier sales one at a time, in time order, with capacity.

    Candidate: same customer, stock_code and unit price, earlier timestamp.
    A cancellation links only when exactly ONE candidate still has enough
    un-cancelled quantity; the link then consumes that quantity, so a sale line
    can never be reversed for more than was sold. Statuses:
      linked_unique          exactly one candidate with remaining capacity
      ambiguous              several candidates with remaining capacity
      insufficient_remaining candidates exist but none has enough remaining quantity
      no_candidate           no earlier sale with the same customer, code and price
      unlinkable_no_customer cancellation has no customer id
    """
    pairs = con.sql("""
      with canc as (select row_id, customer_id, stock_code, price, invoice_ts, -quantity as q
                    from c where line_class = 'cancellation' and customer_id is not null),
           sale as (select row_id, customer_id, stock_code, price, invoice_ts, quantity
                    from c where line_class in ('sale','sale_zero_price'))
      select canc.row_id as cancel_row_id, canc.invoice_ts as cancel_ts, canc.q,
             sale.row_id as sale_row_id, sale.quantity as sale_qty,
             date_diff('minute', sale.invoice_ts, canc.invoice_ts) / 1440.0 as lag_days
      from canc join sale using (customer_id, stock_code, price)
      where sale.invoice_ts < canc.invoice_ts""").df()
    remaining = dict(zip(pairs.sale_row_id, pairs.sale_qty))
    out = []
    for (cid, _ts, q), g in pairs.sort_values(["cancel_ts", "cancel_row_id"]).groupby(
            ["cancel_row_id", "cancel_ts", "q"], sort=False):
        ok = g[[remaining[r] >= q for r in g.sale_row_id]]
        if len(ok) == 1:
            sid = ok.sale_row_id.iloc[0]
            remaining[sid] -= q
            out.append((cid, len(g), len(ok), sid, float(ok.lag_days.iloc[0]), "linked_unique"))
        else:
            out.append((cid, len(g), len(ok), None, None, "ambiguous" if len(ok) > 1 else "insufficient_remaining"))
    links = pd.DataFrame(out, columns=["cancel_row_id", "n_candidates", "n_with_capacity", "linked_sale_row_id",
                                       "lag_days", "status"])
    con.register("links_df", links)
    con.sql("""
    create or replace table c2 as
    select c.*, l.n_candidates, l.n_with_capacity, l.linked_sale_row_id, l.lag_days as cancel_lag_days,
      case when line_class <> 'cancellation' then null
           when customer_id is null then 'unlinkable_no_customer'
           when l.status is null then 'no_candidate'
           else l.status end as cancel_link_status
    from c left join links_df l on c.row_id = l.cancel_row_id
    """)


def clean() -> dict:
    con = duckdb.connect()
    log = TransformLog(SID)
    n0 = con.sql(f"select count(*) from '{STAGED}'").fetchone()[0]
    classify(con)
    link_cancellations(con)
    counts = dict(con.sql("select line_class, count(*) from c2 group by 1 order by 2 desc").fetchall())
    n_dup = counts.get("source_overlap_duplicate", 0)
    log.add("flag_sheet_overlap", "sheet 'Year 2010-2011' rows dated before 2010-12-10 duplicate sheet 'Year 2009-2010' "
            "(line multisets verified identical); flagged and excluded from canonical view, not deleted",
            n0, n0 - n_dup, rows_flagged=n_dup)
    n_rep = con.sql("select count(*) from c2 where is_exact_repeat and not is_sheet_overlap_dup and exact_repeat_rank > 1").fetchone()[0]
    log.add("flag_exact_repeats", "identical (invoice, code, desc, qty, ts, price, customer, country) lines within the canonical view; "
            "kept (could be genuine repeated basket lines), flagged for sensitivity analysis", n0 - n_dup, n0 - n_dup, rows_flagged=n_rep)
    quar = sum(v for k, v in counts.items() if k.startswith("quarantine"))
    log.add("classify_lines", "line_class rules in classify(); no rows dropped", n0 - n_dup, n0 - n_dup - quar,
            rows_quarantined=quar, notes=str(counts))
    links = dict(con.sql("select cancel_link_status, count(*) from c2 where line_class='cancellation' group by 1").fetchall())
    log.add("link_cancellations", "chronological, capacity-aware: unique earlier sale (same customer, code, price) "
            "with enough un-cancelled quantity; a link consumes that quantity", 
            counts.get("cancellation", 0), counts.get("cancellation", 0), rows_flagged=links.get("linked_unique", 0),
            notes=str(links))
    for p in (CLEAN, QUAR):
        p.parent.mkdir(parents=True, exist_ok=True)
    con.sql(f"copy (select * from c2 where line_class not like 'quarantine%' order by sheet, excel_row) to '{CLEAN}' (format parquet)")
    con.sql(f"copy (select * from c2 where line_class like 'quarantine%' order by sheet, excel_row) to '{QUAR}' (format parquet)")
    log.write(REPORT / "transform_log.json")
    return {"counts": counts, "cancel_links": links}


def profile() -> dict:
    """Profile the cleaned table (canonical view excludes the sheet-overlap duplicates)."""
    con = duckdb.connect()
    con.sql(f"create view v as select * from '{CLEAN}' where line_class <> 'source_overlap_duplicate'")
    one = lambda q: con.sql(q).fetchone()
    prof = {
        "grain": "invoice line",
        "staged_rows": one(f"select count(*) from '{STAGED}'")[0],
        "canonical_rows": one("select count(*) from v")[0],
        "quarantined_rows": one(f"select count(*) from '{QUAR}'")[0],
        "date_range": [str(x) for x in one("select min(invoice_ts), max(invoice_ts) from v")],
        "invoices": one("select count(distinct invoice) from v")[0],
        "customers_known": one("select count(distinct customer_id) from v")[0],
        "countries": one("select count(distinct country) from v")[0],
        "product_codes": one("select count(distinct stock_code) from v where line_class like 'sale%'")[0],
        "missing": dict(zip(["description", "customer_id"], one("select sum(description is null), sum(customer_id is null) from v"))),
        "invoices_with_multiple_timestamps": one("select count(*) from (select invoice from v group by 1 having count(distinct invoice_ts) > 1)")[0],
        "invoices_with_multiple_customers": one("select count(*) from (select invoice from v group by 1 having count(distinct coalesce(customer_id,'NA')) > 1)")[0],
        "line_amount_identity": "line_amount = quantity * price (computed; source has no stored totals to reconcile against)",
        "currency": "GBP (per UCI documentation; not stated in file)",
        "time_semantics": "InvoiceDate = invoice creation timestamp, local UK time assumed; no shipment/payment timestamps",
    }
    by_period = con.sql("""
      select is_calibration_period,
        count(*) filter (where line_class='sale') sales,
        count(*) filter (where line_class='cancellation') cancels,
        count(*) filter (where line_class='inventory_writeoff') writeoffs,
        count(*) filter (where line_class='bad_debt_adjustment') bad_debt,
        count(*) filter (where is_exact_repeat and exact_repeat_rank>1) repeats
      from v group by 1 order by 1 desc""").fetchall()
    prof["rare_events_by_period"] = [dict(zip(["calibration", "sales", "cancellations", "writeoffs", "bad_debt_adjustments", "exact_repeat_extras"], r)) for r in by_period]
    return prof


def calibrate() -> dict:
    """Structural statistics from the calibration period only (before 2011-07-01).

    These describe giftware B2B/B2C ordering structure. They are structural
    analogs for a produce world, not produce calibration.
    """
    con = duckdb.connect()
    con.sql(f"create view v as select * from '{CLEAN}' where line_class <> 'source_overlap_duplicate'")
    out = {}
    matched_start = (CALIBRATION_END - pd.Timedelta(days=161)).date()
    periods = (("calibration", "is_calibration_period"), ("validation", "not is_calibration_period"),
               ("calibration_matched_161d", f"is_calibration_period and invoice_ts >= DATE '{matched_start}'"))
    for period, cond in periods:
        con.sql(f"create or replace view p as select * from v where {cond}")
        con.sql("""create or replace table inv as select invoice, customer_id, min(invoice_ts) ts, count(*) n_lines
                   from p where line_class='sale' and customer_id is not null group by 1,2""")
        q = lambda x: [round(float(v), 3) for v in con.sql(x).fetchone()]
        con.sql("""create or replace table gaps as select date_diff('hour', lag(ts) over (partition by customer_id order by ts), ts)/24.0 d
                          from inv""")
        out[period] = {
            "lines_per_invoice_q10_50_90": [round(x, 1) for x in con.sql("select quantile_cont(n_lines, [0.1,0.5,0.9]) from inv").fetchone()[0]],
            "customer_reorder_gap_days_q10_50_90": [round(x, 1) for x in con.sql("select quantile_cont(d, [0.1,0.5,0.9]) from gaps where d > 0").fetchone()[0]],
            "share_customers_single_invoice": q("select avg((n=1)::int) from (select customer_id, count(*) n from inv group by 1)")[0],
            "cancel_line_rate": q("select count(*) filter (where line_class='cancellation') / count(*) filter (where line_class='sale') from p")[0],
            "cancel_lag_days_linked_q10_50_90": [round(x, 2) for x in con.sql("select quantile_cont(cancel_lag_days, [0.1,0.5,0.9]) from p where cancel_link_status='linked_unique'").fetchone()[0] or [None]*3],
            "qty_share_multiple_of_6": q("select avg((quantity % 6 = 0)::int) from p where line_class='sale' and is_standard_product_code")[0],
            "qty_share_multiple_of_12": q("select avg((quantity % 12 = 0)::int) from p where line_class='sale' and is_standard_product_code")[0],
            "share_codes_with_multiple_unit_prices": q("select avg((k>1)::int) from (select stock_code, count(distinct price) k from p where line_class='sale' group by 1)")[0],
            "writeoff_lines_per_1000_sales": q("select 1000.0*count(*) filter (where line_class='inventory_writeoff') / count(*) filter (where line_class='sale') from p")[0],
        }
    out["provenance"] = {"source_id": SID, "status": "derived", "domain": "giftware, mixed retail/wholesale",
                         "use": "structural analog for customer ordering cadence, basket size, pack multiples and reversal rates; "
                                "NOT produce calibration", "split": f"calibration < {CALIBRATION_END.date()} <= validation",
                         "censoring": "reorder gaps and single-invoice shares depend on window length (calibration 576 days, "
                                      "validation 161 days); gaps crossing the split are dropped from both periods. "
                                      "Compare periods only on matched-length windows (see calibration_matched_161d).",
                         "cancel_lag_note": "lag between a sale and a uniquely linked cancellation (credit), not a physical return interval"}
    return out


def main() -> int:
    if not STAGED.exists():
        print("staged", len(stage()))
    res = clean()
    print(res)
    prof = profile()
    write_json(prof, REPORT / "profile.json")
    cal = calibrate()
    write_json(cal, ROOT / "data" / "calibration" / "uci_structure.json")
    print(prof)
    print(cal)
    return 0


if __name__ == "__main__":
    sys.exit(main())
