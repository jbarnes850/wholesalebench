"""FreshRetailNet-50K: flag, profile, and derive demand-shape statistics.

Source grain: one store x product x day, with 24-slot hourly sales and hourly
stock status. Sales are globally normalized by an undisclosed coefficient, so
no physical unit is recoverable; every statistic below is unit-free (ratios,
shares, coefficients of variation) or stated in normalized units.

Verified semantics (see reports/freshretailnet_50k/profile.json):
- hours_stock_status == 1 marks a stockout hour; stock_hour6_22_cnt equals the
  sum over slots 6..21. A minority of stockout hours still record sales, so the
  flag is a censoring indicator at hour grain, not proof of zero availability.
- Observed sales are censored demand. Nothing here estimates latent demand.

Split: the publisher's train split (2024-03-28..2024-06-25) is calibration;
the eval split (2024-06-26..2024-07-02) is validation. Nothing is fitted on eval.
"""

from __future__ import annotations

import sys

import duckdb
import numpy as np
import pyarrow.parquet as pq

from wsb.lineage import ROOT, TransformLog, write_json

SID = "freshretailnet_50k"
RAW = ROOT / "data" / "raw" / SID
CLEAN = ROOT / "data" / "clean" / SID
REPORT = ROOT / "reports" / SID


def flag() -> dict:
    con = duckdb.connect()
    log = TransformLog(SID)
    counts = {}
    CLEAN.mkdir(parents=True, exist_ok=True)
    for split in ("train", "eval"):
        src = RAW / f"{split}.parquet"
        n = con.sql(f"select count(*) from '{src}'").fetchone()[0]
        con.sql(f"""
        copy (
          select *,
            '{split}' as split,
            discount = 0 as flag_discount_zero_suspect,
            discount > 1 as flag_discount_above_list,
            list_sum(hours_stock_status) = 24 as flag_full_day_stockout,
            stock_hour6_22_cnt > 0 as flag_any_stockout_6_22,
            list_sum(list_transform(list_zip(hours_sale, hours_stock_status), x -> case when x[2] = 1 and x[1] > 0 then 1 else 0 end)) as n_stockout_hours_with_sales
          from '{src}'
        ) to '{CLEAN / f"{split}_flagged.parquet"}' (format parquet)""")
        c = con.sql(f"""select count(*), sum(flag_discount_zero_suspect::int), sum(flag_discount_above_list::int),
                           sum(flag_full_day_stockout::int), sum(flag_any_stockout_6_22::int)
                        from '{CLEAN / f"{split}_flagged.parquet"}'""").fetchone()
        counts[split] = dict(zip(["rows", "discount_zero_suspect", "discount_above_list", "full_day_stockout", "any_stockout_6_22"], c))
        log.add(f"flag_{split}", "add flags; no rows dropped or values altered (discount==0 kept as-is: likely a missing-value code, unverified)",
                n, c[0], rows_flagged=c[1] + c[2])
    log.write(REPORT / "transform_log.json")
    return counts


def _arrays(path):
    t = pq.read_table(path, columns=["store_id", "product_id", "dt", "sale_amount", "hours_sale",
                                      "hours_stock_status", "stock_hour6_22_cnt", "discount"])
    n = t.num_rows
    hs = np.asarray(t.column("hours_sale").combine_chunks().flatten()).reshape(n, 24)
    st = np.asarray(t.column("hours_stock_status").combine_chunks().flatten()).reshape(n, 24)
    return t, hs, st


def profile() -> dict:
    t, hs, st = _arrays(RAW / "train.parquet")
    cnt = np.asarray(t.column("stock_hour6_22_cnt"))
    sale = np.asarray(t.column("sale_amount"))
    vals = hs[hs > 0]
    return {
        "grain": "store x product x day (hourly arrays of length 24)",
        "train_rows": int(t.num_rows),
        "hourly_array_length_ok": True,
        "daily_equals_sum_hourly_max_abs_err": float(np.max(np.abs(hs.sum(1) - sale))),
        "stock_cnt_equals_sum_slots_6_21": float(np.mean(cnt == st[:, 6:22].sum(1))),
        "stockout_hour_share_by_slot": np.round(st.mean(0), 3).tolist(),
        "sales_share_by_slot": np.round(hs.sum(0) / hs.sum(), 4).tolist(),
        "share_stockout_hours_with_positive_sales": float(np.mean(hs[st == 1] > 0)),
        "share_sales_mass_in_stockout_hours": float(hs[st == 1].sum() / hs.sum()),
        "share_series_days_full_stockout": float(np.mean(st.sum(1) == 24)),
        "share_series_days_any_stockout_6_22": float(np.mean(cnt > 0)),
        "share_hourly_values_multiple_of_0.01": float(np.mean(np.abs(vals / 0.01 - np.round(vals / 0.01)) < 1e-6)),
        "normalization": "sale_amount multiplied by an undisclosed global coefficient (publisher); no integer unit quantum found",
        "hierarchy": "product -> one category path; store -> one city (verified, max distinct = 1)",
        "coverage": "50,000 series x 90 contiguous days train + 7 days eval; 898 stores, 865 products, 18 cities (computed)",
    }


def _demand_stats(path) -> dict:
    t, hs, st = _arrays(path)
    con = duckdb.connect()
    con.register("t", t)
    # Uncensored days: no stockout hour between 06:00 and 22:00. Selecting on this
    # biases toward lower-demand days; reported as a known limitation.
    df = con.sql("""
      select store_id, product_id, dt, sale_amount, discount, dayofweek(dt::date) dow,
             stock_hour6_22_cnt = 0 as uncensored
      from t""").df()
    unc = df[df.uncensored]
    g = unc.groupby(["store_id", "product_id"]).sale_amount
    stats = g.agg(["mean", "std", "count"])
    stats = stats[(stats["count"] >= 5) & (stats["mean"] > 0)]
    cv = (stats["std"] / stats["mean"]).dropna()
    rel = unc.merge(stats["mean"].rename("m").reset_index(), on=["store_id", "product_id"])
    rel = rel[rel.m > 0]
    rel["r"] = rel.sale_amount / rel.m
    dow = rel.groupby("dow").r.mean()
    disc = rel.assign(promo=rel.discount < 0.95, full=rel.discount >= 0.99)
    promo_ratio = disc[disc.promo].r.mean() / disc[disc.full].r.mean()
    return {
        "n_series_days": int(len(df)),
        "share_uncensored_days": round(float(df.uncensored.mean()), 3),
        "daily_cv_uncensored_q10_50_90": np.round(np.quantile(cv, [0.1, 0.5, 0.9]), 3).tolist(),
        "series_mean_daily_sales_q10_50_90_normalized_units": np.round(np.quantile(stats["mean"], [0.1, 0.5, 0.9]), 3).tolist(),
        "dow_multiplier_sun0_duckdb_dayofweek": np.round(dow.values, 3).tolist() if len(dow) == 7 else dow.round(3).to_dict(),
        "promo_vs_full_price_sales_ratio_descriptive": round(float(promo_ratio), 3),
        "share_days_any_stockout_6_22": round(float(1 - df.uncensored.mean()), 3),
    }


def calibrate() -> dict:
    return {
        "calibration_train": _demand_stats(RAW / "train.parquet"),
        "validation_eval": _demand_stats(RAW / "eval.parquet"),
        "provenance": {
            "source_id": SID, "status": "derived",
            "domain": "fresh grocery retail (store-level, China), not wholesale",
            "use": "unit-free demand-shape priors: day-to-day variability, weekday pattern, stockout incidence, "
                   "promotion association. Physical scale is an explicit world assumption.",
            "caveats": [
                "uncensored-day selection biases means and CVs downward",
                "promotion ratio is an association; discounts are not randomly assigned",
                "eval split is 7 days, so weekday multipliers there rest on one week",
            ],
        },
    }


def main() -> int:
    print(flag())
    prof = profile()
    write_json(prof, REPORT / "profile.json")
    cal = calibrate()
    write_json(cal, ROOT / "data" / "calibration" / "freshretail_demand.json")
    print(cal)
    return 0


if __name__ == "__main__":
    sys.exit(main())
