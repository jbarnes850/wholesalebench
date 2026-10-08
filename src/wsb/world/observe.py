"""Agent-facing tools, fork arms and notes (v0.3).

Information rule: every tool returns only records whose visible_day <= today.
Known future obligations are visible because they are on today's books:
receivables and payables due in the next 7 days (as totals), promised
delivery days of open purchase orders, the credit line. Realizations not yet
revealed are not: future prices, future orders, future V2 offers, lot decay
scales, the V1 latent, customer goodwill and its sensitivity, the market
regime, and every evaluator-only field (decision ids, causal annotations,
reference actions, random-stream keys).

Books are shown as an operator would see them on a balance-sheet screen: open
totals and the amount due in the coming week, not itemized past invoices and
bills. Itemized or per-day schedules would let the fresh arm rebuild past
shipments and purchases, which belong to the ledger tool.

Arms differ ONLY in what the ledger tool returns and whether a note is shown.
The world state, the action interface and the other tools are identical.
Agents must receive only the JSON these tools return. The Obs object itself
holds the full world state and must never be handed to an in-process agent.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field

from wsb.world.engine import World, WorldState

FORBIDDEN_KEYS = {"decision_id", "contributing_decisions", "ahead_decisions", "cause", "regime", "state_path",
                  "lot_scale", "v1_decay_scale", "goodwill", "goodwill_drop", "order_noise", "log_price",
                  "rng_key", "hidden", "reference", "exo", "attempt"}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
ID_FIELDS = ("lot_id", "po_id", "bill_id", "invoice_id")


# Keyed with a deployment secret so a code-executing agent cannot invert aliases by brute force.
# Set WSB_ALIAS_PEPPER in any deployment; the default exists only for local tests.
_PEPPER = os.environ.get("WSB_ALIAS_PEPPER", "local-test-pepper-not-secret").encode()


def alias(seed: int, internal_id: str) -> str:
    """Opaque, arm-invariant display id. Sequential internal ids would reveal how many
    purchases and events preceded the fork, which the fresh arm must not learn."""
    h = hmac.new(_PEPPER, f"{seed}:{internal_id}".encode(), hashlib.blake2b).hexdigest()[:8]
    return f"{internal_id[:1]}-{h}"


def _alias_record(seed: int, rec: dict) -> dict:
    out = dict(rec)
    for k in ID_FIELDS:
        if isinstance(out.get(k), str):
            out[k] = alias(seed, out[k])
    if "lots" in out:
        out["lots"] = [[alias(seed, lid), q] for lid, q in out["lots"]]
    return out


def _due_within(items, today: int, horizon: int = 7) -> int:
    """Total due in the next `horizon` days. A per-day schedule would reveal past daily shipment totals."""
    return sum(a.amount_c for a in items if a.due_day <= today + horizon)


@dataclass(frozen=True)
class ArmConfig:
    name: str
    history: bool = True
    note: str | None = None
    watermark_event_id: int = 0  # fresh arm: the ledger shows only events after this id


@dataclass
class Obs:
    world: World
    s: WorldState
    arm: ArmConfig
    calls: list[str] = field(default_factory=list)

    @property
    def day(self) -> int:
        return self.s.day

    def state(self) -> dict:
        self.calls.append("state")
        w, s, p = self.world, self.s, self.world.p
        seed = s.exo.seed
        nxt = p.next_v1_delivery(s.day)
        open_ = s.day < p.episode_days
        return {
            "day": s.day, "weekday": WEEKDAYS[s.day % 7], "purchasing_open": open_, "episode_days": p.episode_days,
            "cash_c": s.cash_c, "credit_limit_c": p.credit_limit_c,
            "receivables": {"open_total_c": sum(a.amount_c for a in s.ar), "due_next_7_days_c": _due_within(s.ar, s.day)},
            "payables": {"open_total_c": sum(a.amount_c for a in s.ap), "due_next_7_days_c": _due_within(s.ap, s.day)},
            "inventory_lots": [{"lot_id": alias(seed, l.lot_id), "vendor": l.vendor, "received_day": l.received_day,
                                "days_in_stock": s.day - l.received_day, "age_at_receipt": l.receipt_age,
                                "cases": l.qty, "unit_cost_c": l.unit_cost_c}
                               for l in sorted(s.lots, key=lambda l: (l.received_day, l.lot_id))],
            "open_purchase_orders": [{"po_id": alias(seed, x.po_id), "vendor": x.vendor, "cases": x.cases,
                                      "unit_cost_c": x.unit_cost_c, "arrival_day": x.arrival_day} for x in s.pos],
            "customer_orders_today": dict(zip(p.customer_ids, s.orders_today)),
            "customer_profiles": {c: {"kind": p.customer_kind[i], "average_daily_cases": p.customer_base_cases[i],
                                      "weekday_pattern_mon_to_sun": list(p.customer_weekday[i]),
                                      "terms_days": p.customer_terms_days}
                                  for i, c in enumerate(p.customer_ids)},
            "price_list_c": s.price_list_c,
            "vendor_offers": [
                {"vendor": "V1", "kind": p.vendor_kind[0], "unit_cost_c": w.offer_c(s, "V1"),
                 "pallet_cases": p.v1_pallet_cases, "next_delivery_day": nxt,
                 "pallets_already_ordered_for_that_day": w.v1_committed(s, nxt) // p.v1_pallet_cases,
                 "max_pallets_per_delivery": p.v1_max_pallets, "delivery_weekdays": [WEEKDAYS[x] for x in p.v1_delivery_weekdays],
                 "age_at_receipt_days": p.v1_receipt_age, "terms_days": p.vendor_terms_days[0]},
                {"vendor": "V2", "kind": p.vendor_kind[1], "unit_cost_c": w.offer_c(s, "V2"),
                 "available_cases_today": s.exo.v2_avail[s.day] - s.v2_bought_today, "delivery": "same day",
                 "age_at_receipt_days": s.exo.v2_age[s.day], "terms_days": p.vendor_terms_days[1]},
            ] if open_ else [],
        }

    def market(self, n_days: int = 28) -> list[list[int]]:
        """Terminal-market reference price (cents per case) for the last n days, today included."""
        self.calls.append("market")
        lo = max(-self.world.p.pre_episode_market_days, self.s.day - n_days + 1)
        return [[d, self.s.exo.market_c[d]] for d in range(lo, self.s.day + 1)]

    def ledger(self) -> list[dict]:
        """The desk's own records: decisions and their consequences, public fields only."""
        self.calls.append("ledger")
        evs = self.s.events if self.arm.history else [e for e in self.s.events if e.event_id > self.arm.watermark_event_id]
        seed = self.s.exo.seed
        visible = [e for e in evs if e.visible_day <= self.s.day]
        # row numbers restart at 1 in every arm; internal event ids are never shown
        return [{"row": i + 1, "type": e.type, "day": e.day, **_alias_record(seed, e.public)} for i, e in enumerate(visible)]

    def note(self) -> str | None:
        self.calls.append("note")
        return self.arm.note

    def dump(self) -> dict:
        """Everything an agent could retrieve in this arm at this moment."""
        return {"state": self.state(), "market": self.market(10_000), "ledger": self.ledger(), "note": self.note()}


def leak_report(obs: Obs) -> dict:
    """Check one observation for forbidden keys, internal ids and records dated after today."""
    d = obs.dump()
    text = json.dumps(d)
    found_keys = sorted(k for k in FORBIDDEN_KEYS if f'"{k}"' in text)
    future_market = [x for x in d["market"] if x[0] > obs.day]
    future_ledger = [e for e in d["ledger"] if e["day"] > obs.day]
    internal_ids = [k for k in ('"L0', '"PO0', '"B0', '"I0', '"D0', '"OPEN-') if k in text]
    return {"forbidden_keys": found_keys, "future_market_rows": len(future_market),
            "future_ledger_rows": len(future_ledger), "internal_ids_exposed": internal_ids,
            "ok": not found_keys and not future_market and not future_ledger and not internal_ids}


# --------------------------------------------------------------------------- notes

def attribution_note(s: WorldState) -> str:
    """Name the earlier purchase whose lot produced the largest write-off cost so far.

    Built from evaluator provenance. States only past facts; recommends nothing.
    """
    by_decision: dict[str, dict] = {}
    for e in s.events:
        if e.type == "spoilage_writeoff":
            did = e.hidden["contributing_decisions"][0]
            if did == "OPENING":
                continue
            r = by_decision.setdefault(did, {"cases": 0, "cost_c": 0, "first": e.day, "last": e.day})
            r["cases"] += e.public["cases"]
            r["cost_c"] += e.public["cost_c"]
            r["last"] = e.day
    if not by_decision:
        return ""
    did, r = max(by_decision.items(), key=lambda kv: kv[1]["cost_c"])
    po = next(x for x in s.events if x.type == "purchase_order" and x.hidden.get("decision_id") == did)
    when = f"on day {r['first']}" if r["first"] == r["last"] else f"between day {r['first']} and day {r['last']}"
    return (f"Note: {r['cases']} cases from your day-{po.day} order of {po.public['cases']} cases from "
            f"{po.public['vendor']} at ${po.public['unit_cost_c'] / 100:,.2f} per case were written off {when} "
            f"(cost ${r['cost_c'] / 100:,.2f}).")


def placebo_note(s: WorldState, target_len: int) -> str:
    """A ledger-like note with real public numbers that names no decision, length-matched (within 10%)."""
    d = s.day
    window = [s.exo.market_c[x] for x in range(max(-28, d - 13), d + 1)]
    moves = sum(1 for a, b in zip(window, window[1:]) if a != b)
    text = (f"Note: the terminal reference moved on {moves} of the last {len(window) - 1} days and ranged from "
            f"${min(window) / 100:,.2f} to ${max(window) / 100:,.2f} per 1 1/9 bushel carton; amounts on this desk "
            f"are in US dollars, and records are listed in the order they were entered.")
    if len(text) <= target_len:
        return text
    out = ""
    for word in text.split():
        if len(out) + len(word) + 1 > target_len - 1:
            break
        out = f"{out} {word}".strip()
    return out.rstrip(",.;") + "."


def fork_arms(s: WorldState) -> list[ArmConfig]:
    """The four follow-up-decision arms, all built from the same checkpoint."""
    wm = s.events[-1].event_id if s.events else 0
    att = attribution_note(s)
    return [
        ArmConfig("history", history=True),
        ArmConfig("fresh", history=False, watermark_event_id=wm),
        ArmConfig("attribution_note", history=True, note=att or None),
        ArmConfig("placebo_note", history=True, note=placebo_note(s, len(att)) if att else None),
    ]
