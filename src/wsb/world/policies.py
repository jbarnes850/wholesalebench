"""Deterministic baseline policies (v0.3). They see only the agent tools (Obs).

All non-trivial policies share one planner and differ only in the beliefs fed to it:
- do_nothing: never buys (a valid policy; also the floor).
- rule_base_stock: history-BLIND. Plans with prior beliefs about the vendor and
  customers (it still filters the public market). It reads only state() and
  market(), never ledger() or note(), so it acts identically in every fork arm.
  Default allocation (fixed priority).
- adaptive: history-AWARE. Plans with posterior beliefs from its own ledger
  (wsb.world.beliefs, the same likelihood the reference uses). When stock cannot
  cover today's orders, it protects the customers it believes are most
  goodwill-sensitive.
- NoisyRule / ScriptedOverbuy: labelled history generators for fork construction.
  Forks built from their histories test use of records, not learning from one's
  own actions.

Planner (a declared heuristic, not an optimum):
  * today: buy same-day V2 for any shortfall against today's orders (availability
    and credit permitting);
  * on the order-cutoff day for each V1 delivery, choose 0..max pallets by a fluid
    expected-value simulation over that delivery's coverage window: expected demand
    with goodwill, belief-weighted decay by age, FIFO, the regime-aware expected
    price path (so it can stockpile when recovery is expected), V2 cover for
    shortfalls at its expected margin, holding and disposal costs, and leftover stock
    valued at replacement cost times expected survival.
"""

from __future__ import annotations

from dataclasses import dataclass

from wsb.world.beliefs import Beliefs, beliefs_from_dump, prior_beliefs
from wsb.world.engine import Allocate, Buy
from wsb.world.observe import Obs
from wsb.world.params import WorldParams
from wsb.world.rng import stream


def do_nothing(obs: Obs) -> list:
    return []


def expected_demand(p: WorldParams, b: Beliefs, day: int, today: int) -> float:
    return sum(p.customer_base_cases[i] * p.customer_weekday[i][day % 7] * b.goodwill_multiplier(p, i, day - today)
               for i in range(len(p.customer_ids)))


def _fluid_value(p: WorldParams, b: Beliefs, st: dict, pallets: int, path: list[float]) -> float:
    d = st["day"]
    nxt = p.next_v1_delivery(d)
    end = min(p.next_v1_delivery(nxt) - 1, st["episode_days"] - 1)
    offers = {o["vendor"]: o for o in st["vendor_offers"]}
    v1_cost = offers["V1"]["unit_cost_c"]
    v2_avail_mean = sum(p.v2_avail_glut if b.p_glut() > 0.5 else p.v2_avail_normal) / 2
    need = sum(st["customer_orders_today"].values())
    cohorts = []
    for l in st["inventory_lots"]:  # after today's shipments (orders known; FIFO)
        take = min(l["cases"], need)
        need -= take
        if l["cases"] - take:
            cohorts.append([l["vendor"], d - l["received_day"] + l["age_at_receipt"], float(l["cases"] - take)])
    arrivals: dict[int, list] = {}
    for x in st["open_purchase_orders"]:
        if x["arrival_day"] <= end:
            arrivals.setdefault(x["arrival_day"], []).append([x["vendor"], p.v1_receipt_age, float(x["cases"])])
    if pallets:
        arrivals.setdefault(nxt, []).append(["V1", p.v1_receipt_age, float(pallets * p.v1_pallet_cases)])
    value = -pallets * p.v1_pallet_cases * v1_cost
    for t in range(d + 1, end + 1):
        mkt = path[t - d - 1]
        price, v2_cost = p.markup * mkt, p.vendor_spread[1] * mkt
        for c in cohorts:  # overnight aging, belief-weighted
            surv = b.survival_ratio(p, c[0], c[1], c[1] + 1)
            value -= c[2] * (1 - surv) * p.disposal_c_per_case
            c[2] *= surv
            c[1] += 1
        cohorts.extend([list(a) for a in arrivals.get(t, [])])
        dem = expected_demand(p, b, t, d)
        for c in cohorts:
            take = min(c[2], dem)
            c[2] -= take
            dem -= take
            value += take * price
        value += min(dem, v2_avail_mean) * max(0.0, price - v2_cost)
        value -= sum(c[2] for c in cohorts) * p.holding_c_per_case_day
        cohorts = [c for c in cohorts if c[2] > 1e-9]
    replacement = p.vendor_spread[0] * path[min(len(path), end - d + 3) - 1]
    for c in cohorts:
        value += c[2] * replacement * b.survival_ratio(p, c[0], c[1], c[1] + 3)
    return value


def plan(p: WorldParams, st: dict, b: Beliefs, allocate: bool = False) -> list:
    if not st["purchasing_open"]:
        return []
    offers = {o["vendor"]: o for o in st["vendor_offers"]}
    acts: list = []
    orders = st["customer_orders_today"]
    on_hand = sum(l["cases"] for l in st["inventory_lots"])
    short = sum(orders.values()) - on_hand
    v2 = offers["V2"]
    room_cash = max(0, (st["cash_c"] + st["credit_limit_c"]) // max(1, v2["unit_cost_c"]))
    q2 = min(max(short, 0), v2["available_cases_today"], room_cash)
    if q2 > 0:
        acts.append(Buy("V2", int(q2)))
    v1 = offers["V1"]
    if v1["next_delivery_day"] == st["day"] + 1 and v1["next_delivery_day"] < st["episode_days"]:
        room = v1["max_pallets_per_delivery"] - v1["pallets_already_ordered_for_that_day"]
        path = b.expected_market_path(p, 12)
        vals = [(_fluid_value(p, b, st, k, path), -k, k) for k in range(room + 1)]
        k = max(vals)[2]
        if k:
            acts.append(Buy("V1", k * p.v1_pallet_cases))
    if allocate and short - q2 > 0:
        avail = on_hand + q2
        order = sorted(range(len(p.customer_ids)), key=lambda i: -b.mean_goodwill_drop(p, i))
        ship = {}
        for i in order:
            c = p.customer_ids[i]
            ship[c] = min(orders[c], avail)
            avail -= ship[c]
        acts.append(Allocate(tuple((c, int(ship[c])) for c in p.customer_ids)))
    return acts


def _public_dump(obs: Obs) -> dict:
    return {"state": obs.state(), "market": obs.market(10_000)}


def rule_base_stock(obs: Obs) -> list:
    p = obs.world.p
    dump = _public_dump(obs)
    return plan(p, dump["state"], prior_beliefs(p, dump, "prior"))


def adaptive(obs: Obs) -> list:
    p = obs.world.p
    dump = {**_public_dump(obs), "ledger": obs.ledger()}
    return plan(p, dump["state"], beliefs_from_dump(p, dump, "own_ledger"), allocate=True)


@dataclass
class NoisyRule:
    """History generator: the rule policy with keyed random perturbations, giving several
    medium over- and under-buys across both vendors (labelled; not an evaluated policy)."""

    seed: int
    p_perturb: float = 0.6

    def __call__(self, obs: Obs) -> list:
        acts = rule_base_stock(obs)
        st = obs.state()
        if not st["purchasing_open"]:
            return acts
        r = stream(self.seed, "noisy_rule", obs.day)
        out = []
        p = obs.world.p
        offers = {o["vendor"]: o for o in st["vendor_offers"]}
        v1 = offers["V1"]
        room = v1["max_pallets_per_delivery"] - v1["pallets_already_ordered_for_that_day"]
        for a in acts:
            if isinstance(a, Buy) and a.vendor == "V1" and r.random() < self.p_perturb:
                k = a.cases // p.v1_pallet_cases + int(r.choice([-1, 1, 2]))
                a = Buy("V1", max(0, min(room, k)) * p.v1_pallet_cases)
            out.append(a)
        if v1["next_delivery_day"] == obs.day + 1 and not any(isinstance(a, Buy) and a.vendor == "V1" for a in out) \
                and r.random() < self.p_perturb / 2:
            out.append(Buy("V1", min(room, int(r.integers(1, 3))) * p.v1_pallet_cases))
        extra = int(r.integers(0, 15)) if r.random() < 0.3 else 0
        if extra:
            have = sum(a.cases for a in out if isinstance(a, Buy) and a.vendor == "V2")
            q = min(offers["V2"]["available_cases_today"], have + extra)
            out = [a for a in out if not (isinstance(a, Buy) and a.vendor == "V2")] + [Buy("V2", q)]
        return [a for a in out if not isinstance(a, Buy) or a.cases > 0]


@dataclass
class ScriptedOverbuy:
    """History generator: rule policy plus a full V1 delivery (max pallets) ordered on `day`."""

    day: int = 2

    def __call__(self, obs: Obs) -> list:
        acts = rule_base_stock(obs)
        if obs.day == self.day:
            st = obs.state()
            v1 = next(o for o in st["vendor_offers"] if o["vendor"] == "V1")
            room = v1["max_pallets_per_delivery"] - v1["pallets_already_ordered_for_that_day"]
            acts = [a for a in acts if not (isinstance(a, Buy) and a.vendor == "V1")] + [Buy("V1", room * obs.world.p.v1_pallet_cases)]
        return acts
