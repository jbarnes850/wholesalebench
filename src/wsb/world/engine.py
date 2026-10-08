"""Event-sourced transition loop for the v0.3 buying-desk world.

One SKU (a 1 1/9 bushel carton of eggplant, "case"); a grower V1 (Monday and
Thursday deliveries in 48-case pallets, discount, 10-day terms) and a terminal
spot seller V2 (same day, limited availability, premium, varied age, cash on
purchase); three customers. Money is integer cents and quantity is integer
cases, so the identities in `check_day` hold exactly (tolerance 0).

Episode day d (0 <= d < episode_days) runs in two halves:
  begin_day(d): publish today's sell price, receive V1 deliveries due today,
                age every lot received before today (binomial per-case decay),
                reveal today's customer orders and today's V2 offer.
  apply(d, actions): validate; place V1 orders (next delivery day) and V2 buys
                (received now, paid today, within the credit line); allocate and
                ship FIFO; update customer goodwill; invoice; collect receivables
                due; pay payables due; charge holding and interest; close the book.
Settlement tail (d >= episode_days): no orders, no aging, no holding, no buying;
deliveries already ordered still arrive and are billed; receivables and payables
settle. Unsold stock is valued at a declared net realizable value.

Exogenous (never depend on actions): market regime and price path, the noise
component of customer orders, V2 availability and age.
Action-dependent, with ASSUMED response functions: lot aging of what was bought,
customer goodwill after shortfalls (scales later orders), receivables, payables,
cash and credit-line interest.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from wsb.world.params import Scenario, WorldParams, lognormal_mu
from wsb.world.rng import stream


# --------------------------------------------------------------------------- records

@dataclass
class Lot:
    lot_id: str
    vendor: str
    po_id: str
    received_day: int
    receipt_age: int
    qty_received: int
    qty: int
    unit_cost_c: int
    rng_key: tuple  # branch-invariant key for this lot's draws (vendor, order_day, k)
    lot_scale: float  # hidden: decay scale of this lot (vendor latent x frailty)
    decision_id: str  # hidden provenance
    ahead_decisions: tuple = ()  # hidden: decisions behind older stock on hand at receipt (FIFO queue ahead)


@dataclass
class PurchaseOrder:
    po_id: str
    decision_id: str
    vendor: str
    cases: int
    unit_cost_c: int
    order_day: int
    arrival_day: int
    rng_key: tuple
    receipt_age: int


@dataclass
class OpenItem:
    item_id: str
    party: str
    amount_c: int
    issue_day: int
    due_day: int
    ref: str | None = None


@dataclass
class Event:
    event_id: int
    type: str
    day: int
    visible_day: int
    public: dict
    hidden: dict = field(default_factory=dict)


@dataclass
class Exogenous:
    """Exogenous paths plus latent draws the agent never sees."""

    market_c: dict[int, int]
    state: dict[int, int]  # 0 normal, k>=1 glut of depth k
    log_price: dict[int, float]
    order_noise: dict[int, tuple[float, ...]]  # base x weekday x lognormal, before goodwill
    v2_avail: dict[int, int]
    v2_age: dict[int, int]
    v1_decay_scale: float
    goodwill_drop: tuple[float, ...]
    seed: int
    attempt: int = 0


@dataclass(frozen=True)
class Buy:
    vendor: str
    cases: int


@dataclass(frozen=True)
class Allocate:
    """Ship these quantities today instead of the default priority rule."""

    cases: tuple[tuple[str, int], ...]


class ActionError(ValueError):
    pass


@dataclass
class WorldState:
    day: int
    cash_c: int
    lots: list[Lot]
    pos: list[PurchaseOrder]
    ar: list[OpenItem]
    ap: list[OpenItem]
    price_list_c: int
    exo: Exogenous
    goodwill: list[float]  # hidden, per customer
    orders_today: tuple[int, ...] = ()
    v2_bought_today: int = 0
    events: list[Event] = field(default_factory=list)
    books: list[dict] = field(default_factory=list)
    counters: dict = field(default_factory=lambda: {"event": 0, "po": 0, "lot": 0, "inv": 0, "bill": 0, "decision": 0})
    open_book: dict | None = None
    record: bool = True


# --------------------------------------------------------------------------- generation

def _path(p: WorldParams, seed: int, attempt: int, first: int, last: int):
    T = p.transition()
    st = p.stationary()
    s = int(stream(seed, attempt, "regime0").choice(p.n_states, p=st))
    x = p.state_mean_log(s) + float(stream(seed, attempt, "price_noise", first).normal(0, p.price_log_sd))
    states, logp = {first: s}, {first: x}
    for d in range(first + 1, last + 1):
        u = float(stream(seed, attempt, "regime", d).random())
        cum, nxt = 0.0, s
        for j, pj in enumerate(T[s]):
            cum += pj
            if u < cum:
                nxt = j
                break
        s = nxt
        x = x + p.price_reversion * (p.state_mean_log(s) - x) + float(stream(seed, attempt, "price_noise", d).normal(0, p.price_log_sd))
        states[d], logp[d] = s, x
    return states, logp


def _accept(p: WorldParams, sc: Scenario, states: dict[int, int]) -> bool:
    ep = range(0, p.episode_days)
    if sc.family == "calm":  # calm start: no glut before the first fork day; later days follow the Markov model
        return all(states[d] == 0 for d in range(0, sc.onset_window[1] + 1))
    if sc.family == "shock":
        if states[0] != 0:
            return False
        onset = next((d for d in ep if states[d] > 0), None)
        return onset is not None and sc.onset_window[0] <= onset <= sc.onset_window[1]
    return True


def generate_exogenous(p: WorldParams, sc: Scenario) -> Exogenous:
    sc = sc.resolve(p)
    first = -p.pre_episode_market_days
    last = p.episode_days + p.settlement_tail_days
    for attempt in range(20000):
        states, logp = _path(p, sc.seed, attempt, first, last)
        if _accept(p, sc, states):
            break
    else:
        raise RuntimeError(f"no {sc.family} path found for seed {sc.seed}")
    market = {d: int(round(math.exp(x))) for d, x in logp.items()}
    v2, v2_age = {}, {}
    for d in range(first, last + 1):
        lo, hi = p.v2_avail_glut if states[d] else p.v2_avail_normal
        v2[d] = int(stream(sc.seed, "v2_avail", d).integers(lo, hi + 1))
        v2_age[d] = int(stream(sc.seed, "v2_age", d).integers(p.v2_receipt_age_range[0], p.v2_receipt_age_range[1] + 1))
    noise = {}
    mu = lognormal_mu(p.order_log_sigma)
    for d in range(0, p.episode_days):
        noise[d] = tuple(p.customer_base_cases[i] * p.customer_weekday[i][d % 7] *
                         math.exp(stream(sc.seed, "orders", c, d).normal(mu, p.order_log_sigma))
                         for i, c in enumerate(p.customer_ids))
    return Exogenous(market, states, logp, noise, v2, v2_age, float(sc.v1_decay_scale), tuple(sc.goodwill_drop),
                     sc.seed, attempt)


def price_list_for(p: WorldParams, exo: Exogenous, d: int) -> int:
    """Sell price for day d: markup on the same day's terminal reference (exogenous to the agent)."""
    return int(round(p.markup * exo.market_c[d]))


def realized_orders(p: WorldParams, noise: tuple[float, ...], goodwill: list[float]) -> tuple[int, ...]:
    return tuple(max(0, int(round(x * (1 - p.goodwill_weight * (1 - g))))) for x, g in zip(noise, goodwill))


def lot_scale_for(p: WorldParams, exo: Exogenous, vendor: str, rng_key: tuple) -> float:
    frailty = math.exp(float(stream(exo.seed, "frailty", rng_key).normal(0, p.lot_frailty_sigma)))
    if vendor == "V1":
        base = exo.v1_decay_scale
    else:  # V2: varied shippers, each lot's base scale drawn from the family support
        base = float(stream(exo.seed, "v2_base_scale", rng_key).choice(p.decay_scale_support, p=p.decay_scale_prior))
    return base * frailty


# --------------------------------------------------------------------------- world

class World:
    def __init__(self, params: WorldParams | None = None):
        self.p = params or WorldParams()

    # ----- helpers
    def _event(self, s: WorldState, typ: str, public: dict, hidden: dict | None = None):
        s.counters["event"] += 1
        if s.record:
            s.events.append(Event(s.counters["event"], typ, s.day, s.day, public, hidden or {}))

    def _next(self, s: WorldState, kind: str, prefix: str) -> str:
        s.counters[kind] += 1
        return f"{prefix}{s.counters[kind]:04d}"

    def vendor_idx(self, v: str) -> int:
        return self.p.vendor_ids.index(v)

    def offer_c(self, s: WorldState, vendor: str, day: int | None = None) -> int:
        d = s.day if day is None else day
        return int(round(s.exo.market_c[d] * self.p.vendor_spread[self.vendor_idx(vendor)]))

    def v1_committed(self, s: WorldState, arrival_day: int) -> int:
        return sum(x.cases for x in s.pos if x.vendor == "V1" and x.arrival_day == arrival_day)

    @staticmethod
    def inventory_cases(s: WorldState) -> int:
        return sum(l.qty for l in s.lots)

    @staticmethod
    def inventory_cost_c(s: WorldState) -> int:
        return sum(l.qty * l.unit_cost_c for l in s.lots)

    def equity_c(self, s: WorldState) -> int:
        return s.cash_c + sum(a.amount_c for a in s.ar) + self.inventory_cost_c(s) - sum(a.amount_c for a in s.ap)

    def in_tail(self, s: WorldState) -> bool:
        return s.day >= self.p.episode_days

    @property
    def last_day(self) -> int:
        return self.p.episode_days + self.p.settlement_tail_days

    def nrv_c(self, s: WorldState) -> int:
        """Net realizable value of stock at the end of the episode (declared assumption)."""
        p = self.p
        ref_day = p.episode_days - 1
        replacement = self.offer_c(s, "V1", ref_day)
        total = 0.0
        for l in s.lots:
            age = max(ref_day, l.received_day) - l.received_day + l.receipt_age
            total += l.qty * replacement * p.prior_survival_ratio(age, age + p.nrv_horizon_days)
        return int(round(total))

    def terminal_value_c(self, s: WorldState) -> int:
        return s.cash_c + sum(a.amount_c for a in s.ar) - sum(a.amount_c for a in s.ap) + self.nrv_c(s)

    # ----- reset
    def reset(self, scenario: Scenario, record: bool = True) -> WorldState:
        p = self.p
        exo = generate_exogenous(p, scenario)
        s = WorldState(day=0, cash_c=p.opening_cash_c, lots=[], pos=[], ar=[], ap=[], price_list_c=0, exo=exo,
                       goodwill=[1.0] * len(p.customer_ids), record=record)
        cost = int(round(exo.market_c[-2] * p.vendor_spread[0]))
        key = ("V1", -2, 0)
        opening = Lot("L0000", "V1", "OPENING", -1, p.v1_receipt_age, p.opening_inventory_cases,
                      p.opening_inventory_cases, cost, key, lot_scale_for(p, exo, "V1", key), "OPENING")
        s.lots.append(opening)
        # public record of the opening stock, so its later write-offs are usable evidence
        self._event(s, "opening_stock", {"lot_id": opening.lot_id, "vendor": "V1", "cases": opening.qty,
                                         "unit_cost_c": cost, "received_day": opening.received_day,
                                         "age_at_receipt": opening.receipt_age}, {"decision_id": "OPENING"})
        for items, total, n, party in ((s.ar, p.opening_ar_c, p.customer_terms_days, None),
                                       (s.ap, p.opening_ap_c, p.vendor_terms_days[0], "V1")):
            base, rem = divmod(total, n)
            for k in range(n):
                who = party or p.customer_ids[k % len(p.customer_ids)]
                tag = "OPEN-AR" if party is None else "OPEN-AP"
                items.append(OpenItem(f"{tag}{k:02d}", who, base + (1 if k < rem else 0), k - n, k + 1))
        self.begin_day(s)
        return s

    # ----- day halves
    def begin_day(self, s: WorldState) -> None:
        p, d = self.p, s.day
        s.open_book = {
            "day": d, "cash_open": s.cash_c, "ar_open": sum(a.amount_c for a in s.ar), "ap_open": sum(a.amount_c for a in s.ap),
            "inv_cases_open": self.inventory_cases(s), "inv_cost_open": self.inventory_cost_c(s), "equity_open": self.equity_c(s),
            "received": 0, "received_cost": 0, "billed": 0, "shipped": 0, "cogs": 0, "revenue": 0, "invoiced": 0,
            "spoiled": 0, "spoil_cost": 0, "collected": 0, "paid": 0, "holding": 0, "disposal": 0, "interest": 0,
            "ordered": 0, "short": 0, "credit_breach": False,
        }
        s.v2_bought_today = 0
        s.price_list_c = price_list_for(p, s.exo, d)
        self._event(s, "price_list", {"price_c": s.price_list_c, "valid_for_day": d})
        for po in [x for x in s.pos if x.arrival_day == d]:
            self._receive(s, po)
        if not self.in_tail(s):
            for lot in s.lots:
                if lot.received_day < d and lot.qty > 0:
                    self._decay(s, lot)
            s.lots = [l for l in s.lots if l.qty > 0]
            s.orders_today = realized_orders(p, s.exo.order_noise[d], s.goodwill)
            for c, q in zip(p.customer_ids, s.orders_today):
                self._event(s, "customer_order", {"customer": c, "cases": q, "for_day": d})
        else:
            s.orders_today = tuple(0 for _ in p.customer_ids)

    def _receive(self, s: WorldState, po: PurchaseOrder) -> None:
        p = self.p
        vi = self.vendor_idx(po.vendor)
        ahead = tuple(sorted({l.decision_id for l in s.lots if l.qty > 0}))
        lot = Lot(self._next(s, "lot", "L"), po.vendor, po.po_id, s.day, po.receipt_age, po.cases, po.cases,
                  po.unit_cost_c, po.rng_key, lot_scale_for(p, s.exo, po.vendor, po.rng_key), po.decision_id, ahead)
        s.lots.append(lot)
        if po in s.pos:
            s.pos.remove(po)
        amount = po.cases * po.unit_cost_c
        bill = OpenItem(self._next(s, "bill", "B"), po.vendor, amount, s.day, s.day + p.vendor_terms_days[vi], po.po_id)
        s.ap.append(bill)
        b = s.open_book
        b["received"] += po.cases
        b["received_cost"] += amount
        b["billed"] += amount
        self._event(s, "receipt", {"lot_id": lot.lot_id, "po_id": po.po_id, "vendor": po.vendor, "cases": po.cases,
                                   "unit_cost_c": po.unit_cost_c, "age_at_receipt": lot.receipt_age},
                    {"decision_id": po.decision_id})
        self._event(s, "vendor_bill", {"bill_id": bill.item_id, "vendor": po.vendor, "amount_c": amount,
                                       "due_day": bill.due_day, "po_id": po.po_id})

    def _decay(self, s: WorldState, lot: Lot) -> None:
        p = self.p
        age = s.day - lot.received_day + lot.receipt_age
        h = p.hazard(age, lot.lot_scale)
        n = int(stream(s.exo.seed, "decay", lot.rng_key, s.day).binomial(lot.qty, h)) if h > 0 else 0
        if n == 0:
            return
        cost = n * lot.unit_cost_c
        disposal = n * p.disposal_c_per_case
        b = s.open_book
        b["spoiled"] += n
        b["spoil_cost"] += cost
        b["disposal"] += disposal
        s.cash_c -= disposal
        lot.qty -= n
        self._event(s, "spoilage_writeoff", {"lot_id": lot.lot_id, "vendor": lot.vendor, "cases": n, "cost_c": cost,
                                             "received_day": lot.received_day, "age_days_since_harvest": age,
                                             "cases_remaining_in_lot": lot.qty, "disposal_c": disposal},
                    {"contributing_decisions": [lot.decision_id, *[x for x in lot.ahead_decisions if x != lot.decision_id]],
                     "cause": "age-dependent decay; purchase size of this lot plus older stock ahead of it in FIFO"})

    def validate(self, s: WorldState, actions: list) -> None:
        p = self.p
        buys = [a for a in actions if isinstance(a, Buy) and a.cases]
        if self.in_tail(s) and buys:
            raise ActionError("purchasing is closed during the settlement tail")
        v1 = v2 = 0
        allocs = 0
        for a in actions:
            if isinstance(a, Allocate):
                allocs += 1
                for c, q in a.cases:
                    if c not in p.customer_ids or not isinstance(q, (int, np.integer)) or q < 0:
                        raise ActionError(f"bad allocation entry {(c, q)!r}")
                    if q > s.orders_today[p.customer_ids.index(c)]:
                        raise ActionError(f"allocation to {c} exceeds today's order")
                continue
            if not isinstance(a, Buy):
                raise ActionError(f"unknown action {a!r}")
            if a.vendor not in p.vendor_ids:
                raise ActionError(f"unknown vendor {a.vendor!r}")
            if not isinstance(a.cases, (int, np.integer)) or a.cases < 0:
                raise ActionError("cases must be a non-negative integer")
            if a.vendor == "V1":
                if a.cases % p.v1_pallet_cases:
                    raise ActionError(f"V1 sells whole pallets of {p.v1_pallet_cases} cases")
                v1 += int(a.cases)
            else:
                v2 += int(a.cases)
        if allocs > 1:
            raise ActionError("at most one allocation per day")
        nxt = p.next_v1_delivery(s.day)
        if v1 and nxt >= p.episode_days:
            raise ActionError(f"the next V1 delivery (day {nxt}) falls after the episode")
        if v1 + self.v1_committed(s, nxt) > p.v1_max_pallets * p.v1_pallet_cases:
            raise ActionError(f"V1 delivers at most {p.v1_max_pallets} pallets on day {nxt}")
        if v2 + s.v2_bought_today > s.exo.v2_avail[s.day]:
            raise ActionError(f"V2 has {s.exo.v2_avail[s.day] - s.v2_bought_today} cases available today")
        if v2 and s.cash_c - v2 * self.offer_c(s, "V2") < -p.credit_limit_c:
            raise ActionError("V2 is cash on purchase and this buy would exceed the credit line")

    def apply(self, s: WorldState, actions: list) -> WorldState:
        p, d = self.p, s.day
        self.validate(s, actions)
        b = s.open_book
        alloc = next((a for a in actions if isinstance(a, Allocate)), None)
        k_by_vendor: dict[str, int] = {}
        for a in actions:
            if not isinstance(a, Buy) or a.cases == 0:
                continue
            k = k_by_vendor.get(a.vendor, 0)
            k_by_vendor[a.vendor] = k + 1
            did = self._next(s, "decision", "D")
            v1 = a.vendor == "V1"
            po = PurchaseOrder(self._next(s, "po", "PO"), did, a.vendor, int(a.cases), self.offer_c(s, a.vendor), d,
                               p.next_v1_delivery(d) if v1 else d, (a.vendor, d, k),
                               p.v1_receipt_age if v1 else s.exo.v2_age[d])
            self._event(s, "purchase_order", {"po_id": po.po_id, "vendor": po.vendor, "cases": po.cases,
                                              "unit_cost_c": po.unit_cost_c, "arrival_day": po.arrival_day},
                        {"decision_id": did})
            if v1:
                s.pos.append(po)
            else:
                s.v2_bought_today += po.cases
                self._receive(s, po)
        if alloc is not None:
            self._event(s, "allocation", {"cases": dict(alloc.cases)})
        if not self.in_tail(s):
            self._fulfil(s, alloc)
        for a in [x for x in s.ar if x.due_day == d]:
            s.cash_c += a.amount_c
            b["collected"] += a.amount_c
            s.ar.remove(a)
            self._event(s, "customer_payment", {"invoice_id": a.item_id, "customer": a.party, "amount_c": a.amount_c})
        for a in [x for x in s.ap if x.due_day == d]:
            s.cash_c -= a.amount_c
            b["paid"] += a.amount_c
            s.ap.remove(a)
            self._event(s, "vendor_payment", {"bill_id": a.item_id, "vendor": a.party, "amount_c": a.amount_c})
        if not self.in_tail(s):
            holding = self.inventory_cases(s) * p.holding_c_per_case_day
            s.cash_c -= holding
            b["holding"] = holding
        if s.cash_c < 0:
            within = min(-s.cash_c, p.credit_limit_c)
            beyond = max(0, -s.cash_c - p.credit_limit_c)
            interest = int(math.ceil(within * p.overdraft_daily_rate + beyond * p.over_limit_daily_rate))
            s.cash_c -= interest
            b["interest"] = interest
            b["credit_breach"] = beyond > 0
            self._event(s, "interest_charge", {"amount_c": interest, "over_credit_limit": beyond > 0})
        self._close_book(s)
        s.day += 1
        if s.day <= self.last_day:
            self.begin_day(s)
        return s

    def _fulfil(self, s: WorldState, alloc: Allocate | None) -> None:
        p, d, b = self.p, s.day, s.open_book
        s.lots.sort(key=lambda l: (l.received_day, l.lot_id))
        available = self.inventory_cases(s)
        if alloc is not None:
            plan = dict(alloc.cases)
            ship = [min(plan.get(c, 0), q) for c, q in zip(p.customer_ids, s.orders_today)]
            if sum(ship) > available:
                raise ActionError("allocation exceeds stock on hand")
        else:  # default: fixed customer priority C1 > C2 > C3
            ship, left = [], available
            for q in s.orders_today:
                ship.append(min(q, left))
                left -= ship[-1]
        lot_dec = {l.lot_id: l.decision_id for l in s.lots}
        for i, c in enumerate(p.customer_ids):
            want, got = s.orders_today[i], ship[i]
            used, need = [], got
            for lot in s.lots:
                if need == 0:
                    break
                take = min(lot.qty, need)
                if take:
                    lot.qty -= take
                    need -= take
                    b["cogs"] += take * lot.unit_cost_c
                    used.append((lot.lot_id, take))
            b["ordered"] += want
            b["shipped"] += got
            b["short"] += want - got
            if want:
                s.goodwill[i] = max(0.0, s.goodwill[i] - s.exo.goodwill_drop[i] * (want - got) / want)
            s.goodwill[i] = min(1.0, s.goodwill[i] + p.goodwill_recovery)
            if got:
                amount = got * s.price_list_c
                inv = OpenItem(self._next(s, "inv", "I"), c, amount, d, d + p.customer_terms_days)
                s.ar.append(inv)
                b["revenue"] += amount
                b["invoiced"] += amount
                self._event(s, "customer_invoice", {"invoice_id": inv.item_id, "customer": c, "amount_c": amount,
                                                    "due_day": inv.due_day})
            self._event(s, "shipment", {"customer": c, "cases_ordered": want, "cases_shipped": got,
                                        "price_c": s.price_list_c, "lots": used},
                        {"contributing_decisions": sorted({lot_dec[x] for x, _ in used})})
        s.lots = [l for l in s.lots if l.qty > 0]

    def _close_book(self, s: WorldState) -> None:
        b = s.open_book
        b.update({
            "cash_close": s.cash_c, "ar_close": sum(a.amount_c for a in s.ar), "ap_close": sum(a.amount_c for a in s.ap),
            "inv_cases_close": self.inventory_cases(s), "inv_cost_close": self.inventory_cost_c(s), "equity_close": self.equity_c(s),
            "cash_negative": s.cash_c < 0,
        })
        b["net_income"] = b["revenue"] - b["cogs"] - b["spoil_cost"] - b["holding"] - b["disposal"] - b["interest"]
        if s.record:
            check_day(b)
        s.books.append(b)
        s.open_book = None

    def run(self, s: WorldState, policy, until: int | None = None, obs_fn=None) -> WorldState:
        until = self.last_day + 1 if until is None else until
        while s.day < until:
            acts = policy(obs_fn(s) if obs_fn else s) if not self.in_tail(s) else []
            self.apply(s, acts)
        return s


# --------------------------------------------------------------------------- invariants

IDENTITIES = {
    "quantity": "inv_cases_close == inv_cases_open + received - shipped - spoiled",
    "inventory_cost": "inv_cost_close == inv_cost_open + received_cost - cogs - spoil_cost",
    "cash": "cash_close == cash_open + collected - paid - holding - disposal - interest",
    "receivables": "ar_close == ar_open + invoiced - collected",
    "payables": "ap_close == ap_open + billed - paid",
    "accrual_vs_balance_sheet": "equity_close - equity_open == net_income, equity = cash + AR + inventory at cost - AP; "
                                "net_income = revenue - COGS - spoilage at cost - holding - disposal - interest",
    "fulfilment": "shipped + short == ordered",
}
TOLERANCE = "exact (integer cents, integer cases)"


class IdentityError(AssertionError):
    pass


def check_day(b: dict) -> None:
    checks = {
        "quantity": b["inv_cases_close"] == b["inv_cases_open"] + b["received"] - b["shipped"] - b["spoiled"],
        "inventory_cost": b["inv_cost_close"] == b["inv_cost_open"] + b["received_cost"] - b["cogs"] - b["spoil_cost"],
        "cash": b["cash_close"] == b["cash_open"] + b["collected"] - b["paid"] - b["holding"] - b["disposal"] - b["interest"],
        "receivables": b["ar_close"] == b["ar_open"] + b["invoiced"] - b["collected"],
        "payables": b["ap_close"] == b["ap_open"] + b["billed"] - b["paid"],
        "accrual_vs_balance_sheet": b["equity_close"] - b["equity_open"] == b["net_income"],
        "fulfilment": b["shipped"] + b["short"] == b["ordered"],
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise IdentityError(f"day {b['day']}: identity violated: {bad}")


# --------------------------------------------------------------------------- checkpoints

def checkpoint(s: WorldState) -> WorldState:
    return copy.deepcopy(s)


def state_hash(s: WorldState) -> str:
    """Hash of the full world state: hidden fields, pending orders, open book and the event log included."""
    payload = {
        "day": s.day, "cash_c": s.cash_c, "price_list_c": s.price_list_c, "goodwill": s.goodwill,
        "orders_today": s.orders_today, "v2_bought_today": s.v2_bought_today, "record": s.record,
        "lots": [asdict(l) for l in s.lots], "pos": [asdict(x) for x in s.pos],
        "ar": [asdict(x) for x in s.ar], "ap": [asdict(x) for x in s.ap],
        "exo": asdict(s.exo), "events": [asdict(e) for e in s.events], "counters": s.counters,
        "books": s.books, "open_book": s.open_book,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def summarize(w: World, s: WorldState) -> dict:
    books = s.books
    ep = [b for b in books if b["day"] < w.p.episode_days]
    ordered = sum(b["ordered"] for b in ep)
    shipped = sum(b["shipped"] for b in ep)
    received = sum(b["received"] for b in books)
    spoiled = sum(b["spoiled"] for b in books)
    return {
        "final_day": s.day - 1,
        "terminal_value_c": w.terminal_value_c(s),
        "terminal_cash_c": s.cash_c, "nrv_c": w.nrv_c(s),
        "open_ar_c": sum(a.amount_c for a in s.ar), "open_ap_c": sum(a.amount_c for a in s.ap),
        "inventory_cases_left": w.inventory_cases(s),
        "net_income_c": sum(b["net_income"] for b in books),
        "revenue_c": sum(b["revenue"] for b in books),
        "fill_rate_episode": round(shipped / ordered, 4) if ordered else None,
        "cases_ordered_episode": ordered,
        "cases_received": received, "cases_spoiled": spoiled,
        "waste_rate": round(spoiled / (received + w.p.opening_inventory_cases), 4),
        "days_cash_negative": sum(b["cash_negative"] for b in books),
        "days_over_credit_limit": sum(b["credit_breach"] for b in books),
        "identity_days_checked": len(books),
    }
