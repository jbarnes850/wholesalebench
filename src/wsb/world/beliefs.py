"""Posterior beliefs computed ONLY from what an agent can retrieve (an Obs dump), v0.3.

Shared by the history-aware adaptive policy (its own ledger) and by the reference
evaluator (an arm's full dump), so both use one likelihood. Applied to the fresh
arm's dump, the same function yields the fresh arm's posterior. Information the
fresh arm cannot see therefore never enters it, and everything it can see does.

Uses the world family's structural model and priors (public customer profiles,
goodwill rule and sensitivity support, decay family, frailty and support,
regime model). Never touches hidden state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from wsb.world.params import WorldParams, lognormal_mu


@dataclass
class Beliefs:
    info_set: str
    p_state: np.ndarray  # filtered regime distribution today
    x_today: float  # today's log reference price
    v1_post: np.ndarray  # over decay_scale_support
    lot_post: dict[str, tuple[str, np.ndarray]] = field(default_factory=dict)  # alias -> (vendor, P(k, j | lot data))
    gw_post: list[np.ndarray] = field(default_factory=list)  # per customer, over goodwill_drop_support
    gw_now: list[list[float]] = field(default_factory=list)  # per customer, goodwill today under each hypothesis
    n_v1_exposures: int = 0
    n_v1_lots: int = 0
    n_shortfall_days: int = 0

    def p_glut(self) -> float:
        return float(1 - self.p_state[0])

    def expected_market_path(self, p: WorldParams, horizon: int) -> list[float]:
        """Approximate E[reference price] for the next `horizon` days (mean path of the log process)."""
        T = np.asarray(p.transition())
        mu = np.asarray([p.state_mean_log(k) for k in range(p.n_states)])
        b, x, out = self.p_state.copy(), self.x_today, []
        for _ in range(horizon):
            b = b @ T
            x = x + p.price_reversion * (float(b @ mu) - x)
            out.append(math.exp(x))
        return out

    def survival_ratio(self, p: WorldParams, vendor: str, a0: int, a1: int) -> float:
        nodes, weights = p.frailty_nodes()
        post = self.v1_post if vendor == "V1" else np.asarray(p.decay_scale_prior, float)
        out = 0.0
        for pk, sc in zip(post, p.decay_scale_support):
            for f, wj in zip(nodes, weights):
                s0 = p.survival(a0, sc * f)
                out += pk * wj * (p.survival(a1, sc * f) / s0 if s0 > 0 else 0.0)
        return out

    def goodwill_multiplier(self, p: WorldParams, i: int, days_ahead: int) -> float:
        out = 0.0
        for pk, g in zip(self.gw_post[i], self.gw_now[i]):
            gf = min(1.0, g + p.goodwill_recovery * days_ahead)
            out += pk * (1 - p.goodwill_weight * (1 - gf))
        return out

    def mean_goodwill_drop(self, p: WorldParams, i: int) -> float:
        return float(np.dot(self.gw_post[i], p.goodwill_drop_support))

    def summary(self, p: WorldParams) -> dict:
        return {"info_set": self.info_set, "p_glut_today": round(self.p_glut(), 4),
                "v1_decay_posterior": dict(zip(map(str, p.decay_scale_support), np.round(self.v1_post, 4).tolist())),
                "goodwill_sensitivity_posterior": {c: dict(zip(map(str, p.goodwill_drop_support), np.round(x, 4).tolist()))
                                                   for c, x in zip(p.customer_ids, self.gw_post)},
                "n_v1_lots": self.n_v1_lots, "n_v1_exposures": self.n_v1_exposures,
                "n_shortfall_days_seen": self.n_shortfall_days}


def _norm_logpdf(x, mu, sd):
    return -0.5 * ((x - mu) / sd) ** 2 - math.log(sd * math.sqrt(2 * math.pi))


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def regime_filter(p: WorldParams, market: list[list[int]]) -> tuple[np.ndarray, float]:
    """Forward filter for the 4-state regime model with mean-reverting log price."""
    T = np.asarray(p.transition())
    mu = [p.state_mean_log(k) for k in range(p.n_states)]
    b = np.asarray(p.stationary(), float)
    xs = [math.log(c) for _, c in market]
    for t, x in enumerate(xs):
        if t:
            b = b @ T
            lik = np.array([math.exp(_norm_logpdf(x, xs[t - 1] + p.price_reversion * (mu[k] - xs[t - 1]), p.price_log_sd))
                            for k in range(p.n_states)])
        else:
            lik = np.array([math.exp(_norm_logpdf(x, mu[k], 0.15)) for k in range(p.n_states)])
        b = b * np.maximum(lik, 1e-300)
        b = b / b.sum()
    return b, xs[-1]


def _lot_exposures(ledger: list[dict], episode_days: int | None = None) -> dict[str, dict]:
    """Replay every lot that has a receipt record: daily (n alive at check, written off, age)."""
    by_day: dict[int, dict[str, list]] = {}
    for e in ledger:
        if e["type"] in ("receipt", "spoilage_writeoff", "shipment", "opening_stock"):
            by_day.setdefault(e["day"], {"receipt": [], "spoilage_writeoff": [], "shipment": [], "opening_stock": []})[e["type"]].append(e)
    lots: dict[str, dict] = {}
    for d in sorted(by_day):
        ev = by_day[d]
        for r in ev["opening_stock"]:  # stock on hand before the first aging check
            lots[r["lot_id"]] = {"vendor": r["vendor"], "received_day": r["received_day"], "age0": r["age_at_receipt"],
                                 "n": r["cases"], "obs": []}
        w_today = {r["lot_id"]: r["cases"] for r in ev["spoilage_writeoff"]}
        aging = episode_days is None or d < episode_days  # no aging in the settlement tail
        for lid, L in lots.items():  # aging checks happen before today's receipts matter and before shipments
            if aging and L["n"] > 0 and L["received_day"] < d:
                w = w_today.get(lid, 0)
                L["obs"].append((L["n"], w, d - L["received_day"] + L["age0"]))
                L["n"] -= w
        for r in ev["receipt"]:
            lots[r["lot_id"]] = {"vendor": r["vendor"], "received_day": d, "age0": r["age_at_receipt"], "n": r["cases"], "obs": []}
        for sh in ev["shipment"]:
            for lid, q in sh["lots"]:
                if lid in lots:
                    lots[lid]["n"] -= q
    return lots


def decay_beliefs(p: WorldParams, ledger: list[dict]) -> tuple[np.ndarray, dict, int, int]:
    nodes, weights = p.frailty_nodes()
    K, J = len(p.decay_scale_support), len(nodes)
    lots = _lot_exposures(ledger, p.episode_days)
    log_v1 = np.log(np.asarray(p.decay_scale_prior, float))
    lot_post, n_exp, n_lots = {}, 0, 0
    for lid, L in lots.items():
        ll = np.zeros((K, J))
        for n, w, age in L["obs"]:
            for k, sc in enumerate(p.decay_scale_support):
                for j, f in enumerate(nodes):
                    h = min(max(p.hazard(age, sc * f), 1e-12), 1 - 1e-12)
                    ll[k, j] += w * math.log(h) + (n - w) * math.log(1 - h)
        joint = ll + np.log(np.asarray(weights))[None, :]
        if L["vendor"] == "V1":
            n_lots += 1
            n_exp += len(L["obs"])
            m = joint.max()
            log_v1 += np.log(np.exp(joint - m).sum(axis=1)) + m
            post = np.exp(joint - joint.max(axis=1, keepdims=True))
            post = post / post.sum(axis=1, keepdims=True)  # P(j | k, lot data)
        else:
            joint = joint + np.log(np.asarray(p.decay_scale_prior, float))[:, None]
            post = np.exp(joint - joint.max())
            post = post / post.sum()  # P(k, j | lot data) for an independent V2 lot
        lot_post[lid] = (L["vendor"], post)
    v1 = np.exp(log_v1 - log_v1.max())
    return v1 / v1.sum(), lot_post, n_exp, n_lots


def _order_loglik(p: WorldParams, i: int, d: int, o: int, mult: float) -> float:
    wd = p.customer_weekday[i][d % 7]
    if wd <= 0:
        return 0.0
    mu, sd = lognormal_mu(p.order_log_sigma), p.order_log_sigma
    scale = p.customer_base_cases[i] * wd * mult
    hi = _norm_cdf((math.log((o + 0.5) / scale) - mu) / sd)
    lo = _norm_cdf((math.log((o - 0.5) / scale) - mu) / sd) if o > 0 else 0.0
    return math.log(max(hi - lo, 1e-300))


def goodwill_beliefs(p: WorldParams, ledger: list[dict], orders_today: dict[str, int], today: int):
    """Per-customer posterior over goodwill sensitivity, replaying goodwill under each hypothesis."""
    orders: dict[int, dict[str, int]] = {}
    ships: dict[int, dict[str, tuple[int, int]]] = {}
    for e in ledger:
        if e["type"] == "customer_order":
            orders.setdefault(e["for_day"], {})[e["customer"]] = e["cases"]
        elif e["type"] == "shipment":
            ships.setdefault(e["day"], {})[e["customer"]] = (e["cases_ordered"], e["cases_shipped"])
    orders.setdefault(today, {}).update(orders_today)
    days = sorted(set(orders) | set(ships))
    posts, nows, shortfall_days = [], [], 0
    for i, c in enumerate(p.customer_ids):
        logpost = np.log(np.asarray(p.goodwill_drop_prior, float))
        g_now = []
        for k, drop in enumerate(p.goodwill_drop_support):
            g = 1.0
            started = False
            for d in (range(days[0], today + 1) if days else []):
                if d in ships or d in orders:
                    started = True
                if not started:
                    continue
                if d in orders and c in orders[d]:
                    logpost[k] += _order_loglik(p, i, d, orders[d][c], 1 - p.goodwill_weight * (1 - g))
                if d < today:
                    want, got = ships.get(d, {}).get(c, (0, 0))
                    if want:
                        g = max(0.0, g - drop * (want - got) / want)
                    g = min(1.0, g + p.goodwill_recovery)
            g_now.append(g)
        post = np.exp(logpost - logpost.max())
        posts.append(post / post.sum())
        nows.append(g_now)
    shortfall_days = sum(1 for d, v in ships.items() if any(w > s for w, s in v.values()))
    return posts, nows, shortfall_days


def beliefs_from_dump(p: WorldParams, dump: dict, info_set: str) -> Beliefs:
    st, led = dump["state"], dump["ledger"]
    p_state, x = regime_filter(p, dump["market"])
    v1, lot_post, n_exp, n_lots = decay_beliefs(p, led)
    gw_post, gw_now, n_short = goodwill_beliefs(p, led, st["customer_orders_today"], st["day"])
    return Beliefs(info_set, p_state, x, v1, lot_post, gw_post, gw_now, n_exp, n_lots, n_short)


def prior_beliefs(p: WorldParams, dump: dict, info_set: str = "prior") -> Beliefs:
    """Beliefs that ignore the ledger entirely (history-blind), using only the market tool."""
    p_state, x = regime_filter(p, dump["market"])
    k = len(p.goodwill_drop_support)
    return Beliefs(info_set, p_state, x, np.asarray(p.decay_scale_prior, float), {},
                   [np.asarray(p.goodwill_drop_prior, float) for _ in p.customer_ids],
                   [[1.0] * k for _ in p.customer_ids])
