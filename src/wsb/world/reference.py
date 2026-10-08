"""Synthetic reference evaluation for a fork decision (v0.3).

Information set: everything the HISTORY arm can retrieve at the fork, plus the
world family's structural model and priors (wsb.world.beliefs). The reference
never reads realized latents (V1 decay scale, lot frailties, goodwill
sensitivities, true goodwill), the regime path or future draws. Hidden
quantities of today's state are re-drawn from their posteriors in every future.

Valuation: posterior-predictive expected terminal value (cash + receivables -
payables + declared net realizable value of unsold stock, after the settlement
tail). Futures are keyed by (seed, future index, kind, day[, entity]) and shared
across actions (common random numbers). After the fork action every future
follows the same continuation: the shared planner with fork-time beliefs held
fixed. This is a one-step lookahead with a heuristic continuation, not a
proven optimum.

Protocol per fork, fixed before any policy is scored:
  1. a frozen candidate grid (V1 pallets x {0, today's shortfall, all available V2});
  2. the reference best is chosen on SELECTION futures;
  3. every action (grid and policies) is scored on separate EVALUATION futures,
     so regret can be negative and adding another policy's action never changes
     the target, the normalization or the tolerance;
  4. one tolerance per fork: max($25, 2 x median paired SE of grid actions vs best).

Scores are synthetic-reference scores, not operator agreement.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from wsb.world.beliefs import Beliefs, regime_filter
from wsb.world.engine import Allocate, Buy, Exogenous, World, WorldState
from wsb.world.observe import ArmConfig, Obs, alias
from wsb.world.params import lognormal_mu
from wsb.world.policies import plan
from wsb.world.rng import key_seed, stream

TOLERANCE_FLOOR_C = 2500  # $25


def _copy_without_log(s: WorldState) -> WorldState:
    ev, bk = s.events, s.books
    s.events, s.books = [], []
    try:
        out = copy.deepcopy(s)
    finally:
        s.events, s.books = ev, bk
    out.record = False
    return out


def sample_future(w: World, s0: WorldState, b: Beliefs, j: int, seed: int) -> WorldState:
    """One posterior-predictive future; every draw keyed by (seed, j, kind, ...)."""
    p, d0 = w.p, s0.day
    s = _copy_without_log(s0)
    K = len(p.decay_scale_support)
    nodes, weights = p.frailty_nodes()
    k_v1 = int(stream(seed, "future", j, "v1_scale").choice(K, p=b.v1_post))
    for lot in s.lots:
        a = alias(s0.exo.seed, lot.lot_id)
        r = stream(seed, "future", j, "lot", a)
        if a in b.lot_post and lot.vendor == "V1":
            jj = int(r.choice(len(nodes), p=b.lot_post[a][1][k_v1]))
            lot.lot_scale = p.decay_scale_support[k_v1] * nodes[jj]
        elif a in b.lot_post:
            flat = b.lot_post[a][1].ravel()
            kk, jj = divmod(int(r.choice(flat.size, p=flat)), len(nodes))
            lot.lot_scale = p.decay_scale_support[kk] * nodes[jj]
        else:  # no ledger record for this lot in this information set
            kk = k_v1 if lot.vendor == "V1" else int(r.choice(K, p=p.decay_scale_prior))
            lot.lot_scale = p.decay_scale_support[kk] * nodes[int(r.choice(len(nodes), p=weights))]
    drops, gws = [], []
    for i, c in enumerate(p.customer_ids):
        kg = int(stream(seed, "future", j, "goodwill", c).choice(len(p.goodwill_drop_support), p=b.gw_post[i]))
        drops.append(p.goodwill_drop_support[kg])
        gws.append(b.gw_now[i][kg])
    s.goodwill = gws
    T = p.transition()
    market, state, logp = dict(s.exo.market_c), dict(s.exo.state), dict(s.exo.log_price)
    noise, v2, v2_age = dict(s.exo.order_noise), dict(s.exo.v2_avail), dict(s.exo.v2_age)
    st = int(stream(seed, "future", j, "state0").choice(p.n_states, p=b.p_state))
    x = b.x_today
    mu0 = lognormal_mu(p.order_log_sigma)
    for d in range(d0 + 1, w.last_day + 1):
        st = int(stream(seed, "future", j, "regime", d).choice(p.n_states, p=T[st]))
        x = x + p.price_reversion * (p.state_mean_log(st) - x) + float(stream(seed, "future", j, "price", d).normal(0, p.price_log_sd))
        state[d], logp[d], market[d] = st, x, int(round(math.exp(x)))
        lo, hi = p.v2_avail_glut if st else p.v2_avail_normal
        v2[d] = int(stream(seed, "future", j, "v2_avail", d).integers(lo, hi + 1))
        v2_age[d] = int(stream(seed, "future", j, "v2_age", d).integers(p.v2_receipt_age_range[0], p.v2_receipt_age_range[1] + 1))
        if d < p.episode_days:
            noise[d] = tuple(p.customer_base_cases[i] * p.customer_weekday[i][d % 7] *
                             math.exp(float(stream(seed, "future", j, "orders", c, d).normal(mu0, p.order_log_sigma)))
                             for i, c in enumerate(p.customer_ids))
    v1_scale = p.decay_scale_support[k_v1]
    s.exo = Exogenous(market, state, logp, noise, v2, v2_age, v1_scale, tuple(drops), key_seed(seed, "future_world", j))
    return s


def continuation(w: World, b: Beliefs):
    """Shared planner with fork-time vendor and customer beliefs held fixed. The market belief is
    re-filtered from each day's visible prices on V1 cutoff days, so later orders see later prices."""
    arm = ArmConfig("continuation", history=False)
    p = w.p

    def act(s: WorldState) -> list:
        st = Obs(w, s, arm).state()
        bb = b
        if p.next_v1_delivery(s.day) == s.day + 1:
            hist = [[d, s.exo.market_c[d]] for d in range(-p.pre_episode_market_days, s.day + 1)]
            p_state, x = regime_filter(p, hist)
            bb = dataclasses.replace(b, p_state=p_state, x_today=x)
        return plan(p, st, bb, allocate=False)

    return act


def rollout_value(w: World, s: WorldState, first: list, cont) -> int:
    w.apply(s, first)
    while s.day <= w.last_day:
        w.apply(s, cont(s) if not w.in_tail(s) else [])
    return w.terminal_value_c(s)


def action_key(acts) -> tuple:
    agg: dict[str, int] = {}
    alloc = None
    for a in acts:
        if isinstance(a, Buy):
            agg[a.vendor] = agg.get(a.vendor, 0) + int(a.cases)
        elif isinstance(a, Allocate):
            alloc = ("allocate", tuple(a.cases))
    key = tuple(sorted((v, q) for v, q in agg.items() if q))
    return key + ((alloc,) if alloc else ())


def key_actions(key: tuple) -> list:
    out = []
    for item in key:
        if item[0] == "allocate":
            out.append(Allocate(item[1]))
        else:
            out.append(Buy(item[0], item[1]))
    return out


def key_dict(key: tuple) -> dict:
    return {(k if k != "allocate" else "allocate"): (v if k != "allocate" else dict(v)) for k, v in key}


def grid(w: World, s: WorldState) -> list[list[Buy]]:
    p = w.p
    nxt = p.next_v1_delivery(s.day)
    room = p.v1_max_pallets - w.v1_committed(s, nxt) // p.v1_pallet_cases if nxt < p.episode_days else 0
    avail = s.exo.v2_avail[s.day] - s.v2_bought_today  # visible in today's offer
    cash_room = max(0, (s.cash_c + p.credit_limit_c) // max(1, w.offer_c(s, "V2")))
    avail = min(avail, cash_room)
    short = max(0, sum(s.orders_today) - w.inventory_cases(s))
    v2_opts = sorted({0, min(short, avail), avail})
    return [[x for x in (Buy("V1", k * p.v1_pallet_cases), Buy("V2", q)) if x.cases]
            for k in range(room + 1) for q in v2_opts]


@dataclass
class ForkEvaluation:
    grid_keys: list[tuple]
    best: tuple  # chosen on selection futures
    eval_values: dict[tuple, np.ndarray]  # evaluation futures, grid + scored actions
    tolerance_c: float
    span_c: float
    beliefs: Beliefs
    n_futures: int

    def score(self, key: tuple) -> dict:
        vb, va = self.eval_values[self.best], self.eval_values[key]
        diff = vb - va
        se = float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else 0.0
        regret = float(diff.mean())
        return {"action": key_dict(key), "expected_value_c": round(float(va.mean())), "reference_best": key_dict(self.best),
                "regret_c": round(regret), "regret_se_c": round(se), "tolerance_c": round(self.tolerance_c),
                "normalized_regret": round(regret / self.span_c, 4) if self.span_c > 0 else 0.0,
                "synthetic_acceptable": bool(regret <= self.tolerance_c)}

    def best_is_boundary(self, w: World) -> bool:
        q1 = key_dict(self.best).get("V1", 0) // w.p.v1_pallet_cases
        max_q1 = max(key_dict(k).get("V1", 0) for k in self.grid_keys) // w.p.v1_pallet_cases
        return q1 == max_q1 and max_q1 > 0

    def table(self, top: int = 6) -> list[dict]:
        rows = [{"action": key_dict(k), "mean_c": round(float(self.eval_values[k].mean())),
                 "se_c": round(float(self.eval_values[k].std(ddof=1) / math.sqrt(self.n_futures)))} for k in self.grid_keys]
        return sorted(rows, key=lambda r: -r["mean_c"])[:top]


def _values(w: World, futures: list[WorldState], acts: list, cont) -> np.ndarray:
    return np.array([rollout_value(w, _copy_without_log(f), acts, cont) for f in futures], dtype=float)


def evaluate_fork(w: World, s0: WorldState, b: Beliefs, extra_actions=(), n_futures: int = 48, seed: int = 7) -> ForkEvaluation:
    cont = continuation(w, b)
    cands = {action_key(c): c for c in grid(w, s0)}
    sel = [sample_future(w, s0, b, j, key_seed(seed, "selection")) for j in range(n_futures)]
    best = max(cands, key=lambda k: (_values(w, sel, cands[k], cont).mean(), -sum(q for _, q in k)))
    ev = [sample_future(w, s0, b, j, key_seed(seed, "evaluation")) for j in range(n_futures)]
    values = {k: _values(w, ev, a, cont) for k, a in cands.items()}
    for a in extra_actions:
        k = action_key(a)
        if k not in values:
            values[k] = _values(w, ev, key_actions(k), cont)
    ses = [float((values[best] - values[k]).std(ddof=1) / math.sqrt(n_futures)) for k in cands if k != best]
    tol = max(TOLERANCE_FLOOR_C, 2 * float(np.median(ses))) if ses else TOLERANCE_FLOOR_C
    means = [float(values[k].mean()) for k in cands]
    return ForkEvaluation(list(cands), best, values, tol, max(means) - min(means), b, n_futures)


def pair_test(w: World, s0: WorldState, b: Beliefs, a: tuple, c: tuple, n_futures: int = 256, seed: int = 11,
              threshold_c: int = TOLERANCE_FLOOR_C) -> dict:
    """Regret of action c vs a on a fresh, independently keyed future set (not the scoring futures),
    with an equivalence-style label: informative if the lower 95% bound exceeds 0 and the mean exceeds
    the threshold; null if the upper bound is below the threshold; otherwise undetermined."""
    if a == c:
        return {"mean_c": 0, "se_c": 0, "label": "null", "n_futures": 0}
    cont = continuation(w, b)
    fut = [sample_future(w, s0, b, j, key_seed(seed, "voh")) for j in range(n_futures)]
    va, vc = _values(w, fut, key_actions(a), cont), _values(w, fut, key_actions(c), cont)
    diff = va - vc
    m, se = float(diff.mean()), float(diff.std(ddof=1) / math.sqrt(n_futures))
    if m - 2 * se > 0 and m > threshold_c:
        label = "informative"
    elif m + 2 * se < threshold_c:
        label = "null"
    else:
        label = "undetermined"
    return {"mean_c": round(m), "se_c": round(se), "label": label, "n_futures": n_futures}


def select_best(w: World, s0: WorldState, b: Beliefs, n_futures: int = 48, seed: int = 7) -> tuple:
    """The grid action a reference with beliefs `b` would choose (selection futures only)."""
    cont = continuation(w, b)
    cands = {action_key(c): c for c in grid(w, s0)}
    sel = [sample_future(w, s0, b, j, key_seed(seed, "selection")) for j in range(n_futures)]
    return max(cands, key=lambda k: (_values(w, sel, cands[k], cont).mean(), -sum(q for _, q in k)))


def realized_path_values(w: World, s0: WorldState, b: Beliefs, actions) -> dict[tuple, int]:
    """Diagnostic only: each action on the REALIZED future (true latents and draws) under the same
    heuristic continuation. Not an upper bound and not a fair reference."""
    cont = continuation(w, b)
    return {action_key(a): rollout_value(w, _copy_without_log(s0), key_actions(action_key(a)), cont) for a in actions}
