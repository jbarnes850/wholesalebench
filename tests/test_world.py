"""Tests for the v0.3 synthetic world fixture (no network; needs only the committed
calibration JSON under data/calibration/)."""

from __future__ import annotations

import re

import numpy as np
import pytest

from wsb.world import reference as ref
from wsb.world.beliefs import beliefs_from_dump, prior_beliefs
from wsb.world.engine import ActionError, Allocate, Buy, World, checkpoint, generate_exogenous, state_hash, summarize
from wsb.world.observe import ArmConfig, Obs, attribution_note, fork_arms, leak_report, placebo_note
from wsb.world.params import Scenario, WorldParams
from wsb.world.policies import NoisyRule, ScriptedOverbuy, adaptive, do_nothing, rule_base_stock

H = ArmConfig("history")
SCENARIOS = [Scenario(11, "calm"), Scenario(12, "shock", v1_decay_scale=9.5), Scenario(13, "sampled")]


@pytest.fixture(scope="module")
def w():
    return World(WorldParams())


def _run(w, sc, pol, until=None):
    return w.run(w.reset(sc), pol, until=until, obs_fn=lambda st: Obs(w, st, H))


def _fork(w, sc, day=9, gen=None):
    return checkpoint(_run(w, sc, gen or NoisyRule(sc.seed), until=day))


def _overbuy_fork(w):
    return _fork(w, Scenario(400, "shock", onset_window=(4, 6), v1_decay_scale=9.5), 9, ScriptedOverbuy())


# ----------------------------------------------------------------- generator


def test_scenario_families_respect_their_conditions(w):
    p = w.p
    calm = generate_exogenous(p, Scenario(5, "calm"))
    assert all(calm.state[d] == 0 for d in range(0, 9))  # conditioning stops before the first fork day
    shock = generate_exogenous(p, Scenario(5, "shock", onset_window=(3, 8)))
    onset = next(d for d in range(p.episode_days) if shock.state[d] > 0)
    assert shock.state[0] == 0 and 3 <= onset <= 8


# ----------------------------------------------------------------- accounting and transitions


@pytest.mark.parametrize("sc", SCENARIOS, ids=lambda s: f"{s.family}-{s.seed}")
@pytest.mark.parametrize("pol", [do_nothing, rule_base_stock, adaptive, NoisyRule(7), ScriptedOverbuy()],
                         ids=["none", "rule", "adaptive", "noisy", "overbuy"])
def test_identities_hold_every_day(w, sc, pol):
    s = _run(w, sc, pol)  # check_day raises IdentityError on any violation
    assert len(s.books) == w.last_day + 1
    assert s.ar == [] and s.ap == []  # the settlement tail clears every obligation
    tail = [b for b in s.books if b["day"] >= w.p.episode_days]
    assert all(b["ordered"] == 0 and b["spoiled"] == 0 and b["holding"] == 0 for b in tail)


def test_two_valid_policies_differ(w):
    a, b = summarize(w, _run(w, SCENARIOS[1], do_nothing)), summarize(w, _run(w, SCENARIOS[1], rule_base_stock))
    assert a["terminal_value_c"] != b["terminal_value_c"]
    assert b["fill_rate_episode"] > a["fill_rate_episode"]


def test_latent_vendor_quality_changes_outcomes(w):
    poor = summarize(w, _run(w, Scenario(12, "shock", v1_decay_scale=9.5), NoisyRule(12)))
    good = summarize(w, _run(w, Scenario(12, "shock", v1_decay_scale=12.0), NoisyRule(12)))
    assert poor["cases_spoiled"] > good["cases_spoiled"]


def test_deterministic_replay_and_checkpoint_reset(w):
    ck = _fork(w, SCENARIOS[1])
    h = state_hash(ck)
    a, b = checkpoint(ck), checkpoint(ck)
    w.run(a, rule_base_stock, obs_fn=lambda st: Obs(w, st, H))
    w.run(b, rule_base_stock, obs_fn=lambda st: Obs(w, st, H))
    assert state_hash(a) == state_hash(b)
    assert state_hash(ck) == h


def test_exogenous_paths_do_not_depend_on_actions(w):
    a, b = _run(w, SCENARIOS[1], do_nothing), _run(w, SCENARIOS[1], rule_base_stock)
    assert a.exo.market_c == b.exo.market_c and a.exo.order_noise == b.exo.order_noise
    assert a.exo.v2_avail == b.exo.v2_avail and a.exo.v2_age == b.exo.v2_age


def test_same_purchase_gets_same_lot_draws_across_branches(w):
    ck = _fork(w, SCENARIOS[1], day=8)  # Tuesday: no V1 cutoff, so only V2 differs
    a, b = checkpoint(ck), checkpoint(ck)
    w.apply(a, [Buy("V2", 5)])
    w.apply(b, [Buy("V2", 5)])
    la = {l.rng_key: l.lot_scale for l in a.lots}
    lb = {l.rng_key: l.lot_scale for l in b.lots}
    assert la == lb


# ----------------------------------------------------------------- actions and constraints


def test_action_validation(w):
    s = w.reset(SCENARIOS[0])
    for bad in ([Buy("V1", 50)], [Buy("V1", 48 * (w.p.v1_max_pallets + 1))], [Buy("V2", s.exo.v2_avail[0] + 1)],
                [Buy("V9", 1)], [Allocate((("C1", s.orders_today[0] + 1),))]):
        with pytest.raises(ActionError):
            w.apply(checkpoint(s), bad)
    late = w.run(w.reset(SCENARIOS[0]), do_nothing, until=w.p.episode_days)
    with pytest.raises(ActionError):
        w.apply(late, [Buy("V2", 1)])
    last_cutoff = w.run(w.reset(SCENARIOS[0]), do_nothing, until=w.p.episode_days - 1)  # Sunday before the tail
    with pytest.raises(ActionError):
        w.apply(last_cutoff, [Buy("V1", 48)])


def test_credit_line_blocks_cash_purchases(w):
    s = w.reset(SCENARIOS[0])
    s.cash_c = -w.p.credit_limit_c + 100
    with pytest.raises(ActionError):
        w.apply(s, [Buy("V2", 1)])


def test_allocation_overrides_priority_and_moves_goodwill(w):
    s = w.reset(SCENARIOS[0])
    w.apply(s, [Allocate(tuple((c, 0) for c in w.p.customer_ids))])
    ships = [e for e in s.events if e.type == "shipment" and e.day == 0]
    assert all(e.public["cases_shipped"] == 0 for e in ships)
    assert any(g < 1.0 for g in s.goodwill)


# ----------------------------------------------------------------- arms and information rule


def test_arms_start_identical_and_observing_does_not_mutate(w):
    ck = _fork(w, SCENARIOS[1])
    h = state_hash(ck)
    for arm in fork_arms(ck):
        st = checkpoint(ck)
        Obs(w, st, arm).dump()
        for pol in (rule_base_stock, adaptive):
            pol(Obs(w, st, arm))
        assert state_hash(st) == h


@pytest.mark.parametrize("day", [0, 5, 9, 16, 34])
def test_no_forbidden_information_in_any_arm(w, day):
    ck = _fork(w, SCENARIOS[1], day=day)
    for arm in fork_arms(ck):
        rep = leak_report(Obs(w, checkpoint(ck), arm))
        assert rep["ok"], (arm.name, rep)


def test_fresh_arm_view(w):
    ck = _overbuy_fork(w)
    arms = {a.name: a for a in fork_arms(ck)}
    fresh = Obs(w, checkpoint(ck), arms["fresh"])
    assert fresh.ledger() == []
    st = fresh.state()
    assert set(st["receivables"]) == {"open_total_c", "due_next_7_days_c"}  # totals only: no itemized or per-day history
    assert set(st["payables"]) == {"open_total_c", "due_next_7_days_c"}
    assert all(re.fullmatch(r"L-[0-9a-f]{8}", l["lot_id"]) for l in st["inventory_lots"])
    hist = Obs(w, checkpoint(ck), arms["history"]).ledger()
    assert hist and hist[0]["row"] == 1
    assert max(d for d, _ in fresh.market(10_000)) == ck.day


def test_rule_policy_is_history_blind(w):
    for sc in SCENARIOS:
        for day in (9, 16):
            ck = _fork(w, sc, day=day)
            acts, calls = set(), []
            for arm in fork_arms(ck):
                o = Obs(w, checkpoint(ck), arm)
                acts.add(ref.action_key(rule_base_stock(o)))
                calls.append(set(o.calls))
            assert len(acts) == 1
            assert all(c <= {"state", "market"} for c in calls)


# ----------------------------------------------------------------- notes


def test_notes_are_past_facts_and_length_matched(w):
    ck = _overbuy_fork(w)
    att = attribution_note(ck)
    assert att, "expected V1 write-offs before the fork in this world"
    assert all(int(x) <= ck.day for x in re.findall(r"day[- ](\d+)", att))
    for bad in ("should", "recommend", "fewer", "avoid", "future", "will"):
        assert bad not in att.lower()
    pl = placebo_note(ck, len(att))
    assert abs(len(pl) - len(att)) <= 0.1 * len(att)
    assert not re.search(r"day-\d+|\bV1\b|\bV2\b|order of", pl)


# ----------------------------------------------------------------- beliefs


def test_fresh_beliefs_ignore_history_and_history_beliefs_use_it(w):
    p = w.p
    ck = _overbuy_fork(w)
    arms = {a.name: a for a in fork_arms(ck)}
    fresh_dump = Obs(w, checkpoint(ck), arms["fresh"]).dump()
    bf = beliefs_from_dump(p, fresh_dump, "fresh")
    bh = beliefs_from_dump(p, Obs(w, checkpoint(ck), arms["history"]).dump(), "history")
    pr = prior_beliefs(p, fresh_dump)
    assert np.allclose(bf.v1_post, pr.v1_post)
    assert all(np.allclose(a, b) for a, b in zip(bf.gw_post, pr.gw_post))
    assert bh.n_v1_exposures > 0 and not np.allclose(bh.v1_post, pr.v1_post)


def test_opening_stock_is_evidence_in_history_only(w):
    p = w.p
    ck = _fork(w, Scenario(21, "calm", v1_decay_scale=7.5), day=9)
    arms = {a.name: a for a in fork_arms(ck)}
    led = Obs(w, checkpoint(ck), arms["history"]).ledger()
    assert led[0]["type"] == "opening_stock"
    from wsb.world.beliefs import _lot_exposures
    lots = _lot_exposures(led, p.episode_days)
    assert any(L["received_day"] < 0 and L["obs"] for L in lots.values())


def test_fresh_beliefs_invariant_to_prefork_records(w):
    """Changing pre-fork ledger content must not change the fresh arm's posterior."""
    p = w.p
    ck = _overbuy_fork(w)
    arms = {a.name: a for a in fork_arms(ck)}
    b1 = beliefs_from_dump(p, Obs(w, checkpoint(ck), arms["fresh"]).dump(), "fresh")
    tampered = checkpoint(ck)
    for e in tampered.events:
        if e.type == "spoilage_writeoff":
            e.public["cases"] += 7
    b2 = beliefs_from_dump(p, Obs(w, tampered, arms["fresh"]).dump(), "fresh")
    assert np.allclose(b1.v1_post, b2.v1_post) and np.allclose(b1.p_state, b2.p_state)


# ----------------------------------------------------------------- reference protocol


def test_reference_protocol(w):
    ck = _overbuy_fork(w)
    b = beliefs_from_dump(w.p, Obs(w, checkpoint(ck), H).dump(), "history")
    e1 = ref.evaluate_fork(w, ck, b, n_futures=6, seed=3)
    e2 = ref.evaluate_fork(w, ck, b, n_futures=6, seed=3, extra_actions=[[Buy("V2", 1)], [Buy("V1", 48)]])
    assert e1.best == e2.best and e1.tolerance_c == e2.tolerance_c and e1.span_c == e2.span_c  # frozen grid
    assert e1.score(e1.best)["regret_c"] == 0
    assert all((e1.eval_values[k] == e2.eval_values[k]).all() for k in e1.grid_keys)
    assert state_hash(ck) == state_hash(checkpoint(ck))
