"""Run the v0.3 fixture and write reports/fixture/fixture_report.json.

Demonstrates (each item is also asserted in tests/test_world.py):
  1. Valid policies produce different outcomes; every day passes the quantity,
     inventory-cost, cash, receivable, payable, accrual and fulfilment identities
     exactly (integer cents and cases).
  2. Determinism and resettable checkpoints.
  3. History visibility does not change the starting world state.
  4. Forbidden information is unavailable through every tool in every arm.
  5. The history-blind rule policy acts identically in every arm (zero gain).
  6. The history-aware adaptive policy can act differently; every arm is scored
     against the same frozen grid and evaluation futures.
  7. Each fork reports the reference's value of history (VOH) and classifies the fork on an
     independent 256-future set: informative (lower 95% bound > 0 and mean > $25), null
     (upper bound < $25: changing behaviour is unwarranted) or undetermined.

Pre-fork histories come from labelled generators (a noisy rule policy, or the rule
policy plus one full V1 delivery), not from the evaluated policy. They test use of
records, not learning from one's own actions. No language model was run.
"""

from __future__ import annotations

import json
import random
import sys
import time
from statistics import mean, median

from wsb.lineage import ROOT
from wsb.world import reference as ref
from wsb.world.beliefs import beliefs_from_dump
from wsb.world.engine import IDENTITIES, TOLERANCE, World, checkpoint, state_hash, summarize
from wsb.world.observe import ArmConfig, Obs, fork_arms, leak_report
from wsb.world.params import Scenario, WorldParams
from wsb.world.policies import NoisyRule, ScriptedOverbuy, adaptive, do_nothing, rule_base_stock

OUT = ROOT / "reports" / "fixture"
FORK_DAYS = (9, 16)  # Wednesdays: order cutoffs for Thursday V1 deliveries
POLICIES = {"do_nothing": do_nothing, "rule_base_stock": rule_base_stock, "adaptive": adaptive}
H = ArmConfig("history")
N_FUTURES = 64
VOH_FUTURES = 256


def episode_runs(w: World) -> list[dict]:
    rows = []
    scen = {f"{fam}-{seed}": Scenario(seed, fam) for fam in ("calm", "sampled", "shock") for seed in (11, 12, 13)}
    scen["shock-11-v1-poor"] = Scenario(11, "shock", v1_decay_scale=7.5)
    scen["shock-11-v1-good"] = Scenario(11, "shock", v1_decay_scale=12.0)
    for name, sc in scen.items():
        for pname, pol in {**POLICIES, "noisy_rule": NoisyRule(sc.seed)}.items():
            s = w.reset(sc)
            w.run(s, pol, obs_fn=lambda st: Obs(w, st, H))
            s2 = w.reset(sc)
            w.run(s2, pol, obs_fn=lambda st: Obs(w, st, H))
            ep = [b for b in s.books if b["day"] < w.p.episode_days]
            rows.append({"scenario": name, "policy": pname, **summarize(w, s),
                         "stockout_day_share": round(sum(b["short"] > 0 for b in ep) / len(ep), 3),
                         "deterministic_replay": state_hash(s) == state_hash(s2)})
    return rows


def fork_specs() -> list[dict]:
    specs = []
    p = WorldParams()
    for fam in ("calm", "sampled", "shock"):
        for i in range(6):
            # V1 latent pinned alternately poor/good so both appear (labelled); goodwill drawn from the prior
            v1 = p.decay_scale_support[i % 2]
            sc = Scenario(300 + 10 * ("calm", "sampled", "shock").index(fam) + i, fam, v1_decay_scale=v1)
            for d in FORK_DAYS:
                specs.append({"scenario": sc, "fork_day": d, "history": "noisy_rule"})
    for i in range(4):  # overbuy-before-glut pattern: a full V1 delivery just before a price fall
        specs.append({"scenario": Scenario(400 + i, "shock", onset_window=(4, 6), v1_decay_scale=p.decay_scale_support[i % 2]),
                      "fork_day": 9, "history": "scripted_overbuy"})
    return specs


def run_fork(w: World, spec: dict) -> dict:
    p, sc, fd = w.p, spec["scenario"], spec["fork_day"]
    gen = NoisyRule(sc.seed) if spec["history"] == "noisy_rule" else ScriptedOverbuy()
    s = w.reset(sc)
    w.run(s, gen, until=fd, obs_fn=lambda st: Obs(w, st, H))
    ck = checkpoint(s)
    h0 = state_hash(ck)
    arms = fork_arms(ck)
    b_hist = beliefs_from_dump(p, Obs(w, checkpoint(ck), arms[0]).dump(), "history")
    b_none = beliefs_from_dump(p, Obs(w, checkpoint(ck), arms[1]).dump(), "fresh")
    per_arm, all_actions = {}, []
    for arm in arms:
        a_state = checkpoint(ck)
        acts = {k: pol(Obs(w, a_state, arm)) for k, pol in POLICIES.items()}
        all_actions += list(acts.values())
        per_arm[arm.name] = {"start_hash": state_hash(checkpoint(ck)), "hash_after_observing": state_hash(a_state),
                             "ledger_rows_visible": len(Obs(w, a_state, arm).ledger()), "note": arm.note,
                             "leak_check": leak_report(Obs(w, checkpoint(ck), arm)),
                             "actions": {k: ref.key_dict(ref.action_key(a)) for k, a in acts.items()}, "_acts": acts}
    ev = ref.evaluate_fork(w, ck, b_hist, extra_actions=all_actions, n_futures=N_FUTURES, seed=sc.seed * 100 + fd)
    best_none = ref.select_best(w, ck, b_none, n_futures=N_FUTURES, seed=sc.seed * 100 + fd)
    voh = ev.score(best_none)
    # classification on an independent future set (not the scoring futures), equivalence-style
    voh_test = ref.pair_test(w, ck, b_hist, ev.best, best_none, n_futures=VOH_FUTURES, seed=sc.seed * 1000 + fd)
    rp = ref.realized_path_values(w, ck, b_hist, [ref.key_actions(k) for k in ev.grid_keys])
    for arm in arms:
        rec = per_arm[arm.name]
        rec["scores"] = {k: ev.score(ref.action_key(a)) for k, a in rec.pop("_acts").items()}
    gains = {k: {"regret_reduction_c": per_arm["fresh"]["scores"][k]["regret_c"] - per_arm["history"]["scores"][k]["regret_c"],
                 "normalized_regret_reduction": round(per_arm["fresh"]["scores"][k]["normalized_regret"]
                                                      - per_arm["history"]["scores"][k]["normalized_regret"], 4)}
             for k in POLICIES}
    v1_spoil = [e for e in ck.events if e.type == "spoilage_writeoff" and e.public["vendor"] == "V1"]
    short_days = sum(1 for b in ck.books if b["short"] > 0)
    return {
        "seed": sc.seed, "family": sc.family, "fork_day": fd, "history_generator": spec["history"],
        "true_latents_evaluator_only": {"v1_decay_scale": ck.exo.v1_decay_scale, "goodwill_drop": ck.exo.goodwill_drop,
                                        "regime_at_fork": ck.exo.state[fd]},
        "checkpoint_hash": h0,
        "revealed_before_fork": {"v1_writeoff_cases": sum(e.public["cases"] for e in v1_spoil), "shortfall_days": short_days},
        "beliefs_history": b_hist.summary(p), "beliefs_fresh": b_none.summary(p),
        "reference": {"best_with_history": ref.key_dict(ev.best), "best_with_fresh_information": ref.key_dict(best_none),
                      "value_of_history_c": voh["regret_c"], "value_of_history_se_c": voh["regret_se_c"],
                      "voh_independent_test": voh_test, "fork_class": voh_test["label"],
                      "tolerance_c": round(ev.tolerance_c), "span_c": round(ev.span_c),
                      "null_control": voh_test["label"] != "informative",
                      "null_control_by_fork_tolerance": voh["regret_c"] <= ev.tolerance_c,
                      "best_is_boundary": ev.best_is_boundary(w), "grid_top": ev.table(),
                      "n_futures": N_FUTURES, "n_grid": len(ev.grid_keys),
                      "realized_path_best": ref.key_dict(max(rp, key=rp.get)),
                      "realized_path_gap_of_reference_c": max(rp.values()) - rp[ev.best]},
        "arms": per_arm, "history_gain": gains,
    }


def _cluster_bootstrap(forks: list[dict], key, n: int = 4000, seed: int = 0) -> list[float]:
    """Resample worlds (seeds), keeping each world's forks together."""
    by_world: dict[int, list[float]] = {}
    for f in forks:
        by_world.setdefault(f["seed"], []).append(key(f))
    worlds = list(by_world.values())
    if not worlds:
        return [None, None]
    rnd = random.Random(seed)
    ms = []
    for _ in range(n):
        sample = [x for _ in worlds for x in rnd.choice(worlds)]
        ms.append(mean(sample))
    ms.sort()
    return [round(ms[int(0.025 * n)], 2), round(ms[int(0.975 * n)], 2)]


def aggregate(forks: list[dict]) -> dict:
    out = {}
    strata = {"all": forks}
    for label in ("informative", "null", "undetermined"):
        strata[label] = [f for f in forks if f["reference"]["fork_class"] == label]
    for name, fs in strata.items():
        row = {"n_forks": len(fs), "n_worlds": len({f["seed"] for f in fs})}
        for k in POLICIES:
            g = [f["history_gain"][k]["regret_reduction_c"] for f in fs]
            row[k] = {"mean_regret_reduction_c": round(mean(g)) if g else None,
                      "cluster_bootstrap95_c": _cluster_bootstrap(fs, lambda f, k=k: f["history_gain"][k]["regret_reduction_c"]),
                      "forks_with_nonzero_gain": int(sum(x != 0 for x in g)),
                      "history_arm_acceptable_share": round(mean(f["arms"]["history"]["scores"][k]["synthetic_acceptable"] for f in fs), 3) if fs else None}
        out[name] = row
    by_family = {}
    for fam in ("calm", "sampled", "shock"):
        fs = [f for f in forks if f["family"] == fam]
        by_family[fam] = {"n": len(fs), "boundary_best_share": round(mean(f["reference"]["best_is_boundary"] for f in fs), 3),
                          "class_counts": {c: sum(f["reference"]["fork_class"] == c for f in fs) for c in ("informative", "null", "undetermined")},
                          "null_control_share_by_fork_tolerance": round(mean(f["reference"]["null_control_by_fork_tolerance"] for f in fs), 3),
                          "median_value_of_history_c": median(f["reference"]["value_of_history_c"] for f in fs),
                          "median_span_c": median(f["reference"]["span_c"] for f in fs),
                          "median_tolerance_c": median(f["reference"]["tolerance_c"] for f in fs)}
    notes = sum(1 for f in forks if f["arms"]["attribution_note"]["note"])
    by_v1 = {}
    for v1 in sorted({f["true_latents_evaluator_only"]["v1_decay_scale"] for f in forks}):
        fs = [f for f in forks if f["true_latents_evaluator_only"]["v1_decay_scale"] == v1]
        by_v1[str(v1)] = {"n": len(fs), "class_counts": {c: sum(f["reference"]["fork_class"] == c for f in fs) for c in ("informative", "null", "undetermined")},
                          "mean_value_of_history_c": round(mean(f["reference"]["value_of_history_c"] for f in fs))}
    return {"by_stratum": out, "by_family": by_family, "by_v1_latent": by_v1, "forks_with_attribution_note": notes,
            "all_arms_same_start_hash": all(len({a["start_hash"] for a in f["arms"].values()}) == 1 and
                                            all(a["start_hash"] == a["hash_after_observing"] for a in f["arms"].values())
                                            for f in forks),
            "all_leak_checks_ok": all(a["leak_check"]["ok"] for f in forks for a in f["arms"].values())}


def main() -> int:
    t0 = time.time()
    w = World(WorldParams())
    episodes = episode_runs(w)
    forks = [run_fork(w, spec) for spec in fork_specs()]
    agg = aggregate(forks)
    report = {
        "world_version": "v0.3",
        "params_evidence": {k: {"value": v.value, "status": v.status, "evidence": v.evidence} for k, v in w.p.evidence().items()},
        "identities": IDENTITIES, "identity_tolerance": TOLERANCE,
        "episodes": episodes, "forks": forks, "aggregate": agg,
        "runtime_s": round(time.time() - t0, 1),
        "notes": ["deterministic policies only; no language model was run",
                  "pre-fork histories come from labelled generators, not the evaluated policy",
                  "deterministic policies do not read notes: note arms are verified for construction, not effect",
                  "scores are synthetic-reference scores, not operator agreement",
                  "history gain = regret(fresh) - regret(history), same frozen grid and evaluation futures; positive = history helped",
                  "realized-path values are a diagnostic under the heuristic continuation, not an upper bound"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fixture_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({"aggregate": agg, "runtime_s": report["runtime_s"]}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
