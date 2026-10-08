# World specification (v0.3)

Technical reference for the simulator in `src/wsb/world/`. Results are in [`findings.md`](findings.md).

Status labels used for every parameter (see `WorldParams.evidence()` in `params.py`):
- **observed:** read from a source row;
- **derived:** computed from source data;
- **assumed:** an engineering default needing operator validation;
- **generated:** a hidden latent drawn per world.

## 1. Modules

| Module | Responsibility |
|---|---|
| `params.py` | `WorldParams` (all constants, with evidence) and `Scenario` (seed, market family, optional pinned latents) |
| `engine.py` | `World.reset/begin_day/apply/run`; records `Lot`, `PurchaseOrder`, `OpenItem` (AR/AP), `Event`; exogenous generation; `check_day` identities; `checkpoint`, `state_hash`, `summarize` |
| `observe.py` | `Obs` tools (`state`, `market`, `ledger`, `note`), `ArmConfig`, `fork_arms`, attribution and placebo notes, `leak_report`, opaque id `alias` |
| `beliefs.py` | posteriors computed only from an `Obs.dump()`: regime filter, grower decay, customer sensitivity |
| `policies.py` | shared planner `plan()`; `do_nothing`, `rule_base_stock` (history-blind), `adaptive` (history-aware); history generators `NoisyRule`, `ScriptedOverbuy` |
| `reference.py` | candidate grid, posterior-predictive futures, `evaluate_fork`, `select_best`, `pair_test` (VOH classification) |
| `demo.py` | fixture: episodes plus forks → `reports/fixture/fixture_report.json` |
| `rng.py` | event-keyed random streams: `stream(*key)` |

## 2. Entities and parameters

| Item | Value | Status | Evidence |
|---|---|---|---|
| Unit ("case") | 1 1/9 bushel carton of eggplant | observed | most frequent eggplant package in USDA terminal quotes |
| Calendar | 35 episode days (day 0 = Monday) + 21-day settlement tail | assumed | design choice |
| Grower V1 | delivers Mon/Thu; orders in 48-case pallets, ≤ 6 per delivery; price 0.90 × reference; 10-day terms; product 3 days old at receipt | assumed | terms mirror the common US produce prompt-payment window |
| Terminal V2 | same day; 10–50 cases available (30–90 in a glut); 1.05 × reference; cash; product 1–3 days old (shown in the offer) | assumed | — |
| Lot aging | per-case survival S(a) = exp(−(a/scale)^4), a = days since harvest; daily write-offs ~ Binomial(cases, hazard) | assumed | no lot-aging data in any public source |
| Grower decay scale | {7.5 poor, 12.0 good}, prior 0.35/0.65 | **generated, hidden** | from day 3 to day 6: ~68% vs ~94% of cases survive |
| Lot frailty | scale × LogNormal(0, 0.2) per lot; V2 lots draw their own base scale | generated, hidden | one bad lot is weak evidence |
| Customers | C1 restaurant group 16, C2 grocer 12, C3 caterer 8 average cases/day; weekday patterns (mean 1) | assumed; grocer pattern derived | grocer pattern from FreshRetailNet weekday multipliers |
| Order noise | LogNormal, σ = 0.30 | assumed | below FreshRetailNet's median daily CV of 0.485 |
| Goodwill | orders × (1 − 0.5·(1 − g)); after a shorted day g −= sensitivity × short share; +0.05/day recovery | assumed | — |
| Customer sensitivity | {0.2, 0.7} per customer, prior 0.5/0.5 | **generated, hidden** | — |
| Market reference | log price mean-reverts (κ = 0.4, σ = 0.04) to a regime level; normal $25.00; glut depth {0.35, 0.55, 0.75} log points; P(normal→glut) = 0.04/day, P(glut→normal) = 0.08/day | normal level derived; rest assumed | USDA Jan 2024 median $25; glut levels inside USDA historical medians |
| Market families | sampled (unconditioned); calm (no glut days 0–8); shock (normal on day 0, glut onset days 3–8) | design choice | conditioning stops before the first fork, so the reference's model is exact afterwards |
| Sell price | 1.10 × same-day reference | assumed | not an agent decision |
| Costs and credit | holding 10¢/case/day; disposal 50¢/case; overdraft 0.05%/day to a $25,000 line, 0.3%/day beyond | assumed | — |
| Opening position | cash $12,000; 60 cases (V1, received day −1); AR $9,000 and AP $4,000 spread over terms | assumed | — |
| Terminal value | cash + AR − AP + unsold stock × final V1 offer × prior 3-day survival | assumed | — |

## 3. Day loop (`engine.py`)

**`begin_day(d)`**
1. Publish the sell price.
2. Receive V1 deliveries due today, creating a lot and a vendor bill.
3. Age every lot received before today: binomial write-offs plus disposal cost.
4. Reveal today's customer orders (scaled by hidden goodwill) and today's V2 offer.

**`apply(d, actions)`**
1. Validate the actions.
2. Place V1 orders for the next delivery day. V2 buys are received and billed now (due today).
3. Ship FIFO, by default in priority C1 > C2 > C3, or as given by `Allocate`.
4. Update goodwill, then invoice.
5. Collect receivables due and pay payables due.
6. Charge holding and interest, close the day's book and check the identities.

**Tail (d ≥ 35):** settlement only. No orders, aging, holding or buying. Deliveries already ordered still arrive and are billed.

**Actions** (typed, in `engine.py`):
- `Buy(vendor, cases)`.
- `Allocate(cases=((customer, n), ...))`.

Validation raises `ActionError` for:
- non-pallet V1 quantities;
- more than 6 pallets per delivery;
- a V1 delivery that would fall in the tail;
- V2 above today's availability, or beyond the credit line (V2 is cash);
- any purchase in the tail;
- an allocation above an order or above stock.

**Exogenous** (keyed by seed, stream, entity and day; identical across branches): regime and price path, order noise, V2 availability and age.

**Action-dependent:** inventory, aging of what was bought (decay draws are keyed by vendor, order day and order index, so the same purchase gets the same draws in any branch), goodwill, AR, AP, cash.

## 4. Accounting identities

`check_day` asserts these every day, with zero tolerance (integer cents and cases):

```
cases_close   = cases_open + received − shipped − spoiled
invcost_close = invcost_open + received_cost − COGS − spoilage_at_cost          (FIFO by lot)
cash_close    = cash_open + collected − paid − holding − disposal − interest
AR_close      = AR_open + invoiced − collected
AP_close      = AP_open + billed − paid
net_income    = revenue − COGS − spoilage_at_cost − holding − disposal − interest
Δ(cash + AR + inventory_at_cost − AP) = net_income
shipped + short = ordered
```

Revenue and COGS are recognized at shipment. Spoilage is recognized at lot cost on the write-off day. After the tail, AR = AP = 0 (asserted in tests).

## 5. Agent tools and the information rule (`observe.py`)

| Tool | Returns |
|---|---|
| `state()` | day, weekday; cash, credit limit; receivables and payables as open total + total due in the next 7 days; inventory lots (opaque id, vendor, received day, days in stock, age at receipt, cases, unit cost); open POs with arrival day; today's orders; public customer profiles; today's vendor offers |
| `market(n)` | reference price for the last n days, up to today |
| `ledger()` | the agent's own public event records, rows numbered from 1 |
| `note()` | the arm's note, or none |

**Ledger event types:** `opening_stock`, `price_list`, `receipt`, `vendor_bill`, `spoilage_writeoff`, `customer_order`, `purchase_order`, `allocation`, `customer_invoice`, `shipment`, `customer_payment`, `vendor_payment`, `interest_charge`.

**Never exposed:**
- future prices, orders and offers;
- decay scales and frailties;
- goodwill and sensitivity;
- regime;
- decision ids and causal annotations (`Event.hidden`);
- random-stream keys;
- sequential internal ids (shown as HMAC aliases; set the `WSB_ALIAS_PEPPER` env var in any deployment).

**Books are totals, not schedules.** Itemized or per-day books would let the fresh arm rebuild past shipments and purchases.

**Enforcement:**
- `leak_report` checks for forbidden keys, future-dated rows and internal ids; tests cover days 0, 5, 9, 16 and 34.
- An agent must receive only the JSON these tools return. The `Obs` object holds the full world state.

## 6. Fork protocol

1. `w.run(w.reset(scenario), generator, until=fork_day)`, then `checkpoint(state)`. Fork days are V1 cutoffs (Wednesdays 9 and 16).
2. `fork_arms(ck)` builds history, fresh (ledger watermark), attribution note and placebo note (within 10% of the note's length, names no decision).
   - Tests assert identical start hashes and that observing never mutates state.
3. `beliefs_from_dump(params, Obs(...).dump(), info_set)` gives the posterior for any arm from that arm's own tool output. One likelihood covers:
   - **Decay:** binomial over daily write-offs and non-events by lot age, with frailty integrated by 7-node Gauss–Hermite.
   - **Goodwill:** sensitivity hypotheses replayed through shipments, with interval likelihood for integer orders.
   - **Regime:** a 4-state forward filter on the log-price path.

   A test asserts the fresh posterior ignores pre-fork records.
4. `evaluate_fork(w, ck, b_hist, extra_actions, n_futures=64, seed)`:
   - **Frozen grid:** V1 pallets × {0, today's shortfall, all available V2}.
   - **Best:** chosen on selection futures.
   - **Scoring:** every action is scored on separate evaluation futures.
   - **Tolerance:** one per fork, max($25, 2 × median paired SE).

   **Continuation after the fork action:** the shared planner with fork-time vendor and customer beliefs, re-filtering the market on cutoff days. **Futures:** today's hidden quantities are re-drawn from their posteriors; every draw is keyed by (seed, future, kind, day[, entity]).
5. `select_best(w, ck, b_fresh)` returns the fresh-information reference's pick. `pair_test(..., n_futures=256)` classifies the fork as informative, null or undetermined on independent futures.

**Reference limits:** a one-step lookahead with a heuristic continuation, not a proven optimum. It knows the family's priors (an agent knows them only if its task text states them). `realized_path_values` is a diagnostic, not an upper bound.

## 7. Output schema (`fixture_report.json`)

**Top-level keys:** `world_version`, `params_evidence`, `identities`, `episodes[]`, `forks[]`, `aggregate`, `runtime_s`, `notes`.

**Each `forks[]` record:**

| Key | Contents |
|---|---|
| `seed`, `family`, `fork_day`, `history_generator`, `checkpoint_hash` | fork identity |
| `true_latents_evaluator_only` | the hidden values; never shown to agents |
| `beliefs_history`, `beliefs_fresh` | posterior summaries |
| `reference` | `best_with_history`, `best_with_fresh_information`, `value_of_history_c`, `voh_independent_test{mean_c, se_c, label}`, `fork_class`, `tolerance_c`, `span_c`, `best_is_boundary`, `grid_top` |
| `arms.<arm>` | `start_hash`, `hash_after_observing`, `leak_check`, `actions{policy}`, `scores{policy: {action, expected_value_c, regret_c, regret_se_c, normalized_regret, synthetic_acceptable}}` |
| `history_gain.<policy>` | `regret_reduction_c`, `normalized_regret_reduction` |

All money is in cents. Scores are synthetic-reference scores, not agreement with a human operator.

## 8. Invariants and their tests (`tests/test_world.py`)

| Invariant | Test |
|---|---|
| Identities hold daily; AR = AP = 0 after the tail; nothing happens in the tail | `test_identities_hold_every_day` |
| Market families honour their conditioning | `test_scenario_families_respect_their_conditions` |
| Determinism and checkpoint reset | `test_deterministic_replay_and_checkpoint_reset` |
| Exogenous paths independent of actions; same purchase gets same lot draws | `test_exogenous_paths_do_not_depend_on_actions`, `test_same_purchase_gets_same_lot_draws_across_branches` |
| Validation, credit line and allocation | `test_action_validation`, `test_credit_line_blocks_cash_purchases`, `test_allocation_overrides_priority_and_moves_goodwill` |
| Arms identical; no leaks; fresh view; rule policy history-blind | `test_arms_start_identical_and_observing_does_not_mutate`, `test_no_forbidden_information_in_any_arm`, `test_fresh_arm_view`, `test_rule_policy_is_history_blind` |
| Notes state past facts only; length-matched | `test_notes_are_past_facts_and_length_matched` |
| Posteriors use exactly what each arm can see | `test_fresh_beliefs_ignore_history_and_history_beliefs_use_it`, `test_fresh_beliefs_invariant_to_prefork_records`, `test_opening_stock_is_evidence_in_history_only` |
| Reference grid is frozen and seeded | `test_reference_protocol` |

## 9. How to extend

- **New parameter:** add it to `WorldParams` with a `Param(value, status, evidence)` entry in `evidence()`.
- **New action:**
  1. add a frozen dataclass in `engine.py`;
  2. validate it in `World.validate`;
  3. apply it in `World.apply`;
  4. emit public and hidden events;
  5. extend `check_day` if it moves money or stock;
  6. add it to `reference.grid` and `action_key`.
- **New hidden latent:**
  1. draw it in `Scenario.resolve` / `generate_exogenous`;
  2. add its key to `FORBIDDEN_KEYS`;
  3. add its likelihood to `beliefs.py` (fed only from `Obs.dump()`);
  4. re-draw it from the posterior in `reference.sample_future`;
  5. add an invariance test like `test_fresh_beliefs_invariant_to_prefork_records`.
- **New workflow** (e.g. short-ships and claims): reuse the lot, receipt and bill ledger. Add `ShortFill` at receipt, `Inspection`, and `Claim` → a delayed `CreditMemo` event that reduces AP. Give the fill or defect rate a hidden per-vendor latent.
- **Agent harness:**
  - wrap `Obs.state/market/ledger/note` and the two actions as JSON tools;
  - run each arm as a separate clean session from `checkpoint(ck)`;
  - score the agent's action with `ForkEvaluation.score(action_key(actions))`.
