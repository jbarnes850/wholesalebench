# Findings (v0.3, 2026-10-08)

**Summary**
- A fully synthetic produce buying desk, built from public data, with exact accounting and decision-level grading, works end to end.
- In 40 generated decision points, 8 reward using the agent's own history (worth $29–$152 each), 30 are null controls where changing course is wrong, and 2 are undetermined.
- The grading instrument is self-consistent: a history-blind policy shows exactly $0 history gain, and no information leaks or exploits were found.
- No language model has been run. Scripted histories and deterministic baselines show the instrument works. They say nothing yet about model behavior.

Every number below comes from `reports/fixture/fixture_report.json` unless another file is named. Mechanics are in [`world_spec.md`](world_spec.md).

## 1. Research question

When a consequence of an agent's decision arrives days later, does the agent connect it to that decision and decide better at the next related decision?

Four things are measured separately:
- **Constraint validity:** does every action pass validation, and do the books balance?
- **Decision quality:** dollar regret at a decision point.
- **Realized outcomes:** profit, waste and fill rate. These are diagnostics only, because one realized path can reward a bad decision or punish a good one.
- **History-access gain:** regret without history minus regret with history.

## 2. What the agent does

One workflow: buying perishable produce (eggplant, sold by the 1 1/9 bushel carton) for a wholesale desk.

| Decision | When | Choices | Delayed consequence |
|---|---|---|---|
| Grower order (scored at forks) | Sunday and Wednesday, for Monday and Thursday delivery | 0–6 pallets of 48 cases, about 10% under market, 10-day terms; product arrives 3 days after harvest | write-offs 2–5 days later if lots age before selling; shortages if too few |
| Same-day terminal buy | any day | up to that day's available cases, about 5% over market, cash, 1–3 days old | covers today's gap at a thin margin |
| Shortage allocation | when stock can't cover orders | cases per customer (restaurant group, grocer, caterer) | shorted customers order less for days, by a hidden per-customer amount |

Sell prices follow the market; the agent does not set them.

**Hidden from the agent:** the grower's decay rate, each customer's sensitivity to being shorted, and the future. The grower's decay rate can only be learned from the agent's own write-off records.

**Not built:** upfront-payment discounts and cash shortfalls, customer credit terms and late payment, vendor short-ships and claims. These need payment and claims behavior that no public source provides.

## 3. How an agent is scored

1. **Fork.** Run a history generator to a grower-order cutoff (day 9 or 16) and freeze the full state.
2. **Four arms** from that identical state:

   | Arm | What the agent sees |
   |---|---|
   | history | its own ledger: purchases, receipts, write-offs, shipments, payments |
   | fresh | same inventory, cash and prices, ledger removed |
   | attribution note | history plus a note naming the earlier purchase behind the largest write-off |
   | placebo note | history plus a same-length note that names no decision |

3. **Regret.** A reference policy uses the history arm's information plus the world's priors (never the true hidden values). It picks the best action from a frozen candidate grid on one set of simulated futures. Every action, in every arm, is then scored on a separate set of futures as regret = E[value(best)] − E[value(action)], in cents. Regret can be negative.
4. **History gain** = regret(fresh) − regret(history), for the same policy at the same fork.
5. **Value of history (VOH).** The regret of the action a reference with only fresh-arm information would pick, tested on an independent 256-future set:
   - **informative:** lower 95% bound > 0 and mean > $25;
   - **null:** upper bound < $25;
   - **undetermined:** otherwise.

   Report results per class; never pool undetermined forks with nulls.

The two note arms separate "doesn't connect the loss to its cause" from "connects it but doesn't act". The comparison is suggestive, not proof, because a note also adds salience. Agent reasoning is logged, never scored.

## 4. Results

Run: 40 forks in 22 generated worlds (calm, sampled and shock market families), plus 44 full episodes; 983 s on one CPU core.

| Check | Result |
|---|---|
| Daily accounting identities (7 of them, integer cents and cases) | pass on every day of every run |
| Deterministic replay | 44/44 episodes replay to the same state hash |
| All four arms start from an identical state | 40/40 forks |
| No forbidden information in any arm | 160/160 arm checks |
| History-blind rule policy | history gain exactly $0 in 40/40 forks |
| Reference best at the edge of the grid (a speculation symptom) | 0% in every family |
| Fork classes | 8 informative, 30 null, 2 undetermined |
| Informative lessons | 6 × "this grower's lots decay fast: order one pallet less"; 2 × "this grower is good: order one more" |
| VOH on informative forks | $28.52–$151.98 (SE $8.78–$14.02); the median candidate-value span is $1,556 |
| History-aware adaptive baseline, history gain | informative +$11.07 per fork (95% world-cluster bootstrap −$8.23 to +$30.76, n=8); null +$9.48 (−$1.51 to +$22.90); undetermined −$12.88 (n=2). It over-corrected once ($35.21 regret with history vs $0 fresh) |
| Episode medians, history-blind rule | terminal value $22,882, fill 97.4%, waste 2.1%, gross margin 15.9% (range 14.8–19.6% over 9 worlds) |
| Episode medians, do-nothing | terminal value $18,525, fill 7.7% |

**What this means**
- The world produces decisions where a fact visible only in the agent's own records changes the right action, in both directions. It also produces null controls where changing course is the mistake.
- The signal is small next to the stakes, and only 20% of natural decision points are informative. A benchmark needs on the order of 150+ forks to get 30 informative ones, and must classify forks on independent futures.
- Whether history matters depends on mechanics, not just on a hidden variable existing. In an earlier configuration where grower product arrived 1 day after harvest instead of 3 (decay scales 9.5/12), all 40 forks were null controls: stock sold before decay mattered. To explore it, build the world as `World(dataclasses.replace(WorldParams(), v1_receipt_age=1))`.

## 5. What the public data supports

| Source | What we took | Checks | Limit |
|---|---|---|---|
| USDA AMS terminal reports: 40 legacy reports from 10 markets (Jan 2024) + 119 Boston captures (1999–2023) | the case unit (1 1/9 bushel carton); normal price $25 (Jan 2024 median); glut levels within the historical range ($11–$24 report-date medians); lower-quality lots at 0.75× standard (114 dates) | 52,904 quotes, 98.6% yield, every price token accounted for. Three independent hand audits: strict precision 42.5% → 74.0% → 66.0% (95% CI 52–78%, stratified beyond Boston). Price fields 50/50 in every audit; eggplant fields 15/15 | offering quotes, not transactions or volumes; sparse snapshots, not a daily series; nominal prices across 25 years. A free [MARS API](https://mymarketnews.ams.usda.gov/) key would add daily history and shipment volumes |
| FreshRetailNet-50K: 50,000 store-product series, 90 + 7 days, hourly stock flags | grocer weekday pattern; demand variability (median daily CV 0.485) | hourly totals match daily; the stock-flag count matches slots 06:00–22:00 in 100% of rows; 2.9% of stockout hours still record sales (a censoring flag, not a hard zero) | retail, not wholesale; sales normalized by an undisclosed factor, so physical units are not recoverable |
| UCI Online Retail II: 1,067,371 invoice lines | structure of a messy ledger: 17,973 cancellations, 3,391 write-offs, 6 bad-debt entries; cancellation links (9,638 unique) | the two sheets overlap with identical lines (22,523 flagged); 11,812 exact repeats kept and flagged; no row dropped | giftware, not produce; statistics are structural only |
| Wide World Importers | schema reference: ordered vs received quantities, case packs, payment days, credit limits | model checksum verified | rows not inspected (SQL Server cannot run on this ARM host); its rows are generated anyway |

**What no public source gives:** how spoilage, vendor short-ships, claims and late payments respond to a buyer's decisions. Every action-dependent response in the world is therefore an explicit assumption, labelled in `src/wsb/world/params.py` (`WorldParams.evidence()`). Calibrating them needs real operator records that link purchases to receipts, write-offs, credits and payments.

## 6. Open issues

Independent reviews found the mechanics sound. They found the measurement not yet ready for a headline. Still open:

1. **Shared planner.** The reference continuation and the baselines use the same heuristic planner, so the rule policy is "acceptable" in 90% of forks. A stronger, distinct reference (rolling-horizon DP or MPC) is needed before comparing models.
2. **One latent, one lesson.** Informative forks all come from the grower-decay latent. The customer-sensitivity latent rarely bites (shortfall days are rare before forks). A second independent lesson would strengthen the benchmark, for example vendor short-ships.
3. **Fresh-arm goodwill.** The fresh-arm posterior assumes customers are fully happy. Today's orders carry weak evidence otherwise; true goodwill is below 0.9 in about 5% of customer-forks.
4. **Attribution is easy.** Write-off records carry lot ids, so the note names a purchase the agent could also look up. The note arms test salience, not hard attribution.
5. **Inert credit line.** No baseline touches it, so solvency is modelled but not measured.
6. **Monte Carlo noise.** Tolerance is set by noise (median about $74 at 64 futures); VOH classification uses 256 independent futures.
7. **Scripted histories.** Pre-fork histories come from scripted generators, so forks test use of records, not learning from one's own actions.
8. **Stockouts.** The rule policy has a stockout on 0–9% of episode days in the fixture. Plausibility needs operator input.

## 7. Next steps

1. **Close issues 1–4** above.
2. **Agent harness.** Expose the four tools as a JSON tool API. Run each arm as a separate clean-context session. Let the evaluated agent play the days before each fork itself. Log actions, observations, tool errors and short rationales.
3. **Scale and hold out.** Generate 150+ forks, report by class, hold out whole market families and parameter regimes, and use world-clustered confidence intervals.
4. **Add the remaining workflows** (vendor short-ships and claims, upfront-payment discounts, customer credit terms) once operator data can ground payment and claims behavior.
5. **Validate with an operator.** Collect a short list of assumption checks (case packs, receipt age, lot aging, fill, claims, allocation, payment practice), then acceptable-action labels on a sample of forks, then a replay check against real records.

## 8. Related work (bounded check, Oct 2026)

Summaries were read through a fetch tool. Verify against the papers before citing.

- **E-Commerce Bench** (arXiv:2608.30730) measures within-episode learning from an agent's own deals (repeat-order prices, re-dealing with fraudulent suppliers). It has no history ablation.
- **Business Arena** (arXiv:2608.08621) has save–fork–load checkpoints and action-level profit attribution. Its forks are used for trace search, without a memory ablation.
- **Bazaar** (arXiv:2608.00102) scores rounds against a hindsight oracle and measures recovery after shocks.
- **RetailBench** (arXiv:2603.16453) covers perishables, suppliers and cash at the episode level.
- **Vending-Bench** (arXiv:2502.15840) reports that larger memory performed worse.
- Beer-game studies with LLMs (e.g. arXiv:2605.17036) report that more information can worsen ordering.

**Not found in this check:** paired forks from an identical checkpoint that vary only access to the agent's own decision–outcome history; attribution vs length-matched placebo notes; decision-level posterior-predictive scoring.
