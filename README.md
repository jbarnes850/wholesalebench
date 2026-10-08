# WholesaleBench

A synthetic produce-wholesale buying desk for testing whether AI agents learn from the delayed consequences of their own decisions.

The agent runs one desk for five weeks. It orders pallets from a cheaper grower whose lots may spoil fast, tops up same-day at a pricier terminal market, and decides who gets shorted when stock runs out. Spoilage, lost sales and customer drift arrive days later. Evaluation re-runs the same frozen decision with and without the agent's own history and scores each choice in dollars against a reference policy.

**Status (v0.3, 2026-10-08):** the simulator, grading protocol, data pipeline and deterministic baselines work. All 42 tests pass. No language model has been run yet. Results and limits: [`docs/findings.md`](docs/findings.md).

## Read next

| File | Read it for |
|---|---|
| [`docs/findings.md`](docs/findings.md) | What was built, how agents are scored, results, what the public data can and can't support, open issues, next steps |
| [`docs/world_spec.md`](docs/world_spec.md) | Exact mechanics: state, actions, tools, information rule, accounting identities, latents, fork protocol, reference policy, output schema, how to extend |

## Quickstart

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                   # install pinned dependencies
uv run pytest -q                          # 42 tests, ~5 s; data-dependent checks skip if data is absent
uv run python -m wsb.world.demo           # full fixture: 40 forks + 44 episodes, ~16 min on one core
uv run python -m wsb.pipeline             # optional: download and rebuild all data artifacts (~0.4 GB)
uv run python -m wsb.pipeline --wayback   # also fetch 119 archived USDA reports (rate-limited, ~10+ min)
```

The simulator and tests need only the committed calibration files in `data/calibration/`. Raw and cleaned data are not in git; `wsb.pipeline` rebuilds them from the public sources, checksums every raw file, and records any source that is unavailable instead of failing.

## Layout

| Path | Contents |
|---|---|
| `src/wsb/world/` | simulator: `params.py` (every parameter with its evidence status), `engine.py` (event-sourced day loop, exact accounting), `observe.py` (agent tools, fork arms, notes), `beliefs.py` (posteriors from agent-visible records), `policies.py` (baselines), `reference.py` (scoring), `demo.py` (fixture runner), `rng.py` |
| `src/wsb/sources/` | per-source staging, cleaning, profiling, calibration: `usda_ams.py`, `freshretail.py`, `uci_retail.py`, `wwi_schema.py` |
| `src/wsb/` | `acquire.py` (downloads with checksums), `pipeline.py` (one-command rebuild), `register.py` (source register), `lineage.py` (transformation logs) |
| `tests/` | world invariants, information rule, fork arms, reference protocol, parser and pipeline checks |
| `data/calibration/*.json` | derived statistics with provenance (committed) |
| `registry/` | `sources.yaml` (metadata, licenses), `raw_manifest.jsonl` (per-file sha256), `source_register.csv` |
| `reports/fixture/fixture_report.json` | latest fixture output (40 forks, 44 episodes) |
| `reports/<source>/` | profiles and transformation logs for each source |

## Data sources and attribution

| Source | License | Used for |
|---|---|---|
| [USDA AMS Specialty Crops Market News](https://www.ams.usda.gov/market-news/fruits-vegetables) terminal-market reports, including [Internet Archive](https://web.archive.org/) captures | US government work | produce package units, price levels, seasonal range, quality discount |
| [FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) (Dingdong Inc., arXiv:2505.16319) | CC BY 4.0 | weekday demand pattern, order variability, stockout semantics |
| [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (D. Chen, doi:10.24432/C5CG6D) | CC BY 4.0 | ledger-structure analog: cancellations, write-offs, bad-debt entries |
| [Wide World Importers](https://github.com/microsoft/sql-server-samples/releases/tag/wide-world-importers-v1.0) (Microsoft) | MIT | relational schema reference only; rows not inspected |
| [BPI Challenge 2019](https://data.4tu.nl/articles/dataset/BPI_Challenge_2019/12715853) (4TU.ResearchData) | CC BY 4.0 | planned; download blocked by publisher maintenance on 2026-10-07 |

Derived files in `data/calibration/` and `reports/` contain statistics computed from FreshRetailNet-50K and UCI Online Retail II under CC BY 4.0. Credit those sources when reusing them.

## License

No license has been chosen for this repository's code yet. Until one is added, the default copyright terms apply.
