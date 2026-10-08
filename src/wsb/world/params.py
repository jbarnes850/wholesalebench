"""World parameters with evidence status (v0.3).

Every parameter carries one status label:
  observed   - read directly from a source row
  derived    - computed from source data (artifact path given)
  assumed    - engineering default; needs partner validation
  generated  - drawn by the generator per world (latent, hidden from agents)

Nothing here is partner data. Dollar values are synthetic; ranges cite the
public evidence that bounds them.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from wsb.lineage import ROOT

CAL = ROOT / "data" / "calibration"


def _cal(name: str) -> dict:
    return json.loads((CAL / name).read_text())


@dataclass(frozen=True)
class Param:
    value: object
    status: str
    evidence: str


def _mean1(xs) -> tuple[float, ...]:
    m = sum(xs) / len(xs)
    return tuple(round(x / m, 4) for x in xs)


def _grocer_weekday_mon0() -> tuple[float, ...]:
    sun0 = _cal("freshretail_demand.json")["calibration_train"]["dow_multiplier_sun0_duckdb_dayofweek"]
    return _mean1([sun0[(i + 1) % 7] for i in range(7)])


@dataclass(frozen=True)
class WorldParams:
    # --- calendar (day 0 is a Monday)
    episode_days: int = 35
    settlement_tail_days: int = 21  # settlement only: no orders, no aging, no holding cost
    pre_episode_market_days: int = 28

    # --- product: one SKU, a 1 1/9 bushel carton of eggplant ("case")
    # --- customers: average daily cases (public) x weekday pattern (mean 1, public) x lognormal noise
    customer_ids: tuple[str, ...] = ("C1", "C2", "C3")
    customer_kind: tuple[str, ...] = ("restaurant_group", "independent_grocer", "caterer")
    customer_base_cases: tuple[float, ...] = (16.0, 12.0, 8.0)
    customer_weekday: tuple[tuple[float, ...], ...] = field(default_factory=lambda: (
        _mean1((0.8, 0.8, 0.9, 1.2, 1.4, 1.3, 0.6)),
        _grocer_weekday_mon0(),
        _mean1((1.0, 0.0, 1.6, 0.0, 2.0, 0.0, 0.0)),
    ))
    customer_terms_days: int = 14
    order_log_sigma: float = 0.30
    # service response (ASSUMED): goodwill g in [0,1]; orders x (1 - goodwill_weight * (1 - g));
    # after a shorted day g -= drop_c * shortfall_fraction; g += recovery per day. drop_c is a latent per customer.
    goodwill_drop_support: tuple[float, ...] = (0.2, 0.7)
    goodwill_drop_prior: tuple[float, ...] = (0.5, 0.5)
    goodwill_recovery: float = 0.05
    goodwill_weight: float = 0.5

    # --- market reference (USD/case): log price mean-reverts toward a regime level.
    # States: 0 normal, k>=1 glut with depth glut_depth_support[k-1] (log points below normal).
    normal_price_c: int = 2500
    glut_depth_support: tuple[float, ...] = (0.35, 0.55, 0.75)
    price_reversion: float = 0.4
    price_log_sd: float = 0.04
    p_normal_to_glut: float = 0.04
    p_glut_to_normal: float = 0.08

    # --- sell price list: markup on the SAME day's reference; exogenous to the agent
    markup: float = 1.10

    # --- vendors: V1 grower (scheduled pallets, discount, 10-day terms); V2 terminal spot
    vendor_ids: tuple[str, ...] = ("V1", "V2")
    vendor_kind: tuple[str, ...] = ("grower", "terminal_spot")
    vendor_spread: tuple[float, ...] = (0.90, 1.05)
    v1_delivery_weekdays: tuple[int, ...] = (0, 3)  # Mon, Thu
    v1_pallet_cases: int = 48
    v1_max_pallets: int = 6
    v1_receipt_age: int = 3  # distant grower: product reaches the dock 3 days after harvest
    v2_avail_normal: tuple[int, int] = (10, 50)
    v2_avail_glut: tuple[int, int] = (30, 90)
    v2_receipt_age_range: tuple[int, int] = (1, 3)  # varies by day; shown in the offer
    vendor_terms_days: tuple[int, ...] = (10, 0)

    # --- lot aging (ASSUMED family): per-case survival S(a) = exp(-(a/scale)^shape).
    # V1 has a persistent latent scale; every lot (V1 and V2) also has a frailty multiplier.
    # V2 lots come from varied shippers: each lot's base scale is drawn from the same support.
    decay_shape: float = 4.0
    decay_scale_support: tuple[float, ...] = (7.5, 12.0)
    decay_scale_prior: tuple[float, ...] = (0.35, 0.65)
    lot_frailty_sigma: float = 0.2

    # --- costs and credit
    holding_c_per_case_day: int = 10
    disposal_c_per_case: int = 50
    overdraft_daily_rate: float = 0.0005
    credit_limit_c: int = 2_500_000
    over_limit_daily_rate: float = 0.003

    # --- opening position
    opening_cash_c: int = 1_200_000
    opening_inventory_cases: int = 60
    opening_ar_c: int = 900_000
    opening_ap_c: int = 400_000

    # --- terminal valuation of unsold stock: replacement cost x expected 3-day survival (prior mixture)
    nrv_horizon_days: int = 3

    @property
    def n_states(self) -> int:
        return 1 + len(self.glut_depth_support)

    def state_mean_log(self, s: int) -> float:
        base = math.log(self.normal_price_c)
        return base if s == 0 else base - self.glut_depth_support[s - 1]

    def transition(self) -> list[list[float]]:
        k = len(self.glut_depth_support)
        T = [[0.0] * self.n_states for _ in range(self.n_states)]
        T[0][0] = 1 - self.p_normal_to_glut
        for j in range(1, self.n_states):
            T[0][j] = self.p_normal_to_glut / k
            T[j][j] = 1 - self.p_glut_to_normal
            T[j][0] = self.p_glut_to_normal
        return T

    def stationary(self) -> list[float]:
        a, b, k = self.p_normal_to_glut, self.p_glut_to_normal, len(self.glut_depth_support)
        p0 = b / (a + b)
        return [p0] + [(1 - p0) / k] * k

    def survival(self, age: float, scale: float) -> float:
        return math.exp(-((max(age, 0.0) / scale) ** self.decay_shape))

    def hazard(self, age: int, scale: float) -> float:
        """Probability that a case saleable at age-1 is unsaleable at age."""
        s0, s1 = self.survival(age - 1, scale), self.survival(age, scale)
        return 0.0 if s0 <= 0 else 1.0 - s1 / s0

    def frailty_nodes(self, n: int = 7) -> tuple[list[float], list[float]]:
        """Gauss-Hermite nodes and weights for a LogNormal(0, sigma) multiplier."""
        import numpy as np

        x, w = np.polynomial.hermite_e.hermegauss(n)
        w = w / w.sum()
        return [math.exp(self.lot_frailty_sigma * xi) for xi in x], list(w)

    def prior_survival_ratio(self, a0: int, a1: int) -> float:
        out = 0.0
        for p, sc in zip(self.decay_scale_prior, self.decay_scale_support):
            s0 = self.survival(a0, sc)
            out += p * (self.survival(a1, sc) / s0 if s0 > 0 else 0.0)
        return out

    def is_v1_delivery_day(self, d: int) -> bool:
        return d % 7 in self.v1_delivery_weekdays

    def next_v1_delivery(self, d: int) -> int:
        x = d + 1
        while not self.is_v1_delivery_day(x):
            x += 1
        return x

    def evidence(self) -> dict[str, Param]:
        frn = _cal("freshretail_demand.json")["calibration_train"]
        egg = _cal("usda_eggplant.json")
        return {
            "episode_days": Param(self.episode_days, "assumed", "4-8 week design target; fixture uses 5 weeks"),
            "settlement_tail_days": Param(self.settlement_tail_days, "assumed", "settlement only (no orders, aging or holding) so terminal value counts settled positions"),
            "case_definition": Param("1 1/9 bushel carton", "observed", "most frequent eggplant package in USDA AMS terminal quotes (data/clean/usda_ams/terminal_quotes.parquet)"),
            "customer_base_cases": Param(self.customer_base_cases, "assumed", "public average daily cases; scale not recoverable from normalized FreshRetailNet sales"),
            "customer_weekday_grocer": Param(self.customer_weekday[1], "derived", "FreshRetailNet train weekday multipliers re-indexed Mon=0, normalized to mean 1 (retail-to-wholesale transfer assumed)"),
            "customer_weekday_restaurant_caterer": Param((self.customer_weekday[0], self.customer_weekday[2]), "assumed", "no restaurant/caterer ordering data in any source"),
            "order_log_sigma": Param(self.order_log_sigma, "assumed", f"below FreshRetailNet median uncensored daily CV {frn['daily_cv_uncensored_q10_50_90'][1]} (which also contains weekday variation)"),
            "goodwill_response": Param((self.goodwill_drop_support, self.goodwill_recovery, self.goodwill_weight), "assumed", "action-dependent response; per-customer sensitivity is a generated latent; sensitivity analysis required"),
            "normal_price_c": Param(self.normal_price_c, "derived", f"USDA eggplant 1 1/9 bu Jan 2024 cross-section median mid ${egg['jan_2024_cross_section_mid_usd_q10_50_90'][1]} (data/calibration/usda_eggplant.json)"),
            "glut_depth_support": Param(self.glut_depth_support, "assumed", f"glut levels ${self.normal_price_c * math.exp(-max(self.glut_depth_support)) / 100:.2f}-${self.normal_price_c * math.exp(-min(self.glut_depth_support)) / 100:.2f}, inside USDA report-date medians q10-q90 ${egg['report_date_median_mid_usd_q10_50_90'][0]}-${egg['report_date_median_mid_usd_q10_50_90'][2]} (nominal, 1999-2024)"),
            "price_dynamics": Param((self.price_reversion, self.price_log_sd, self.p_normal_to_glut, self.p_glut_to_normal), "assumed", "no keyless daily series; gluts ramp over ~3 days and last ~12 days on average"),
            "markup": Param(self.markup, "assumed", "exogenous same-day list = markup x terminal reference (sell prices are not an agent decision)"),
            "vendor_spread": Param(self.vendor_spread, "assumed", "grower discount / terminal premium relative to reference; V1 gross margin ~18% at stable prices"),
            "v1_schedule_pallets": Param((self.v1_delivery_weekdays, self.v1_pallet_cases, self.v1_max_pallets), "assumed", "needs operator validation: case packs, delivery schedule"),
            "v2_availability_and_age": Param((self.v2_avail_normal, self.v2_avail_glut, self.v2_receipt_age_range), "assumed", "same-day terminal supply is limited and of varied age"),
            "vendor_terms_days": Param(self.vendor_terms_days, "assumed", "V1 10 days mirrors the PACA default prompt-payment window (applicability assumed); V2 cash on purchase"),
            "decay": Param((self.decay_shape, self.decay_scale_support, self.lot_frailty_sigma), "assumed", "Weibull survival by age; V1 latent scale; per-lot frailty; no lot-aging data in any source. A lot received at age 3 keeps ~68% (poor) vs ~94% (good) of cases to age 6 (before lot frailty)"),
            "v1_receipt_age": Param(self.v1_receipt_age, "assumed", "distant grower; needs operator validation. With age 1 at receipt, vendor quality barely changed any decision (all 40 v0.3 forks were null controls)"),
            "quality_discount_reference": Param(egg.get("lower_quality_to_standard_ratio_same_date_median"), "derived", "USDA same-date lower-quality/standard price ratio (not yet used; seed-4 inspection/claims)"),
            "costs_credit": Param((self.holding_c_per_case_day, self.disposal_c_per_case, self.overdraft_daily_rate, self.credit_limit_c, self.over_limit_daily_rate), "assumed", "explicit penalties; sensitivity required"),
            "opening_position": Param((self.opening_cash_c, self.opening_inventory_cases, self.opening_ar_c, self.opening_ap_c), "assumed", "synthetic opening balances"),
            "terminal_nrv": Param(self.nrv_horizon_days, "assumed", "unsold stock at episode end valued at final V1 offer x prior-mixture survival over 3 days"),
        }


@dataclass(frozen=True)
class Scenario:
    """A generated world: seed, scenario family and optional fixed latents.

    Families draw the regime path from the SAME Markov model the reference knows:
      sampled: unconditioned;
      calm: rejection-sampled to have no glut through onset_window[1] (before the first fork day);
      shock: rejection-sampled to be normal on day 0 and start a glut between onset_window days.
    Conditioning touches only days before the first fork (day 9), so after any fork the
    Markov model the reference knows is the true conditional law.
    """

    seed: int
    family: str = "sampled"
    onset_window: tuple[int, int] = (3, 8)
    v1_decay_scale: float | None = None
    goodwill_drop: tuple[float, ...] | None = None

    def resolve(self, p: WorldParams) -> "Scenario":
        from wsb.world.rng import stream

        dk = self.v1_decay_scale
        if dk is None:
            dk = float(stream(self.seed, "latent", "v1_decay").choice(p.decay_scale_support, p=p.decay_scale_prior))
        gd = self.goodwill_drop
        if gd is None:
            gd = tuple(float(stream(self.seed, "latent", "goodwill", c).choice(p.goodwill_drop_support, p=p.goodwill_drop_prior))
                       for c in p.customer_ids)
        return Scenario(self.seed, self.family, self.onset_window, dk, gd)


def lognormal_mu(sigma: float) -> float:
    """Location giving a lognormal with mean 1."""
    return -0.5 * sigma * sigma
