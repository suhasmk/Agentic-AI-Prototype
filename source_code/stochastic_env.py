"""
Stochastic Simulation Environment

Demand distributions supported:
    'poisson'  -- Poisson(lambda)           CV = 1/sqrt(lambda) ~ 0.26 at lambda=15
    'negbin'   -- NegativeBinomial(r=5)     CV ~ 0.52 at lambda=15 (2x Poisson)

Lead time distribution:
    Log-normal (same mean and variance as original Gaussian but right-skewed),
    reflecting the empirical observation that supply chain delays are
    asymmetric -- late arrivals are more common than early ones.
    Reference: Chopra & Meindl (2016), Supply Chain Management, S.12.

Disruptions:
    Bernoulli(p) onset with exponential duration. Lead time multiplied
    by DISRUPTION_LT_MULTIPLIER during active disruption events.

    supplier_reliability.py now modulates effective lead time variance.
    A low-reliability supplier (e.g. ASIA = 0.78) adds up to
    RELIABILITY_LT_IMPACT extra days of lead time std, reflecting
    the empirical relationship between supplier reliability and
    delivery consistency (Chopra & Meindl, 2019, S.14).
"""

import numpy as np
from config import *

class StochasticEnvironment:
    """
    Simulation environment for manufacturing and logistics operations.

    Uncertainty sources:
        - Demand:     Poisson(lambda) [default] or NegBin(r=5, same mean) per day
        - Lead times: LogNormal(mu_ln, sigma_ln) matching N(mu, sigma) moments
                      but right-skewed -- truncated at 1 day minimum
        - Disruptions: Bernoulli(p) onset, Exponential(duration) recovery
        - Supplier reliability: modulates lead time std
    """

    def __init__(self, uncertainty_level: str = 'base', seed: int = None,
                 unit_cost: float = None, demand_lambda: int = None,
                 lead_time_mean: int = None,
                 demand_distribution: str = 'poisson'):
        """
        Args:
            uncertainty_level:    'base' or 'high'
            seed:                 random seed for reproducibility
            unit_cost:            per-SKU unit cost override 
            demand_lambda:        per-SKU mean demand override 
            lead_time_mean:       per-SKU lead time override 
            demand_distribution:  'poisson' (default) or 'negbin'
                                  NegBin uses r=5 giving CV ~ 2x Poisson at
                                  the same mean -- useful for robustness tests.
        """
        self.uncertainty_level    = uncertainty_level
        self.demand_distribution  = demand_distribution

        # the demand sequence is NEVER contaminated by lead-time or disruption
        # draws. This guarantees paired experimental validity -- baseline and
        # agentic runs with the same seed face identical demand sequences,
        # regardless of how many orders (and thus lead-time draws) each places.
        # Seeds are derived deterministically from the master seed using
        # SeedSequence.spawn(), which is the NumPy-recommended approach for
        # multiple independent streams (Harris et al., 2020, Nature).
        if seed is not None:
            ss = np.random.SeedSequence(seed)
            child_seeds = ss.spawn(3)
            self.demand_rng     = np.random.default_rng(child_seeds[0])
            self.lt_rng         = np.random.default_rng(child_seeds[1])
            self.disruption_rng = np.random.default_rng(child_seeds[2])
        else:
            self.demand_rng     = np.random.default_rng()
            self.lt_rng         = np.random.default_rng()
            self.disruption_rng = np.random.default_rng()
        # Keep self.rng as alias for demand_rng for any external code that reads it
        self.rng = self.demand_rng

        # Reliability score in [0,1]; lower score -> higher lead time variance.
        reliability = SUPPLIER_RELIABILITY.get(ACTIVE_SUPPLIER, 0.90)
        # Extra std = RELIABILITY_LT_IMPACT * (1 - reliability)
        # At full reliability (1.0) -> 0 extra; at 0.78 (ASIA) -> ~0.18 days extra
        self._reliability_std_extra = RELIABILITY_LT_IMPACT * (1.0 - reliability)

        # Calibrate parameters (allow per-SKU overrides for multi-product catalogueuct catalogue)
        if uncertainty_level == 'base':
            self.demand_lambda   = demand_lambda   if demand_lambda   is not None else DEMAND_LAMBDA_BASE
            self.lead_time_mean  = lead_time_mean  if lead_time_mean  is not None else LEAD_TIME_MEAN
            self.lt_mean         = self.lead_time_mean
            self.lt_std          = LEAD_TIME_STD_BASE + self._reliability_std_extra
            self.disruption_prob = DISRUPTION_PROB_BASE
        else:
            self.demand_lambda   = demand_lambda   if demand_lambda   is not None else DEMAND_LAMBDA_HIGH
            self.lead_time_mean  = lead_time_mean  if lead_time_mean  is not None else LEAD_TIME_MEAN
            self.lt_mean         = self.lead_time_mean
            self.lt_std          = LEAD_TIME_STD_HIGH + self._reliability_std_extra
            self.disruption_prob = DISRUPTION_PROB_HIGH

        # Pre-compute log-normal parameters for lead time sampling
        self._recompute_lognormal_params()

        # State
        self.day = 0
        self.disruption_active = False
        self.disruption_days_remaining = 0
        self.disruption_history = []
        self._disruption_days_total = 0

    def _recompute_lognormal_params(self):
        """
        Recompute log-normal parameters from current lt_mean / lt_std.
        Called on init and after domain-randomisation parameter overrides.
        """
        self._lt_mu_ln, self._lt_sigma_ln = self._lognormal_params(
            self.lt_mean, self.lt_std)
        d_lt_mean = self.lt_mean * DISRUPTION_LT_MULTIPLIER
        d_lt_std  = self.lt_std  * DISRUPTION_STD_MULTIPLIER
        self._lt_d_mu_ln, self._lt_d_sigma_ln = self._lognormal_params(
            d_lt_mean, d_lt_std)

    @staticmethod
    def _lognormal_params(mean: float, std: float):
        """
        Convert (mean, std) of the desired distribution to log-normal
        parameters (mu_ln, sigma_ln) such that E[X]=mean, Std[X]=std.

        Formulae (Wikipedia: Log-normal distribution):
            sigma_ln = sqrt(log(1 + (std/mean)^2))
            mu_ln    = log(mean) - sigma_ln^2 / 2
        """
        sigma_ln = np.sqrt(np.log(1.0 + (std / mean) ** 2))
        mu_ln    = np.log(mean) - 0.5 * sigma_ln ** 2
        return mu_ln, sigma_ln

    def reset(self, seed: int = None):
        """Reset environment for a new simulation run."""
        if seed is not None:
            ss = np.random.SeedSequence(seed)
            child_seeds = ss.spawn(3)
            self.demand_rng     = np.random.default_rng(child_seeds[0])
            self.lt_rng         = np.random.default_rng(child_seeds[1])
            self.disruption_rng = np.random.default_rng(child_seeds[2])
            self.rng = self.demand_rng
        self.day = 0
        self.disruption_active = False
        self.disruption_days_remaining = 0
        self.disruption_history = []
        self._disruption_days_total = 0

    def sample_demand(self) -> int:
        """
        Sample daily demand.

        'poisson' -- Poisson(lambda)                    CV ~ 0.26 at lambda=15
        'negbin'  -- NegativeBinomial(r=5, p=r/(r+lam)) CV ~ 0.52 at lambda=15
                    Same mean as Poisson, variance = mean + mean^2/r.
                    Used for overdispersion robustness tests.
        """
        if self.demand_distribution == 'negbin':
            r   = NEGBIN_R
            p   = r / (r + self.demand_lambda)
            return int(self.rng.negative_binomial(r, p))

        if self.demand_distribution == 'seasonal':
            # Weekly cycle: peaks mid-week (days 2-3), troughs weekend (days 5-6).
            # Quarterly uplift: +20% in Q4 (days 274-365 of simulated year).
            # Day-of-week factor: sin curve, amplitude 30%.
            day_of_week    = (self.day - 1) % 7  # advance_day() increments before sampling; subtract 1 to align Day 1 → day_of_week=0
            weekly_factor  = 1.0 + 0.30 * float(np.sin(2 * np.pi * day_of_week / 7))
            day_of_year    = (self.day - 1) % 365  # align with corrected day_of_week offset
            q4_factor      = 1.20 if day_of_year >= 274 else 1.0
            adj_lambda     = max(1.0, self.demand_lambda * weekly_factor * q4_factor)
            return int(self.rng.poisson(adj_lambda))

        # Default: Poisson
        return int(self.rng.poisson(self.demand_lambda))

    @property
    def demand_variance(self) -> float:
        """
        Theoretical daily demand variance for the current distribution.

        Poisson:  Var(D) = lambda  (variance equals mean)
        NegBin:   Var(D) = lambda + lambda^2/r  (overdispersed; 4* Poisson at lambda=15, r=5)

        Used by the safety stock floor to scale SS appropriately for the
        active demand distribution. Without this correction, the Holt SS
        underestimates required buffer for NegBin conditions by ~50%
        (SS_NegBin = 2* SS_Poisson at same mean, same LT, same service factor).
        """
        lam = self.demand_lambda
        if self.demand_distribution == 'negbin':
            return lam + lam**2 / NEGBIN_R
        return lam  # Poisson: variance = mean

    def sample_lead_time(self) -> int:
        """
        Sample lead time from log-normal distribution.
        Uses lt_rng (independent of demand_rng) to preserve paired-seed validity.
        """
        lt = self.lt_rng.lognormal(self._lt_mu_ln, self._lt_sigma_ln)
        return max(1, int(round(lt)))

    def sample_lead_time_disrupted(self) -> int:
        """
        Lead time during disruption -- elevated log-normal.
        Uses lt_rng (independent of demand_rng) to preserve paired-seed validity.
        """
        lt = self.lt_rng.lognormal(self._lt_d_mu_ln, self._lt_d_sigma_ln)
        return max(2, int(round(lt)))

    def step_disruption(self) -> bool:
        """
        Advance disruption state machine by one day.
        Uses disruption_rng (independent of demand_rng) to preserve paired-seed validity.
        Returns True if disruption is currently active.
        """
        if self.disruption_active:
            self.disruption_days_remaining -= 1
            if self.disruption_days_remaining <= 0:
                self.disruption_active = False
        else:
            # Check for new disruption onset
            if self.disruption_rng.random() < self.disruption_prob:
                self.disruption_active = True
                # Duration from geometric distribution
                self.disruption_days_remaining = max(
                    1, int(self.disruption_rng.exponential(DISRUPTION_DURATION_MEAN)))
                self.disruption_history.append(self.day)

        return self.disruption_active

    def get_effective_lead_time(self) -> int:
        """Get lead time accounting for disruption state."""
        if self.disruption_active:
            return self.sample_lead_time_disrupted()
        return self.sample_lead_time()

    def advance_day(self):
        """Advance simulation by one day."""
        self.day += 1
        return self.step_disruption()

    def get_disruption_summary(self) -> dict:
        """Summary statistics of disruption events."""
        return {
            'total_disruptions': len(self.disruption_history),
            'disruption_days':   self._disruption_days_total,
            'disruption_rate':   self._disruption_days_total / max(1, self.day),
        }

