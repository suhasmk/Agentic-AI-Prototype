"""
Supplier Reliability Feed -- 

Simulates a live supplier reliability feed, modelling each supplier
region's historical on-time delivery rate and the resulting impact on
effective lead time and disruption probability.

In production, this would connect to a supplier portal or EDI feed
(e.g., SAP Ariba, Coupa). In the simulation, reliability scores are
drawn from a Beta distribution calibrated to the DataCo dataset's
observed late-delivery rates by product category.

Effect on simulation:
  - Effective lead time  += (1 - reliability) * SUPPLIER_RELIABILITY_IMPACT * lead_time
  - Disruption probability += (1 - reliability) * 0.05

This means a reliability of 0.78 (ASIA suppliers) increases lead time
by an average of 11% and disruption probability by +1.1 percentage
points -- consistent with empirical late-delivery rates in the DataCo
dataset (~22% late shipments for Asian-origin orders).

Reference: Tang, C.S. (2006). Perspectives in supply chain risk management.
International Journal of Production Economics, 103(2), 451-488.
"""

import numpy as np
from typing import Dict, Optional
from config import SUPPLIER_RELIABILITY, SUPPLIER_RELIABILITY_IMPACT

class SupplierProfile:
    """Dynamic reliability profile for one supplier region."""

    def __init__(self, region: str, base_reliability: float, seed: int = 0):
        self.region           = region
        self.base_reliability = base_reliability
        self.rng              = np.random.default_rng(seed)
        # Beta distribution parameters calibrated to base_reliability
        # Mean = alpha/(alpha+beta); set concentration = 20 for tight distribution
        concentration = 20
        alpha = base_reliability * concentration
        beta  = (1 - base_reliability) * concentration
        self._alpha = max(alpha, 0.1)
        self._beta  = max(beta,  0.1)
        self.current_reliability = base_reliability
        self._history: list = []

    def sample_daily_reliability(self) -> float:
        """Sample today's reliability score from Beta distribution."""
        r = float(self.rng.beta(self._alpha, self._beta))
        self._history.append(r)
        self.current_reliability = r
        return r

    def rolling_reliability(self, window: int = 30) -> float:
        """Rolling mean reliability over last `window` days."""
        if not self._history:
            return self.base_reliability
        tail = self._history[-window:]
        return float(np.mean(tail))

    def lead_time_adjustment(self) -> float:
        """
        Fractional lead time increase due to unreliability.
        Returns a multiplier: 1.0 = no impact; >1.0 = extended lead time.
        """
        return 1.0 + (1.0 - self.current_reliability) * SUPPLIER_RELIABILITY_IMPACT

    def disruption_probability_adjustment(self) -> float:
        """Additional disruption probability from supplier unreliability."""
        return (1.0 - self.current_reliability) * 0.05

class SupplierReliabilityFeed:
    """
    Aggregates reliability profiles for all supplier regions.
    Called each simulation day to update reliability scores before
    the logistics and inventory agents make their decisions.
    """

    def __init__(self, seed: int = 42):
        self._profiles: Dict[str, SupplierProfile] = {}
        for i, (region, base_rel) in enumerate(SUPPLIER_RELIABILITY.items()):
            self._profiles[region] = SupplierProfile(region, base_rel,
                                                      seed=seed + i * 100)
        self.event_log: list = []

    def advance_day(self, day: int) -> Dict[str, float]:
        """
        Sample daily reliability for all regions.
        Returns dict of {region: reliability_score}.
        """
        scores = {}
        for region, profile in self._profiles.items():
            r = profile.sample_daily_reliability()
            scores[region] = r
            # Log significant reliability drops
            if r < profile.base_reliability * 0.80:
                self.event_log.append({
                    'day': day, 'region': region,
                    'reliability': round(r, 3),
                    'base': profile.base_reliability,
                    'drop_pct': round((1 - r / profile.base_reliability) * 100, 1),
                    'event': 'LOW_RELIABILITY_ALERT',
                })
        return scores

    def get_lead_time_multiplier(self, region: str) -> float:
        """Lead time multiplier for a given supplier region today."""
        if region not in self._profiles:
            return 1.0
        return self._profiles[region].lead_time_adjustment()

    def get_disruption_adjustment(self, region: str) -> float:
        """Extra disruption probability for a given region today."""
        if region not in self._profiles:
            return 0.0
        return self._profiles[region].disruption_probability_adjustment()

    def summary(self) -> dict:
        return {
            'regions': {
                region: {
                    'base_reliability':    p.base_reliability,
                    'current_reliability': round(p.current_reliability, 3),
                    'rolling_30d':         round(p.rolling_reliability(30), 3),
                    'lead_time_multiplier': round(p.lead_time_adjustment(), 3),
                }
                for region, p in self._profiles.items()
            },
            'low_reliability_events': len(self.event_log),
        }
