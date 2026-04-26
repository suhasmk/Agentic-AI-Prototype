"""
Model Drift Monitor -- 

Implements rolling service-level monitoring with automatic retraining
recommendations when the deployed Q-learning policy degrades below
configurable thresholds.

Concept drift occurs when the statistical properties of the environment
change such that a previously trained policy becomes sub-optimal. In
supply chain contexts, common causes include seasonal demand shifts,
supplier base changes, and structural disruption pattern changes.

The monitor tracks a 30-day rolling service level. Two thresholds:
  DRIFT_ALERT_THRESHOLD  (0.90): Generates a governance warning
  DRIFT_RETRAIN_TRIGGER  (0.85): Recommends full policy retraining

Reference: Gama, J. et al. (2014). A survey on concept drift adaptation.
ACM Computing Surveys, 46(4), 1-37.
"""

import numpy as np
from collections import deque
from typing import List, Dict, Optional
from config import DRIFT_WINDOW_DAYS, DRIFT_ALERT_THRESHOLD, DRIFT_RETRAIN_TRIGGER

class DriftEvent:
    """Records a single drift detection event."""
    def __init__(self, day: int, sku: str, sl_current: float,
                 sl_baseline: float, event_type: str, message: str):
        self.day         = day
        self.sku         = sku
        self.sl_current  = sl_current
        self.sl_baseline = sl_baseline
        self.event_type  = event_type   # "ALERT" or "RETRAIN"
        self.message     = message

class ModelDriftMonitor:
    """
    Tracks rolling service level and detects policy degradation.

    Algorithm: Page-Hinkley test (simplified variant) over a rolling
    window. A drift event is raised when the rolling mean service level
    crosses either threshold for >=3 consecutive windows (to avoid
    false positives from transient demand spikes).

    In the simulation, retraining recommendations are logged to the
    governance audit trail. In a production system, this would trigger
    an automated retraining pipeline (e.g., Airflow DAG or MLflow job).
    """

    def __init__(self, sku: str = "DEFAULT", baseline_sl: float = None):
        self.sku           = sku
        self.baseline_sl   = baseline_sl    # Established from training evaluation
        self._daily_sl     = deque(maxlen=DRIFT_WINDOW_DAYS)
        self._fulfilled    = deque(maxlen=DRIFT_WINDOW_DAYS)
        self._demanded     = deque(maxlen=DRIFT_WINDOW_DAYS)
        self.drift_events: List[DriftEvent] = []
        self._consecutive_alerts   = 0
        self._consecutive_retrains = 0
        self._retraining_recommended = False
        self.daily_log: List[Dict] = []

    def update(self, day: int, units_fulfilled: float,
               units_demanded: float) -> Optional[DriftEvent]:
        """
        Update monitor with today's fulfilment data.
        Returns a DriftEvent if a threshold crossing is detected.
        """
        daily_sl = (units_fulfilled / units_demanded
                    if units_demanded > 0 else 1.0)
        self._fulfilled.append(units_fulfilled)
        self._demanded.append(units_demanded)
        self._daily_sl.append(daily_sl)

        if len(self._daily_sl) < DRIFT_WINDOW_DAYS:
            self.daily_log.append({'day': day, 'daily_sl': daily_sl,
                                   'rolling_sl': None, 'event': None})
            return None

        rolling_sl = (sum(self._fulfilled) / max(sum(self._demanded), 1))
        event = None

        if rolling_sl < DRIFT_RETRAIN_TRIGGER:
            self._consecutive_retrains += 1
            self._consecutive_alerts   = 0
            if self._consecutive_retrains >= 3 and not self._retraining_recommended:
                self._retraining_recommended = True
                msg = (f"[RETRAIN] SKU={self.sku} | Day={day} | "
                       f"Rolling-{DRIFT_WINDOW_DAYS}d SL={rolling_sl:.1%} "
                       f"< RETRAIN threshold ({DRIFT_RETRAIN_TRIGGER:.0%}). "
                       f"Policy retraining recommended immediately.")
                event = DriftEvent(day, self.sku, rolling_sl,
                                   self.baseline_sl or 0.0, "RETRAIN", msg)
                self.drift_events.append(event)

        elif rolling_sl < DRIFT_ALERT_THRESHOLD:
            self._consecutive_alerts   += 1
            self._consecutive_retrains = 0
            self._retraining_recommended = False
            if self._consecutive_alerts >= 3:
                msg = (f"[ALERT] SKU={self.sku} | Day={day} | "
                       f"Rolling-{DRIFT_WINDOW_DAYS}d SL={rolling_sl:.1%} "
                       f"< ALERT threshold ({DRIFT_ALERT_THRESHOLD:.0%}). "
                       f"Monitor closely; consider preventive retraining.")
                event = DriftEvent(day, self.sku, rolling_sl,
                                   self.baseline_sl or 0.0, "ALERT", msg)
                self.drift_events.append(event)
        else:
            self._consecutive_alerts   = 0
            self._consecutive_retrains = 0
            self._retraining_recommended = False

        self.daily_log.append({
            'day':        day,
            'daily_sl':   round(daily_sl, 4),
            'rolling_sl': round(rolling_sl, 4),
            'event':      event.event_type if event else None,
        })
        return event

    def summary(self) -> dict:
        rolling_sls = [r['rolling_sl'] for r in self.daily_log
                       if r['rolling_sl'] is not None]
        return {
            'sku':                   self.sku,
            'n_drift_events':        len(self.drift_events),
            'n_alerts':              sum(1 for e in self.drift_events if e.event_type == "ALERT"),
            'n_retrain_triggers':    sum(1 for e in self.drift_events if e.event_type == "RETRAIN"),
            'min_rolling_sl':        min(rolling_sls) if rolling_sls else None,
            'avg_rolling_sl':        float(np.mean(rolling_sls)) if rolling_sls else None,
            'retraining_recommended': self._retraining_recommended,
            'policy_stable':         len(self.drift_events) == 0,
        }

class PortfolioDriftMonitor:
    """
    Coordinates drift monitoring across all SKUs in the portfolio.
    Provides portfolio-level drift summary for governance reporting.
    """

    def __init__(self):
        self._monitors: Dict[str, ModelDriftMonitor] = {}

    def register_sku(self, sku: str, baseline_sl: float = None):
        self._monitors[sku] = ModelDriftMonitor(sku, baseline_sl)

    def update(self, sku: str, day: int, fulfilled: float,
               demanded: float) -> Optional[DriftEvent]:
        if sku not in self._monitors:
            self.register_sku(sku)
        return self._monitors[sku].update(day, fulfilled, demanded)

    def portfolio_summary(self) -> dict:
        summaries = {sku: m.summary() for sku, m in self._monitors.items()}
        total_events = sum(s['n_drift_events'] for s in summaries.values())
        all_stable   = all(s['policy_stable'] for s in summaries.values())
        return {
            'per_sku':            summaries,
            'total_drift_events': total_events,
            'portfolio_stable':   all_stable,
        }
