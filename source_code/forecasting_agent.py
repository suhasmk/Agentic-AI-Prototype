"""
Forecasting Agents -- Agentic AI Prototype v2

Implements three forecasting approaches:
  1. MovingAverageForecaster  -- Baseline (7-day MA)
  2. AdaptiveForecaster        -- Holt double-exponential smoothing 
  3. ARIMAForecaster           -- Autoregressive model AR(p) with differencing 
     Implemented in pure NumPy (no statsmodels dependency) as AR(3) with
     first-order differencing -- equivalent to ARIMA(3,1,0).

ARIMA-based per-SKU demand forecasting for products with trend or
autocorrelated demand patterns. Implemented in pure NumPy as AR(3) with
first-order differencing (ARIMA(3,1,0) equivalent). Coefficients estimated
by OLS on first-differenced series -- equivalent to conditional MLE for
stationary AR processes (Hamilton 1994, S.8.2).
"""

import numpy as np
from collections import deque
from typing import List
from config import HOLT_ALPHA, HOLT_BETA, HOLT_SS_SERVICE_FACTOR, DEMAND_LAMBDA_BASE, NEGBIN_R, ARIMA_AR_ORDER, ARIMA_REFIT_EVERY

class MovingAverageForecaster:
    """7-day moving average forecaster.
    """
    def __init__(self, window: int = 7):
        self.window = window
        self.history = deque(maxlen=window)
        self._all_history: List[float] = []   # full unbounded history for log
        self.name = f"Moving Average (window={window})"

    def update(self, observation: float):
        self.history.append(observation)
        self._all_history.append(observation)

    def forecast(self) -> float:
        return float(np.mean(self.history)) if self.history else float(DEMAND_LAMBDA_BASE)

    def get_forecast_log(self) -> List[dict]:
        """Return list of {'actual': ..., 'forecast': ...} for each observed day.

        The one-step-ahead forecast on day t is computed using the window
        of observations up to (but not including) day t.
        """
        log = []
        window = self.window
        for i, actual in enumerate(self._all_history):
            if i == 0:
                pred = actual  # no history yet
            else:
                past = self._all_history[max(0, i - window):i]
                pred = float(np.mean(past))
            log.append({'actual': actual, 'forecast': pred})
        return log

    def get_mae(self) -> float:
        """Mean Absolute Error of one-step-ahead forecasts."""
        log = self.get_forecast_log()
        if len(log) < 2:
            return 0.0
        errors = [abs(e['actual'] - e['forecast']) for e in log[1:]]
        return float(np.mean(errors))

class AdaptiveForecaster:
    """
    Holt double-exponential smoothing (trend-adjusted).
    
    """
    def __init__(self, alpha: float = HOLT_ALPHA, beta: float = HOLT_BETA):
        self.alpha = alpha
        self.beta  = beta
        self.level = None
        self.trend = 0.0
        self.history: List[float] = []
        self.name = "Holt Adaptive Forecaster"

    def update(self, observation: float):
        self.history.append(observation)
        if self.level is None:
            self.level = observation
        else:
            prev_level = self.level
            self.level = self.alpha * observation + (1 - self.alpha) * (self.level + self.trend)
            self.trend = self.beta * (self.level - prev_level) + (1 - self.beta) * self.trend

    def forecast(self, horizon: int = 1) -> float:
        if self.level is None:
            return float(DEMAND_LAMBDA_BASE)
        return max(0.0, self.level + horizon * self.trend)

    def forecast_accuracy(self) -> dict:
        """Compute retrospective MAE and RMSE on one-step-ahead predictions.
        Note: replays the Holt sequence from history. This is a retrospective
        re-simulation, not the actual forecast errors logged during simulation.
        """
        if len(self.history) < 10:
            return {'MAE': None, 'RMSE': None}
        preds, actuals = [], []
        level, trend = self.history[0], 0.0
        for obs in self.history[1:]:
            pred = level + trend
            preds.append(pred)
            actuals.append(obs)
            prev = level
            level = self.alpha * obs + (1 - self.alpha) * (level + trend)
            trend = self.beta * (level - prev) + (1 - self.beta) * trend
        errors = np.array(actuals) - np.array(preds)
        return {
            'MAE':  float(np.mean(np.abs(errors))),
            'RMSE': float(np.sqrt(np.mean(errors**2))),
            'n':    len(errors),
        }

    def get_forecast_log(self) -> List[dict]:
        """Return list of {'actual': ..., 'forecast': ...} for each observed day.

        Replays the Holt update sequence to produce one-step-ahead forecasts.
        """
        if not self.history:
            return []
        log = [{'actual': self.history[0], 'forecast': self.history[0]}]
        level = self.history[0]
        trend = 0.0
        for obs in self.history[1:]:
            pred = max(0.0, level + trend)
            log.append({'actual': obs, 'forecast': pred})
            prev = level
            level = self.alpha * obs + (1 - self.alpha) * (level + trend)
            trend = self.beta * (level - prev) + (1 - self.beta) * trend
        return log

    def get_mae(self) -> float:
        """Mean Absolute Error of one-step-ahead forecasts."""
        log = self.get_forecast_log()
        if len(log) < 2:
            return 0.0
        return float(np.mean([abs(e['actual'] - e['forecast']) for e in log[1:]]))

    def forecast_safety_stock(self, lead_time: float = 5.0,
                               service_factor: float = HOLT_SS_SERVICE_FACTOR) -> float:
        """
        Compute dynamic safety stock from Holt forecast error variance.

        Safety stock = service_factor * sigma_forecast * sqrt(lead_time)
        where sigma_forecast is the rolling std of forecast errors over last 14 days.

        service_factor defaults to HOLT_SS_SERVICE_FACTOR from config (z=2.33 -> 99% SL).
        Returns 0.0 if insufficient history (<14 observations).

        Reference: Silver, Pyke & Thomas (2017), Inventory Management, S.7.3.

        Note: simulation_runner.py uses env.demand_variance directly for the
        safety stock floor (NegBin-corrected formula). This method is retained
        for diagnostics but is not the primary SS calculation in production.
        """
        log = self.get_forecast_log()
        if len(log) < 14:
            return 0.0
        errors = [abs(e['actual'] - e['forecast']) for e in log[-14:]]
        sigma_forecast = float(np.std(errors, ddof=1))
        return service_factor * sigma_forecast * np.sqrt(lead_time)

    def forecast_uncertainty(self) -> float:
        """
        Returns rolling sigma of forecast errors over last 14 days.
        Used by the Q-learning state encoder to signal forecast confidence
        
        """
        log = self.get_forecast_log()
        if len(log) < 14:
            return 2.0   # conservative default when warming up
        errors = [abs(e['actual'] - e['forecast']) for e in log[-14:]]
        return float(np.std(errors, ddof=1))

class ARIMAForecaster:
    """
    Autoregressive model AR(p) with first-order differencing -- ARIMA(p,1,0).

    Implemented in pure NumPy without external dependencies.
    Fits AR(3) coefficients via ordinary least squares on the differenced
    series, then integrates the forecast back to the original scale.

    Mathematical formulation:
        Let d_t = y_t - y_{t-1}  (first differences)
        Fit: d_t = c + phi1*d_{t-1} + phi2*d_{t-2} + phi3*d_{t-3} + eps_t
        OLS: phi = inv(X^T X) X^T d  where X is the lagged design matrix
        Forecast: y_hat_{t+h} = y_t + c + phi1*d_t + phi2*d_{t-1} + phi3*d_{t-2}

    
    demand (Bicycles, Golf) compared to Holt smoothing. Fitted every 30
    days on the most recent 90 observations.

    Reference: Box, G.E.P., Jenkins, G.M. (1976). Time Series Analysis.
    """

    AR_ORDER    = ARIMA_AR_ORDER    # Number of autoregressive lags (config)
    REFIT_EVERY = ARIMA_REFIT_EVERY # Refit coefficients every N observations (config)
    MIN_FIT_OBS = 20   # Minimum history before AR fitting

    def __init__(self):
        self.history: List[float] = []
        self.coeffs   = None       # [intercept, phi1, phi2, phi3]
        self.fit_count = 0
        self.name = "ARIMA(3,1,0) Demand Forecaster"
        self._fallback = AdaptiveForecaster()  # used until enough history

    def update(self, observation: float):
        self.history.append(float(observation))
        self._fallback.update(observation)
        # Refit every REFIT_EVERY observations (or when first sufficient)
        n = len(self.history)
        if n >= self.MIN_FIT_OBS and (n % self.REFIT_EVERY == 0 or self.coeffs is None):
            self._refit()

    def _refit(self):
        """Fit AR(p) on first differences via OLS."""
        # Use last 90 observations (or all if fewer)
        series = np.array(self.history[-90:], dtype=float)
        # First differences
        d = np.diff(series)
        p = self.AR_ORDER
        if len(d) <= p:
            return
        # Build lagged design matrix [1, d_{t-1}, d_{t-2}, d_{t-3}]
        n = len(d) - p
        X = np.ones((n, p + 1))
        for lag in range(1, p + 1):
            X[:, lag] = d[p - lag: p - lag + n]
        y = d[p:]
        # OLS: coeffs = (XTX)?^1 XTy
        try:
            XtX = X.T @ X
            self.coeffs = np.linalg.lstsq(XtX, X.T @ y, rcond=None)[0]
        except np.linalg.LinAlgError:
            self.coeffs = None
        self.fit_count += 1

    def forecast(self, horizon: int = 1) -> float:
        """One-step-ahead forecast. horizon > 1 uses iterated prediction."""
        if self.coeffs is None or len(self.history) < self.AR_ORDER + 2:
            return self._fallback.forecast()

        # Get recent differences
        series = np.array(self.history[-10:], dtype=float)
        d = np.diff(series)

        level = self.history[-1]
        for _ in range(horizon):
            if len(d) < self.AR_ORDER:
                pred_d = self.coeffs[0]
            else:
                lags = d[-self.AR_ORDER:][::-1]
                pred_d = self.coeffs[0] + np.dot(self.coeffs[1:], lags)
            level = level + pred_d
            d = np.append(d, pred_d)

        return max(0.0, level)

    def forecast_accuracy(self) -> dict:
        """Walk-forward one-step-ahead accuracy on last 60 observations."""
        n = len(self.history)
        if n < self.MIN_FIT_OBS + 10:
            return {'MAE': None, 'RMSE': None}

        preds, actuals = [], []
        # Rolling window evaluation
        window = min(60, n - self.MIN_FIT_OBS)
        for i in range(window):
            idx = n - window + i
            if idx < self.AR_ORDER + 2:
                continue
            # Fit on data up to idx
            sub = np.array(self.history[:idx], dtype=float)
            d   = np.diff(sub)
            p   = self.AR_ORDER
            if len(d) <= p:
                continue
            n_sub = len(d) - p
            X = np.ones((n_sub, p + 1))
            for lag in range(1, p + 1):
                X[:, lag] = d[p - lag: p - lag + n_sub]
            y_sub = d[p:]
            try:
                c = np.linalg.lstsq(X.T @ X, X.T @ y_sub, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            # One-step forecast
            lags  = d[-p:][::-1]
            pred_d = c[0] + np.dot(c[1:], lags)
            pred   = sub[-1] + pred_d
            preds.append(pred)
            actuals.append(self.history[idx])

        if not preds:
            return {'MAE': None, 'RMSE': None}

        errors = np.array(actuals) - np.array(preds)
        return {
            'MAE':  float(np.mean(np.abs(errors))),
            'RMSE': float(np.sqrt(np.mean(errors**2))),
            'n':    len(errors),
        }

    def get_forecast_log(self) -> List[dict]:
        """Return list of {'actual': ..., 'forecast': ...} for each observed day.

        Uses the walk-forward AR(3) evaluation (same as forecast_accuracy).
        Falls back to Holt one-step predictions until enough history exists.
        """
        if not self.history:
            return []
        p = self.AR_ORDER
        log = []
        level_f = self._fallback
        # Replay with a fresh fallback to get clean one-step predictions
        from forecasting_agent import AdaptiveForecaster as _AF
        fb = _AF(self.alpha if hasattr(self, 'alpha') else 0.3,
                 self.beta  if hasattr(self, 'beta')  else 0.1)
        for i, actual in enumerate(self.history):
            if i < p + self.MIN_FIT_OBS:
                pred = fb.forecast() if i > 0 else float(actual)
            else:
                sub = np.array(self.history[:i], dtype=float)
                d   = np.diff(sub)
                if len(d) <= p:
                    pred = fb.forecast()
                else:
                    n_sub = len(d) - p
                    X = np.ones((n_sub, p + 1))
                    for lag in range(1, p + 1):
                        X[:, lag] = d[p - lag: p - lag + n_sub]
                    y_sub = d[p:]
                    try:
                        c = np.linalg.lstsq(X.T @ X, X.T @ y_sub, rcond=None)[0]
                        lags = d[-p:][::-1]
                        pred_d = c[0] + np.dot(c[1:], lags)
                        pred = float(max(0.0, sub[-1] + pred_d))
                    except np.linalg.LinAlgError:
                        pred = fb.forecast()
            fb.update(float(actual))
            log.append({'actual': float(actual), 'forecast': pred})
        return log

    def get_mae(self) -> float:
        """Mean Absolute Error of one-step-ahead forecasts."""
        log = self.get_forecast_log()
        if len(log) < 2:
            return 0.0
        return float(np.mean([abs(e['actual'] - e['forecast']) for e in log[1:]]))

def select_forecaster(sku_params: dict = None) -> AdaptiveForecaster:
    """
    Factory: select appropriate forecaster for a given SKU.
    Uses ARIMA(3,1,0) for SKUs with higher demand variability or trend
    (bicycles, golf); Holt smoothing for all others.
    """
    if sku_params is None:
        return AdaptiveForecaster()
    category = sku_params.get("category", "").lower()
    if category in ("bicycles", "golf"):
        return ARIMAForecaster()
    return AdaptiveForecaster()
