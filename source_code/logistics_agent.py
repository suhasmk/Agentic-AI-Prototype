"""
Logistics Routing Agent

The OptimizedLogisticsAgent selects the shipping mode that minimises a
composite cost function comprising shipping cost, carbon penalty, and
stockout-risk cost. With STOCKOUT_PENALTY = 250 £/day, express modes
are preferred whenever inventory urgency demands fast replenishment,
producing genuine mode differentiation from the baseline.
"""

import numpy as np
from typing import Dict, List
from config import SHIPPING_COSTS, SHIPPING_DAYS, CARBON_FACTORS, CARBON_PENALTY_PER_KGCO2

class ShipmentRequest:
    """Represents a logistics shipment request."""

    def __init__(self, req_id: int, order_qty: int, urgency_days: int, day: int):
        self.req_id = req_id
        self.order_qty = order_qty
        self.urgency_days = urgency_days  # days until projected stockout
        self.day = day
        self.selected_mode = None
        self.cost = 0.0
        self.carbon = 0.0
        self.delivery_days = 0

class DefaultLogisticsAgent:
    """
    Baseline logistics agent: always uses Standard Class shipping.
    
    """

    def __init__(self):
        self.name = "Standard Class Default (Baseline)"
        self.decision_log: List[Dict] = []

    def route(self, request: ShipmentRequest) -> ShipmentRequest:
        """Assign Standard Class regardless of urgency."""
        mode = "Standard Class"
        request.selected_mode = mode
        request.cost = SHIPPING_COSTS[mode] * request.order_qty
        # Carbon = kg CO2e per unit * units ordered (mode-specific intensity).
        # SHIPPING_DAYS multiplier removed: CARBON_FACTORS already encodes
        # total emissions intensity per unit shipped per mode.
        request.carbon = CARBON_FACTORS[mode] * request.order_qty
        request.delivery_days = SHIPPING_DAYS[mode]

        self.decision_log.append({
            'day': request.day,
            'req_id': request.req_id,
            'mode': mode,
            'qty': request.order_qty,
            'cost': request.cost,
            'carbon': request.carbon,
            'urgency_days': request.urgency_days,
            'rationale': 'Default Standard Class -- baseline policy'
        })
        return request

class OptimizedLogisticsAgent:
    """
    Agentic logistics agent: selects the shipping mode that minimises
    total cost (shipping cost + carbon penalty + stockout-risk cost).

    STOCKOUT_PENALTY = 250 £/day -- calibrated to reflect the economic
    cost of a stockout (lost margin + penalty clause) per the DataCo
    dataset unit economics. At this penalty level, the agent selects
    express modes (First Class, Same Day) when urgency_days < transit
    days of Standard Class, and Standard Class only when inventory
    position is comfortable. This produces genuine multi-modal behaviour
    distinct from the baseline.

    
    """

    # CARBON_PENALTY sourced from config.CARBON_PENALTY_PER_KGCO2 (no duplicate).
    STOCKOUT_PENALTY = 250.0

    def __init__(self, carbon_penalty: float = None, stockout_penalty: float = None):
        self.name = "Cost-Minimising Route Optimiser (Agentic)"
        self.decision_log: List[Dict] = []
        # Allow override for Pareto sweep 
        self.CARBON_PENALTY = carbon_penalty if carbon_penalty is not None else CARBON_PENALTY_PER_KGCO2
        if stockout_penalty is not None:
            self.STOCKOUT_PENALTY = stockout_penalty

    def score_mode(self, mode: str, request: ShipmentRequest) -> float:
        """
        Composite cost score for a shipping mode. Lower = better.

        Components:
          1. Direct shipping cost  -- £/unit * quantity
          2. Carbon penalty        -- kg CO₂e * carbon price proxy
          3. Stockout-risk cost    -- days of delay beyond urgency * penalty
        """
        ship_cost = SHIPPING_COSTS[mode] * request.order_qty
        # Carbon scoring: includes transit days to penalise slow modes
        # that delay replenishment (transport time × emission intensity).
        # Note: recorded request.carbon uses per-unit intensity only.
        carbon_cost = (CARBON_FACTORS[mode] * request.order_qty
                       * SHIPPING_DAYS[mode] * self.CARBON_PENALTY)
        delay_days = max(0, SHIPPING_DAYS[mode] - request.urgency_days)
        stockout_risk_cost = delay_days * self.STOCKOUT_PENALTY
        return ship_cost + carbon_cost + stockout_risk_cost

    def route(self, request: ShipmentRequest) -> ShipmentRequest:
        """Select cost-optimal shipping mode."""
        scores = {mode: self.score_mode(mode, request)
                  for mode in SHIPPING_COSTS}
        best_mode = min(scores, key=scores.get)

        request.selected_mode = best_mode
        request.cost = SHIPPING_COSTS[best_mode] * request.order_qty
        # Actual emissions: per-unit intensity × quantity.
        request.carbon = CARBON_FACTORS[best_mode] * request.order_qty
        request.delivery_days = SHIPPING_DAYS[best_mode]

        self.decision_log.append({
            'day': request.day,
            'req_id': request.req_id,
            'mode': best_mode,
            'qty': request.order_qty,
            'cost': request.cost,
            'carbon': request.carbon,
            'urgency_days': request.urgency_days,
            'all_scores': {m: round(s, 2) for m, s in scores.items()},
            'rationale': (f"Minimised composite cost: £{scores[best_mode]:.2f} "
                          f"(urgency={request.urgency_days}d, "
                          f"transit={SHIPPING_DAYS[best_mode]}d)")
        })
        return request

def create_shipment_request(day: int, order_qty: int,
                             inventory: float, demand_forecast: float,
                             req_id: int) -> ShipmentRequest:
    """
    Construct a ShipmentRequest from current simulation state.

    urgency_days: estimated days until stockout given current inventory
    and forecast demand -- used by OptimizedLogisticsAgent to assess
    whether fast shipping is needed to prevent a service-level failure.
    """
    if demand_forecast > 0:
        urgency_days = max(1, int(inventory / demand_forecast))
    else:
        urgency_days = 10  # comfortable; no urgency
    return ShipmentRequest(req_id, order_qty, urgency_days, day)
