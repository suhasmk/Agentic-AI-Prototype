"""
Consolidated Shipping Manager -- 

Aggregates same-day orders across multiple SKUs and combines those from
the same supplier region into a single consolidated shipment.

Business logic:
  - Orders from the same supplier_region placed on the same day are
    eligible for consolidation.
  - Consolidated shipments attract a 15% cost discount and 10% carbon
    reduction per the config values (based on Fuller Truck Load economics).
  - The logistics agent still selects the optimal shipping MODE per
    consolidated shipment; the discount is applied on top.

This addresses the identified gap that independent single-SKU ordering
overlooks inter-product consolidation opportunities -- a key lever in
real-world logistics cost management (McKinnon et al., 2015).

Academic justification:
  Freight consolidation is a well-established logistics optimisation
  technique. Daganzo (1988) shows that consolidating shipments from
  multiple origins reduces per-unit shipping cost by 15-30%. Our
  implementation uses a conservative 15% cost and 10% carbon saving,
  consistent with the lower bound of literature estimates.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import numpy as np
from config import CONSOLIDATION_DISCOUNT, CONSOLIDATION_CARBON_SAVING

@dataclass
class PendingOrder:
    """Represents a single SKU order awaiting dispatch."""
    sku:              str
    day:              int
    order_qty:        int
    order_value:      float          # £ value of order
    supplier_region:  str
    shipping_mode:    str
    base_cost:        float          # Cost before consolidation
    base_carbon:      float          # Carbon before consolidation
    weight_kg:        float
    urgency_days:     int
    req_id:           int

@dataclass
class ConsolidatedShipment:
    """Represents a merged shipment for multiple SKUs."""
    shipment_id:      int
    day:              int
    supplier_region:  str
    shipping_mode:    str
    orders:           List[PendingOrder] = field(default_factory=list)
    total_cost:       float = 0.0
    total_carbon:     float = 0.0
    total_weight_kg:  float = 0.0
    consolidation_saving_cost:   float = 0.0
    consolidation_saving_carbon: float = 0.0
    consolidated:     bool  = False    # True if >=2 SKUs merged

class ConsolidatedShippingManager:
    """
    Manages daily order batching and consolidation across all SKUs.

    Protocol:
      1. Each SKU's logistics agent selects the optimal shipping mode.
      2. Orders are grouped by (supplier_region, shipping_mode, day).
      3. Groups with >=2 orders are consolidated: discount applied to ALL
         orders in the group proportionally by order value.
      4. Savings are logged to the governance audit trail.

    Note: Same-day delivery (Same Day) is never consolidated -- it is by
    definition a single-order emergency shipment.
    """

    def __init__(self):
        self._pending:  List[PendingOrder]      = []
        self._dispatch: List[ConsolidatedShipment] = []
        self._shipment_counter = 0
        self.total_cost_saved    = 0.0
        self.total_carbon_saved  = 0.0
        self.consolidation_events = 0
        self.dispatch_log: List[dict] = []

    def register_order(self, order: PendingOrder):
        """Register a new order for potential consolidation."""
        self._pending.append(order)

    def dispatch_day(self, day: int) -> List[ConsolidatedShipment]:
        """
        Process all pending orders for the given day.
        Groups by supplier_region + shipping_mode; consolidates where
        >=2 orders exist (excluding Same Day).
        Returns list of dispatched ConsolidatedShipment objects.
        """
        today_orders = [o for o in self._pending if o.day == day]
        self._pending  = [o for o in self._pending if o.day != day]

        if not today_orders:
            return []

        # Group by (supplier_region, shipping_mode)
        groups: Dict[Tuple, List[PendingOrder]] = {}
        for order in today_orders:
            key = (order.supplier_region, order.shipping_mode)
            groups.setdefault(key, []).append(order)

        dispatched = []
        for (region, mode), orders in groups.items():
            self._shipment_counter += 1
            shipment = ConsolidatedShipment(
                shipment_id=self._shipment_counter,
                day=day,
                supplier_region=region,
                shipping_mode=mode,
                orders=orders,
            )

            # Consolidate if >=2 orders and mode is not Same Day
            can_consolidate = len(orders) >= 2 and mode != "Same Day"
            shipment.consolidated = can_consolidate

            total_base_cost   = sum(o.base_cost   for o in orders)
            total_base_carbon = sum(o.base_carbon  for o in orders)
            total_weight      = sum(o.weight_kg    for o in orders)

            if can_consolidate:
                cost_saving   = total_base_cost   * CONSOLIDATION_DISCOUNT
                carbon_saving = total_base_carbon * CONSOLIDATION_CARBON_SAVING
                shipment.consolidation_saving_cost   = cost_saving
                shipment.consolidation_saving_carbon = carbon_saving
                shipment.total_cost   = total_base_cost   - cost_saving
                shipment.total_carbon = total_base_carbon - carbon_saving
                self.total_cost_saved   += cost_saving
                self.total_carbon_saved += carbon_saving
                self.consolidation_events += 1

                # Distribute savings proportionally by order value
                total_val = sum(o.order_value for o in orders) or 1.0
                for order in orders:
                    share = order.order_value / total_val
                    order.base_cost   -= cost_saving   * share
                    order.base_carbon -= carbon_saving * share
            else:
                shipment.total_cost   = total_base_cost
                shipment.total_carbon = total_base_carbon

            shipment.total_weight_kg = total_weight

            self.dispatch_log.append({
                'shipment_id':    shipment.shipment_id,
                'day':            day,
                'region':         region,
                'mode':           mode,
                'n_skus':         len(orders),
                'skus':           [o.sku for o in orders],
                'total_cost':     round(shipment.total_cost, 2),
                'total_carbon':   round(shipment.total_carbon, 2),
                'consolidated':   can_consolidate,
                'cost_saving':    round(shipment.consolidation_saving_cost, 2),
                'carbon_saving':  round(shipment.consolidation_saving_carbon, 2),
            })

            dispatched.append(shipment)
            self._dispatch.append(shipment)

        return dispatched

    def summary(self) -> dict:
        """Return portfolio-level consolidation performance summary."""
        n_dispatched   = len(self._dispatch)
        n_consolidated = sum(1 for s in self._dispatch if s.consolidated)
        return {
            'total_shipments':        n_dispatched,
            'consolidated_shipments': n_consolidated,
            'consolidation_rate_pct': (n_consolidated / n_dispatched * 100
                                       if n_dispatched > 0 else 0.0),
            'total_cost_saved':       round(self.total_cost_saved, 2),
            'total_carbon_saved_kg':  round(self.total_carbon_saved, 2),
            'consolidation_events':   self.consolidation_events,
        }
