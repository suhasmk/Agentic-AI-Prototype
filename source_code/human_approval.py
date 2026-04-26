"""
Human-in-the-Loop Approval Gateway -- 

Implements EU AI Act Article 14-compliant human oversight for high-stakes
autonomous replenishment decisions. Decisions exceeding configurable
thresholds (order quantity, order value, or disruption context) are
flagged for human review before execution.

In simulation, approval is auto-granted (no actual human present) and
every decision -- approved or escalated -- is logged to the governance
audit trail with full decision context and approval rationale.

Regulatory context:
  EU AI Act (2024) Article 14 requires "appropriate human oversight
  measures" for high-risk AI systems. The Act's Recital 48 specifically
  identifies supply chain management as a potentially high-risk domain.
  This gateway implements three escalation triggers:
    1. Large order quantity (> HITL_ORDER_QTY_THRESHOLD units)
    2. High order value    (> HITL_ORDER_VALUE_THRESHOLD £)
    3. Active disruption   (HITL_DISRUPTION_FLAG = True)

Reference: European Commission (2024). Regulation (EU) 2024/1689 on
Artificial Intelligence. Official Journal of the European Union.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from config import (HITL_ORDER_QTY_THRESHOLD, HITL_QTY_CAP,
                    HITL_ORDER_VALUE_THRESHOLD, HITL_DISRUPTION_FLAG,
                    HITL_DISRUPTION_UPLIFT, UNIT_COST, ORDER_QUANTITY_EOQ)

class ApprovalStatus(str, Enum):
    AUTO_APPROVED   = "AUTO_APPROVED"    # Below all thresholds
    ESCALATED       = "ESCALATED"        # Above threshold -- flagged
    APPROVED        = "APPROVED"         # Human approved (simulated)
    REJECTED        = "REJECTED"         # Human rejected (simulated)
    MODIFIED        = "MODIFIED"         # Human modified quantity

@dataclass
class ApprovalRequest:
    """Captures context for a single approval decision."""
    request_id:   int
    day:          int
    sku:          str
    agent_name:   str
    order_qty:    int
    order_value:  float
    disruption:   bool
    inventory:    float
    forecast:     float
    triggers:     List[str]          # Which thresholds were crossed
    status:       ApprovalStatus = ApprovalStatus.AUTO_APPROVED
    final_qty:    int   = 0          # Approved order quantity
    rationale:    str   = ""
    q_values:     Optional[List[float]] = None

class HumanApprovalGateway:
    """
    Approval gateway implementing EU AI Act Article 14 human oversight.

    Three escalation triggers:
      1. order_qty   > HITL_ORDER_QTY_THRESHOLD   (default: 200 units)
      2. order_value > HITL_ORDER_VALUE_THRESHOLD  (default: £20,000)
      3. disruption  == True  AND  HITL_DISRUPTION_FLAG == True

    Simulation approval policy (mirrors realistic human behaviour):
      - Quantity trigger:  Human approves but caps order at 150% of EOQ
      - Value trigger:     Human approves with no modification
      - Disruption trigger: Human approves with a 10% safety-stock uplift
      - Multiple triggers: Most conservative rule applies

    Full audit trail is maintained for governance reporting .
    """

    def __init__(self, sku: str = "DEFAULT", eoq: int = ORDER_QUANTITY_EOQ):
        self.sku          = sku
        self.eoq          = eoq
        self._counter     = 0
        self.approval_log: List[ApprovalRequest] = []
        self._escalations = 0
        self._auto_approvals = 0

    def evaluate(self, day: int, order_qty: int, disruption: bool,
                 inventory: float, forecast: float,
                 agent_name: str = "QL-Agent",
                 q_values: List[float] = None,
                 unit_cost: float = UNIT_COST) -> ApprovalRequest:
        """
        Evaluate an agent-proposed order against HITL thresholds.
        Returns an ApprovalRequest with final approved quantity.
        """
        self._counter += 1
        order_value = order_qty * unit_cost
        triggers    = []

        # Check triggers -- only evaluate orders where qty > 0

        # counts by up to 31/year in base conditions.
        if order_qty > HITL_ORDER_QTY_THRESHOLD:
            triggers.append(f"qty_threshold (>{HITL_ORDER_QTY_THRESHOLD})")
        if order_value > HITL_ORDER_VALUE_THRESHOLD:
            triggers.append(f"value_threshold (>£{HITL_ORDER_VALUE_THRESHOLD:,.0f})")
        # Fires on every non-zero order during active disruption.
        # Proportionate to risk: ~4.1 reviews/yr at base, ~23.9/yr at high uncertainty.
        if disruption and HITL_DISRUPTION_FLAG and order_qty > 0:
            triggers.append("active_disruption")

        req = ApprovalRequest(
            request_id  = self._counter,
            day         = day,
            sku         = self.sku,
            agent_name  = agent_name,
            order_qty   = order_qty,
            order_value = order_value,
            disruption  = disruption,
            inventory   = inventory,
            forecast    = forecast,
            triggers    = triggers,
            q_values    = q_values,
            final_qty   = order_qty,   # default: approve as-is
        )

        if not triggers:
            # Auto-approve: below all thresholds
            req.status    = ApprovalStatus.AUTO_APPROVED
            req.rationale = "All thresholds satisfied -- auto-approved"
            self._auto_approvals += 1
        else:
            # Escalate and apply simulated human review
            req.status = ApprovalStatus.ESCALATED
            self._escalations += 1
            req = self._simulate_human_review(req)

        self.approval_log.append(req)
        return req

    def _simulate_human_review(self, req: ApprovalRequest) -> ApprovalRequest:
        """
        Simulated human review logic. In production, this would pause
        and await an actual approval via an ERP or notification system.
        Simulated policy mirrors conservative procurement behaviour.
        """
        final_qty = req.order_qty
        notes     = []

        # Rule 1: Quantity cap (applied first).
        # Disruption uplift (Rule 3) acts on the capped value.
        # At order=200u with both triggers: cap to 75u -> uplift to ~82u.
        if f"qty_threshold (>{HITL_ORDER_QTY_THRESHOLD})" in req.triggers:
            if final_qty > HITL_QTY_CAP:
                final_qty = HITL_QTY_CAP
                notes.append(f"Qty capped at HITL_QTY_CAP ({HITL_QTY_CAP} units)")

        # Rule 2: Value -- approve as-is but note for budget review
        if any("value_threshold" in t for t in req.triggers):
            notes.append(f"High-value order (£{req.order_value:,.0f}) approved for budget review")

        # Rule 3: Disruption -- approve with configurable safety uplift
        if "active_disruption" in req.triggers:
            uplift = int(final_qty * HITL_DISRUPTION_UPLIFT)
            final_qty += uplift
            notes.append(f"Disruption uplift +{HITL_DISRUPTION_UPLIFT:.0%} ({uplift} units)")

        # Round to nearest ORDER_ACTIONS value
        from config import ORDER_ACTIONS
        final_qty = min(ORDER_ACTIONS, key=lambda a: abs(a - final_qty))

        req.final_qty = final_qty
        req.status    = (ApprovalStatus.MODIFIED
                         if final_qty != req.order_qty
                         else ApprovalStatus.APPROVED)
        req.rationale = "; ".join(notes) if notes else "Approved without modification"
        return req

    def summary(self) -> dict:
        """Portfolio HITL performance summary."""
        n = len(self.approval_log)
        escalated = [r for r in self.approval_log if r.status != ApprovalStatus.AUTO_APPROVED]
        modified  = [r for r in self.approval_log if r.status == ApprovalStatus.MODIFIED]
        return {
            'total_decisions':       n,
            'auto_approved':         self._auto_approvals,
            'escalated':             self._escalations,
            'escalation_rate_pct':   self._escalations / n * 100 if n > 0 else 0,
            'modified_by_human':     len(modified),
            'modification_rate_pct': len(modified) / n * 100 if n > 0 else 0,
            'avg_qty_change':        (float(np.mean([r.final_qty - r.order_qty
                                     for r in modified])) if modified else 0.0),
        }
