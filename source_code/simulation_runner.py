"""
Simulation Runner
Executes both rule-based heuristic baseline  and agentic 
control systems across 30 repeated simulation runs for statistical comparison
(
satisfying the pre-registered paired experimental design.

         Both systems run in identical stochastic environments; the distinction is
         fixed policy (ROP/EOQ) vs learned policy (Q-learning).

         a governance flag and a conservative order-quantity uplift.
         and disruption-period orders are reviewed before placement.
         not just log decisions passively.
         depletes before supplier stock, making scheduling affect service level.
"""

import numpy as np
from typing import List
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from config import *
from stochastic_env import StochasticEnvironment
from forecasting_agent import MovingAverageForecaster, AdaptiveForecaster
from inventory_agent import FixedReorderAgent, QLearningInventoryAgent
from production_agent import OptimizedScheduler, FIFOScheduler, generate_production_jobs
from logistics_agent import (DefaultLogisticsAgent, OptimizedLogisticsAgent,
                              create_shipment_request)
from governance_agent import GovernanceAgent
from drift_monitor import ModelDriftMonitor
from human_approval import HumanApprovalGateway

class KPITracker:
    """Tracks all performance KPIs across a simulation run."""

    def __init__(self):
        self.days: List[int] = []
        self.inventory_levels: List[float] = []
        self.demand_history:   List[float] = []
        self.units_sold_history: List[float] = []
        self.stockout_days:    List[int]   = []
        self.order_costs:      List[float] = []
        self.holding_costs:    List[float] = []
        self.stockout_costs:   List[float] = []
        self.shipping_costs:   List[float] = []
        self.carbon_emissions: List[float] = []
        self.production_costs: List[float] = []
        self.revenues:         List[float] = []
        self.n_orders    = 0
        self.total_demand    = 0.0
        self.total_fulfilled = 0.0
        # Shipping mode tracking
        self.mode_counts: dict = {}
        # M2: governance and HITL event counters (surfaced in compute_kpis output)
        self.gov_cap_count:        int = 0   # governance cap enforcements
        self.gov_drift_count:      int = 0   # drift alerts
        self.gov_ss_trigger_count: int = 0   # Holt safety-stock floor triggers
        self.hitl_review_count:    int = 0   # HITL reviews triggered
        self.hitl_modify_count:    int = 0   # HITL modifications made

    def record_day(self, day: int, inventory: float, demand: float,
                   units_sold: float, order_cost: float, holding_cost: float,
                   stockout_cost: float, shipping_cost: float,
                   carbon: float, production_cost: float,
                   shipping_mode: str = None):
        self.days.append(day)
        self.inventory_levels.append(inventory)
        self.demand_history.append(demand)
        self.units_sold_history.append(units_sold)
        self.stockout_days.append(1 if units_sold < demand else 0)
        self.order_costs.append(order_cost)
        self.holding_costs.append(holding_cost)
        self.stockout_costs.append(stockout_cost)
        self.shipping_costs.append(shipping_cost)
        self.carbon_emissions.append(carbon)
        self.production_costs.append(production_cost)
        self.revenues.append(units_sold * UNIT_PRICE)
        self.total_demand    += demand
        self.total_fulfilled += units_sold
        if shipping_mode:
            self.mode_counts[shipping_mode] = (
                self.mode_counts.get(shipping_mode, 0) + 1)

    def compute_kpis(self) -> dict:
        """Compute all KPIs for the simulation run."""
        total_cost    = (sum(self.order_costs) + sum(self.holding_costs) +
                         sum(self.stockout_costs) + sum(self.shipping_costs) +
                         sum(self.production_costs))
        total_revenue = sum(self.revenues)
        total_cogs    = self.total_fulfilled * UNIT_COST
        gross_profit  = total_revenue - total_cogs - total_cost

        service_level       = (self.total_fulfilled / self.total_demand
                               if self.total_demand > 0 else 0.0)
        avg_inventory       = float(np.mean(self.inventory_levels))
        avg_inventory_value = avg_inventory * UNIT_COST
        total_carbon        = sum(self.carbon_emissions)

        return {
            'service_level':        service_level,
            'avg_inventory':        avg_inventory,
            'avg_inventory_value':  avg_inventory_value,
            'total_cost':           total_cost,
            'total_revenue':        total_revenue,
            'profit':               gross_profit,
            'carbon_kgco2e':        total_carbon,
            'stockout_days':        sum(self.stockout_days),
            'stockout_rate':        float(np.mean(self.stockout_days)),
            'n_orders':             self.n_orders,
            'total_demand':         self.total_demand,
            'total_fulfilled':      self.total_fulfilled,
            'order_cost_total':     sum(self.order_costs),
            'holding_cost_total':   sum(self.holding_costs),
            'stockout_cost_total':  sum(self.stockout_costs),
            'shipping_cost_total':  sum(self.shipping_costs),
            'production_cost_total': sum(self.production_costs),
            'shipping_mode_counts': dict(self.mode_counts),
            # M2: governance and HITL event counts (verifiable oversight activity)
            'gov_cap_enforcements':   self.gov_cap_count,
            'gov_drift_alerts':       self.gov_drift_count,
            'gov_ss_triggers':        self.gov_ss_trigger_count,
            'hitl_reviews':           self.hitl_review_count,
            'hitl_modifications':     self.hitl_modify_count,
        }

def run_baseline_simulation(seed: int, uncertainty_level: str = 'base',
                            demand_distribution: str = 'poisson',
                            demand_lambda: int = None) -> dict:
    """
    Run baseline control system :
      - Moving-average forecasting (7-day window)
      - Fixed reorder point inventory control (ROP=75, EOQ=100)
      - FIFO production scheduling
      - Standard Class logistics (fixed mode)

    demand_lambda: optional override for OOD generalisation tests.
    """
    env        = StochasticEnvironment(uncertainty_level=uncertainty_level, seed=seed,
                                       demand_distribution=demand_distribution,
                                       demand_lambda=demand_lambda)
    forecaster = MovingAverageForecaster(window=7)
    inv_agent  = FixedReorderAgent()
    prod_agent = FIFOScheduler()
    log_agent  = DefaultLogisticsAgent()
    kpi        = KPITracker()
    rng        = np.random.default_rng(seed + PROD_SCHED_SEED_OFFSET)

    inventory      = float(INITIAL_INVENTORY)
    pending_orders = {}
    ship_req_id    = 0

    for _ in range(7):                         # warm-up forecaster
        forecaster.update(float(env.sample_demand()))

    for day in range(SIMULATION_DAYS):
        disruption = env.advance_day()
        inventory += pending_orders.pop(day, 0)

        demand          = float(env.sample_demand())
        forecaster.update(demand)
        demand_forecast = forecaster.forecast()
        pending_qty     = sum(pending_orders.values())

        order_qty, _ = inv_agent.decide(
            inventory, demand_forecast, pending_qty, disruption, day)

        shipping_cost = 0.0
        carbon        = 0.0
        shipping_mode = None
        if order_qty > 0:
            lt = env.get_effective_lead_time()
            pending_orders[day + lt] = pending_orders.get(day + lt, 0) + order_qty
            kpi.n_orders += 1
            req = create_shipment_request(day, order_qty, inventory,
                                          demand_forecast, ship_req_id)
            ship_req_id += 1
            req           = log_agent.route(req)
            shipping_cost = req.cost
            carbon        = req.carbon
            shipping_mode = req.selected_mode

        # Weekly production scheduling
        production_cost = 0.0
        if day % 7 == 0:
            jobs      = generate_production_jobs(day, demand_forecast, rng)
            scheduled = prod_agent.schedule(jobs, day)
            production_cost = sum(j.tardiness * j.unit_revenue * TARDINESS_PENALTY_RATE
                                  for j in scheduled)
            if prod_agent.schedule_log:
                production_cost += prod_agent.schedule_log[-1]['total_setup_cost']

        units_sold = min(inventory, demand)
        inventory  = max(0.0, inventory - demand)

        order_cost    = ORDER_FIXED_COST if order_qty > 0 else 0.0
        holding_cost  = (inventory * UNIT_COST * HOLDING_COST_RATE) / DAYS_PER_YEAR
        unmet         = max(0.0, demand - units_sold)
        stockout_cost = unmet * UNIT_COST * STOCKOUT_COST_RATE

        kpi.record_day(day, inventory, demand, units_sold,
                       order_cost, holding_cost, stockout_cost,
                       shipping_cost, carbon, production_cost, shipping_mode)

    return kpi.compute_kpis()

def run_agentic_simulation(seed: int, rl_agent: QLearningInventoryAgent,
                           uncertainty_level: str = 'base',
                           demand_distribution: str = 'poisson',
                           demand_lambda: int = None) -> dict:
    """
    Run agentic AI control system :
      - Adaptive Holt forecasting agent
      - Q-learning inventory replenishment agent
      - SPT-optimised production scheduling agent 
      - Cost-minimising logistics routing agent (multi-modal)
      - GovernanceAgent: logs + enforces policy limits
      - HumanApprovalGateway: reviews large/disruption orders
      - DriftMonitor: detects demand distribution shifts daily

    demand_lambda: optional override for OOD generalisation tests.
                   If None, uses StochasticEnvironment default for uncertainty_level.
    """
    env        = StochasticEnvironment(uncertainty_level=uncertainty_level, seed=seed,
                                       demand_distribution=demand_distribution,
                                       demand_lambda=demand_lambda)
    forecaster = AdaptiveForecaster()
    prod_agent = OptimizedScheduler()
    log_agent  = OptimizedLogisticsAgent()
    gov_agent  = GovernanceAgent("Agentic AI")
    hitl       = HumanApprovalGateway()
    drift_mon  = ModelDriftMonitor(sku='default') if DRIFT_MONITOR_ACTIVE else None
    kpi        = KPITracker()
    rng        = np.random.default_rng(seed + PROD_SCHED_SEED_OFFSET)

    inventory         = float(INITIAL_INVENTORY)
    pending_orders    = {}
    ship_req_id       = 0
    production_buffer = 0.0
    drift_active      = False
    demand_history_ag: list = []   # rolling window for demand regime detection

    for _ in range(7):
        d_w = float(env.sample_demand())
        forecaster.update(d_w)
        demand_history_ag.append(d_w)

    for day in range(SIMULATION_DAYS):
        disruption = env.advance_day()

        inventory += pending_orders.pop(day, 0)

        demand          = float(env.sample_demand())
        forecaster.update(demand)
        demand_forecast = forecaster.forecast()
        demand_history_ag.append(demand)
        if len(demand_history_ag) > DEMAND_RATE_WINDOW:
            demand_history_ag.pop(0)
        demand_rate = float(np.mean(demand_history_ag))   # regime signal
        pending_qty = sum(pending_orders.values())

        # Safety stock floor: uses theoretical demand variance (NegBin-corrected).
        # Holt forecast sigma. For Poisson: Var(D)=lambda. For NegBin: Var(D)=lambda+lambda^2/r=4*lambda.
        # Without this correction the SS floor underestimates required buffer for
        # NegBin conditions by ~50% (SS_NegBin = 2* SS_Poisson at same mean/LT/z).
        # Formula: SS = z * sqrt(Var_D * LT_mean) where Var_D = env.demand_variance.
        lt_mean_f = float(env.lt_mean)
        demand_var = env.demand_variance  # theoretical variance per day
        dyn_safety_stock = HOLT_SS_SERVICE_FACTOR * float(np.sqrt(demand_var * lt_mean_f))

        order_qty, audit = rl_agent.decide(
            inventory, demand_forecast, pending_qty, disruption, day,
            demand_rate=demand_rate)   # pass regime signal

        # If inventory + pending stock falls below the forecaster's dynamic safety
        # stock, boost the order to cover the shortfall. This is the mechanism by
        # which the Holt forecaster's error signal directly lifts service level.
        effective_stock = inventory + pending_qty
        if effective_stock < dyn_safety_stock and order_qty == 0:
            order_qty = max(ORDER_ACTIONS[1], int(dyn_safety_stock - effective_stock))
            kpi.gov_ss_trigger_count += 1   # M2: count Holt SS floor triggers
            gov_agent.log_decision(
                day, 'AdaptiveForecaster', 'dynamic_safety_stock_trigger',
                {'effective_stock': effective_stock, 'dyn_safety_stock': round(dyn_safety_stock, 1),
                 'sigma_forecast': round(forecaster.forecast_uncertainty(), 2)},
                order_qty,
                f"Holt safety stock floor triggered: stock={effective_stock:.0f} < SS={dyn_safety_stock:.1f}",
                None
            )

        if drift_active and order_qty > 0:
            uplift    = int(order_qty * DRIFT_SAFETY_UPLIFT)
            order_qty = min(order_qty + uplift, max(ORDER_ACTIONS))

        # Cap is derived from ORDER_ACTIONS 87.5th percentile so it fires in practice.
        if order_qty > GOVERNANCE_ORDER_CAP:
            order_qty = GOVERNANCE_ORDER_CAP
            kpi.gov_cap_count += 1   # M2: count enforcements
            gov_agent.log_decision(
                day, gov_agent.system_name, 'policy_enforcement',
                {'original_order': audit.get('action'), 'cap': GOVERNANCE_ORDER_CAP},
                order_qty,
                f"Order capped at governance policy limit ({GOVERNANCE_ORDER_CAP} units)",
                None
            )

        # Uses simulated review (auto-approves within thresholds; modifies if above).
        approval = hitl.evaluate(
            day, order_qty, disruption, inventory, demand_forecast,
            rl_agent.name, audit.get('q_values'))
        if approval.triggers:
            kpi.hitl_review_count += 1          # M2: count reviews
        if approval.final_qty != order_qty:
            kpi.hitl_modify_count += 1          # M2: count modifications
        order_qty = approval.final_qty   # Use HITL-reviewed quantity

        gov_agent.log_decision(
            day, rl_agent.name, 'inventory_replenishment',
            {'inventory': inventory, 'demand_forecast': demand_forecast,
             'pending_qty': pending_qty, 'disruption': disruption,
             'hitl_triggered': bool(approval.triggers),
             'drift_active': drift_active},
            order_qty,
            f"Q-learning greedy action: order {order_qty} units"
            + (f" [HITL: {approval.rationale}]" if approval.triggers else ""),
            audit.get('q_values')
        )
        gov_agent.detect_anomalies(inventory, demand, day)

        shipping_cost = 0.0
        carbon        = 0.0
        shipping_mode = None
        if order_qty > 0:
            lt = env.get_effective_lead_time()
            pending_orders[day + lt] = pending_orders.get(day + lt, 0) + order_qty
            kpi.n_orders += 1
            req = create_shipment_request(day, order_qty, inventory,
                                          demand_forecast, ship_req_id)
            ship_req_id += 1
            req           = log_agent.route(req)
            shipping_cost = req.cost
            carbon        = req.carbon
            shipping_mode = req.selected_mode

        production_cost = 0.0
        if day % 7 == 0:
            jobs      = generate_production_jobs(day, demand_forecast, rng)
            scheduled = prod_agent.schedule(jobs, day)
            production_cost = sum(j.tardiness * j.unit_revenue * TARDINESS_PENALTY_RATE
                                  for j in scheduled)
            if prod_agent.schedule_log:
                production_cost += prod_agent.schedule_log[-1]['total_setup_cost']

            weekly_forecast  = demand_forecast * 7
            buffer_produced  = min(
                weekly_forecast * PRODUCTION_FILL_RATE,
                PRODUCTION_BUFFER_MAX - production_buffer
            )
            buffer_produced  = max(0.0, buffer_produced)
            production_cost += buffer_produced * UNIT_COST * PRODUCTION_UNIT_COST_FACTOR
            production_buffer = min(production_buffer + buffer_produced,
                                    PRODUCTION_BUFFER_MAX)

        buffer_used  = min(production_buffer, max(0.0, demand - inventory))
        eff_inventory = inventory + buffer_used
        units_sold    = min(eff_inventory, demand)
        inventory     = max(0.0, inventory - (units_sold - buffer_used))
        production_buffer = max(0.0, production_buffer - buffer_used)

        # drift_active is updated HERE (end of day) and used NEXT day's order decision,
        # which is causal: we detect deteriorating SL from yesterday's observations
        # and respond with a safety uplift in tomorrow's order -- not today's.
        if drift_mon is not None:
            drift_event = drift_mon.update(day, units_sold, demand)
            if drift_event is not None:
                drift_active = True          # carries into NEXT day's order decision
                kpi.gov_drift_count += 1
                gov_agent.log_decision(
                    day, 'DriftMonitor', 'drift_alert',
                    {'sl_current': drift_event.sl_current,
                     'event_type': drift_event.event_type},
                    'DRIFT_DETECTED',
                    drift_event.message, None
                )
            else:
                drift_active = False         # reset when SL recovers

        order_cost    = ORDER_FIXED_COST if order_qty > 0 else 0.0
        holding_cost  = (inventory * UNIT_COST * HOLDING_COST_RATE) / DAYS_PER_YEAR
        unmet         = max(0.0, demand - units_sold)
        stockout_cost = unmet * UNIT_COST * STOCKOUT_COST_RATE

        kpi.record_day(day, inventory, demand, units_sold,
                       order_cost, holding_cost, stockout_cost,
                       shipping_cost, carbon, production_cost, shipping_mode)

    return kpi.compute_kpis()

def run_experiment(rl_agent: QLearningInventoryAgent,
                   uncertainty_level: str = 'base',
                   n_runs: int = SIMULATION_RUNS,
                   verbose: bool = True,
                   demand_distribution: str = 'poisson') -> tuple:
    """
    Paired comparative experiment (
    Each run shares the same random seed across baseline and agentic
    systems, ensuring matched stochastic conditions for the paired t-test.
    Returns (baseline_results_list, agentic_results_list).

    demand_distribution: 'poisson' (default) or 'negbin' for overdispersion robustness.
    """
    baseline_results = []
    agentic_results  = []

    for run in range(n_runs):
        seed = RANDOM_SEED_BASE + run
        b = run_baseline_simulation(seed, uncertainty_level, demand_distribution)
        a = run_agentic_simulation(seed, rl_agent, uncertainty_level, demand_distribution)
        baseline_results.append(b)
        agentic_results.append(a)
        if verbose and (run + 1) % 5 == 0:
            print(f"  [Experiment] Run {run+1:2d}/{n_runs}  "
                  f"[{uncertainty_level}/{demand_distribution}]  "
                  f"Baseline SL={b['service_level']:.1%}  "
                  f"Agentic SL={a['service_level']:.1%}")

    return baseline_results, agentic_results

def run_agentic_simulation_ablated(seed: int,
                                    rl_agent: QLearningInventoryAgent,
                                    uncertainty_level: str = 'base',
                                    ablate_forecaster: bool = False,
                                    ablate_inventory: bool = False,
                                    ablate_logistics: bool = False,
                                    ablate_production: bool = False) -> dict:
    """
    Single-agent ablation variant of run_agentic_simulation.
    Replaces one agent at a time with its baseline equivalent.
    All other agents remain at their full agentic configuration.
    Used by run_ablation_study() to measure marginal contributions.
    """
    env = StochasticEnvironment(uncertainty_level=uncertainty_level, seed=seed)

    # Each flag swaps one agent back to its baseline equivalent
    forecaster = MovingAverageForecaster(window=7) if ablate_forecaster else AdaptiveForecaster()
    inv_agent  = FixedReorderAgent()               if ablate_inventory  else None  # None = use rl_agent
    log_agent  = DefaultLogisticsAgent()            if ablate_logistics  else OptimizedLogisticsAgent()
    prod_agent = FIFOScheduler()                    if ablate_production else OptimizedScheduler()

    gov_agent  = GovernanceAgent("Ablation")
    hitl       = HumanApprovalGateway()
    drift_mon  = ModelDriftMonitor(sku='ablation') if DRIFT_MONITOR_ACTIVE else None
    kpi        = KPITracker()
    rng        = np.random.default_rng(seed + PROD_SCHED_SEED_OFFSET)

    inventory         = float(INITIAL_INVENTORY)
    pending_orders    = {}
    ship_req_id       = 0
    production_buffer = 0.0
    demand_history_ab: list = []   # rolling window for regime detection

    for _ in range(7):
        d_w = float(env.sample_demand())
        forecaster.update(d_w)
        demand_history_ab.append(d_w)

    for day in range(SIMULATION_DAYS):
        disruption    = env.advance_day()
        inventory    += pending_orders.pop(day, 0)
        demand        = float(env.sample_demand())
        forecaster.update(demand)
        demand_history_ab.append(demand)
        if len(demand_history_ab) > DEMAND_RATE_WINDOW:
            demand_history_ab.pop(0)
        demand_rate_ab = float(np.mean(demand_history_ab))
        demand_forecast = forecaster.forecast()
        pending_qty   = sum(pending_orders.values())

        # Holt dynamic safety stock (only active when forecaster is NOT ablated)
        dyn_safety_stock = (
            forecaster.forecast_safety_stock(float(env.lt_mean), HOLT_SS_SERVICE_FACTOR)
            if not ablate_forecaster else 0.0
        )

        # Inventory decision: use FixedReorder or Q-learning with regime signal
        if inv_agent is not None:
            order_qty, audit = inv_agent.decide(inventory, demand_forecast, pending_qty, disruption, day)
        else:
            order_qty, audit = rl_agent.decide(
                inventory, demand_forecast, pending_qty, disruption, day,
                demand_rate=demand_rate_ab)

        # Safety stock floor (only when Holt is active)
        effective_stock = inventory + pending_qty
        if effective_stock < dyn_safety_stock and order_qty == 0:
            order_qty = max(ORDER_ACTIONS[1], int(dyn_safety_stock - effective_stock))

        # Governance cap
        MAX_ORDER_CAP = GOVERNANCE_ORDER_CAP
        order_qty = min(order_qty, MAX_ORDER_CAP)

        # HITL
        approval  = hitl.evaluate(day, order_qty, disruption, inventory, demand_forecast,
                                   rl_agent.name, audit.get('q_values'))
        order_qty = approval.final_qty

        shipping_cost = 0.0; carbon = 0.0; shipping_mode = None
        if order_qty > 0:
            lt = env.get_effective_lead_time()
            pending_orders[day + lt] = pending_orders.get(day + lt, 0) + order_qty
            kpi.n_orders += 1
            req = create_shipment_request(day, order_qty, inventory, demand_forecast, ship_req_id)
            ship_req_id += 1
            req = log_agent.route(req)
            shipping_cost = req.cost; carbon = req.carbon; shipping_mode = req.selected_mode

        production_cost = 0.0
        if day % 7 == 0:
            jobs      = generate_production_jobs(day, demand_forecast, rng)
            scheduled = prod_agent.schedule(jobs, day)
            production_cost = sum(j.tardiness * j.unit_revenue * TARDINESS_PENALTY_RATE
                                  for j in scheduled)
            if prod_agent.schedule_log:
                production_cost += prod_agent.schedule_log[-1]['total_setup_cost']
            weekly_forecast = demand_forecast * 7
            buffer_produced = min(
                weekly_forecast * PRODUCTION_FILL_RATE,
                PRODUCTION_BUFFER_MAX - production_buffer
            )
            buffer_produced  = max(0.0, buffer_produced)
            production_cost += buffer_produced * UNIT_COST * PRODUCTION_UNIT_COST_FACTOR
            production_buffer = min(production_buffer + buffer_produced, PRODUCTION_BUFFER_MAX)

        if drift_mon:
            drift_mon.update(day, min(inventory + production_buffer, demand), demand)

        buffer_used = min(production_buffer, max(0.0, demand - inventory))
        units_sold  = min(inventory + buffer_used, demand)
        inventory   = max(0.0, inventory - (units_sold - buffer_used))
        production_buffer = max(0.0, production_buffer - buffer_used)

        order_cost   = ORDER_FIXED_COST if order_qty > 0 else 0.0
        holding_cost = (inventory * UNIT_COST * HOLDING_COST_RATE) / DAYS_PER_YEAR
        unmet        = max(0.0, demand - units_sold)
        stockout_cost = unmet * UNIT_COST * STOCKOUT_COST_RATE
        kpi.record_day(day, inventory, demand, units_sold, order_cost, holding_cost,
                       stockout_cost, shipping_cost, carbon, production_cost, shipping_mode)

    return kpi.compute_kpis()

def run_ablation_study(rl_agent: QLearningInventoryAgent,
                       uncertainty_level: str = 'base',
                       n_runs: int = None) -> dict:
    """
    Agent contribution ablation study.

    n_runs defaults to ABLATION_RUNS from config (30 runs).
    n=30 gives 80% power to detect ~2pp effects at sigma~=3pp (paired Wilcoxon).
    Previously hardcoded at n=15 which was statistically underpowered.

    Measures each agent's marginal SL contribution by disabling one agent at a time.
    'Full Agentic' = all agents active.
    Each ablation replaces exactly one agent with its baseline equivalent.
    Marginal contribution = Ablated_SL - Full_SL (positive = agent helps; negative = hurts).
    """
    import warnings; warnings.filterwarnings('ignore')
    if n_runs is None:
        n_runs = ABLATION_RUNS   # H2: derive from config, never hardcode

    configs = {
        'Full Agentic (all agents active)':  dict(),
        'Ablate Forecaster (-> 7d MA)':       dict(ablate_forecaster=True),
        'Ablate Inventory (-> ROP heuristic)': dict(ablate_inventory=True),
        'Ablate Logistics (-> fixed Standard)': dict(ablate_logistics=True),
        'Ablate Production (-> FIFO)':         dict(ablate_production=True),
    }

    results = {}
    for label, flags in configs.items():
        sls = [
            run_agentic_simulation_ablated(
                RANDOM_SEED_BASE + i, rl_agent, uncertainty_level, **flags
            )['service_level'] * 100
            for i in range(n_runs)
        ]
        results[label] = {
            'sl_mean': round(float(np.mean(sls)), 2),
            'sl_std':  round(float(np.std(sls, ddof=1)), 2),
            'n_runs':  n_runs,
        }

    full_sl = results['Full Agentic (all agents active)']['sl_mean']
    for label, res in results.items():
        # marginal_contribution_pp: positive means ablating that agent reduces SL
        # i.e. the agent contributes positively; negative means it hurts SL
        res['marginal_contribution_pp'] = round(res['sl_mean'] - full_sl, 2)

    return results

def run_ood_experiment(rl_agent: QLearningInventoryAgent,
                       n_runs: int = 20) -> dict:
    """
    Out-of-distribution (OOD) generalisation test.

    Tests the trained policy at demand lambdas outside its training envelope:
      OOD_LAMBDA_LOW  = DR_DEMAND_LAMBDA_RANGE[0] * 0.80  (20% below training min)
      OOD_LAMBDA_HIGH = DR_DEMAND_LAMBDA_RANGE[1] * 1.10  (10% above training max)

    Both conditions use paired seeds, Poisson demand, base uncertainty.
    A policy that truly generalises should outperform the heuristic baseline on
    OOD conditions, though by a smaller margin than in-distribution conditions.

    Returns dict with per-condition results including Cohen's d and p-value.
    """
    from scipy import stats as _stats
    import warnings; warnings.filterwarnings('ignore')

    ood_conditions = [
        ('OOD_Low  (lambda=%.1f, below training)' % OOD_LAMBDA_LOW,  int(round(OOD_LAMBDA_LOW))),
        ('OOD_High (lambda=%.1f, above training)' % OOD_LAMBDA_HIGH, int(round(OOD_LAMBDA_HIGH))),
    ]

    ood_results = {}
    for label, lam in ood_conditions:
        b_sls, a_sls = [], []
        for i in range(n_runs):
            seed = RANDOM_SEED_BASE + i
            b = run_baseline_simulation(seed, 'base', 'poisson', demand_lambda=lam)
            a = run_agentic_simulation(seed, rl_agent, 'base', 'poisson', demand_lambda=lam)
            b_sls.append(b['service_level'] * 100)
            a_sls.append(a['service_level'] * 100)

        diff = np.array(a_sls) - np.array(b_sls)
        _, p = _stats.wilcoxon(a_sls, b_sls)
        d = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0.0

        ood_results[label] = {
            'lambda':              lam,
            'in_training_range':   False,
            'baseline_sl_mean':    round(float(np.mean(b_sls)), 2),
            'baseline_sl_std':     round(float(np.std(b_sls, ddof=1)), 2),
            'agentic_sl_mean':     round(float(np.mean(a_sls)), 2),
            'agentic_sl_std':      round(float(np.std(a_sls, ddof=1)), 2),
            'delta_sl_pp':         round(float(np.mean(diff)), 2),
            'cohens_d':            round(d, 2),
            'p_value':             float(p),
            'wins':                int(np.sum(diff > 0)),
            'n_runs':              n_runs,
        }

    return ood_results

# ─── TIER 1 ADDITIONS ──────────────────────────────────────────────────────

def run_sS_simulation(seed: int, uncertainty_level: str,
                      demand_distribution: str = 'poisson',
                      demand_lambda: int = None,
                      s: int = None, S: int = None) -> dict:
    """
    (s,S) inventory policy: order up to S when on-hand+in-transit drops to or below s.
    This is the classical optimal policy for stationary demand (Silver et al. 2017).
    Default: s = REORDER_POINT_BASELINE, S = REORDER_POINT_BASELINE + ORDER_QUANTITY_EOQ.
    """
    from forecasting_agent import AdaptiveForecaster
    from production_agent import FIFOScheduler, generate_production_jobs
    s_val = s if s is not None else REORDER_POINT_BASELINE
    S_val = S if S is not None else REORDER_POINT_BASELINE + ORDER_QUANTITY_EOQ

    env = StochasticEnvironment(uncertainty_level, seed=seed,
                                demand_distribution=demand_distribution,
                                demand_lambda=demand_lambda)
    fc          = AdaptiveForecaster()
    prod_sS     = FIFOScheduler()
    rng_sS      = np.random.default_rng(seed + PROD_SCHED_SEED_OFFSET)
    kpi         = KPITracker()
    inventory   = float(INITIAL_INVENTORY)
    pending     = {}

    for _ in range(7):
        fc.update(float(env.sample_demand()))

    for day in range(SIMULATION_DAYS):
        env.advance_day()
        inventory += pending.pop(day, 0)
        demand = float(env.sample_demand())
        fc.update(demand)

        # (s,S): if effective inventory <= s, order up to S
        effective = inventory + sum(pending.values())
        order_qty = int(max(0, S_val - effective)) if effective <= s_val else 0
        order_qty = min(order_qty, GOVERNANCE_ORDER_CAP)

        if order_qty > 0:
            lt = env.get_effective_lead_time()
            pending[day + lt] = pending.get(day + lt, 0) + order_qty
            kpi.n_orders += 1

        units_sold = min(inventory, demand)
        stockout   = max(0.0, demand - inventory)
        inventory  = max(0.0, inventory - demand)

        revenue       = units_sold * UNIT_PRICE
        holding_cost  = inventory * UNIT_COST * HOLDING_COST_RATE / DAYS_PER_YEAR
        stockout_cost = stockout * UNIT_COST * STOCKOUT_COST_RATE
        order_cost    = ORDER_FIXED_COST if order_qty > 0 else 0.0
        shipping_cost = SHIPPING_COSTS.get('Standard Class', 5.0) * order_qty
        carbon_kg     = CARBON_FACTORS.get('Standard Class', 0.5) * order_qty

        kpi.total_demand    += demand
        kpi.total_fulfilled += units_sold
        kpi.revenues.append(revenue)
        kpi.order_costs.append(order_cost)
        kpi.holding_costs.append(holding_cost)
        kpi.stockout_costs.append(stockout_cost)
        kpi.shipping_costs.append(shipping_cost)
        ss_prod_cost = 0.0
        if day % 7 == 0:
            ss_fc_val = fc.forecast()
            ss_jobs   = generate_production_jobs(day, ss_fc_val, rng_sS)
            ss_sched  = prod_sS.schedule(ss_jobs, day)
            ss_prod_cost = sum(j.tardiness * j.unit_revenue * TARDINESS_PENALTY_RATE
                               for j in ss_sched)
            if prod_sS.schedule_log:
                ss_prod_cost += prod_sS.schedule_log[-1]['total_setup_cost']
        kpi.production_costs.append(ss_prod_cost)
        kpi.inventory_levels.append(inventory)
        kpi.stockout_days.append(1 if stockout > 0 else 0)
        kpi.carbon_emissions.append(carbon_kg)

    result = kpi.compute_kpis()
    result['policy'] = 'sS'
    result['s'] = s_val
    result['S'] = S_val
    return result


# ── Extended ablation: SL + cost + carbon per agent ──────────────────────

def run_extended_ablation(rl_agent: QLearningInventoryAgent,
                          uncertainty_level: str = 'base',
                          n_runs: int = None) -> dict:
    """
    Extended ablation study: reports SL, total cost AND carbon per agent removal.

    Each agent is the primary driver of its designated KPI:
      - RL Inventory Agent: service level (SL driver)
      - Logistics Agent:    shipping cost and carbon (cost/carbon driver)
      - Production Agent:   production scheduling cost (cost driver)
      - Forecaster:         safety stock calibration (marginal SL via SS floor)

    Marginal contribution = full_value - ablated_value (positive = agent helps).
    """
    import warnings; warnings.filterwarnings('ignore')
    if n_runs is None:
        n_runs = ABLATION_RUNS

    configs = {
        'Full Agentic (all agents active)':   dict(),
        'Ablate Forecaster (-> 7d MA)':        dict(ablate_forecaster=True),
        'Ablate Inventory (-> ROP heuristic)': dict(ablate_inventory=True),
        'Ablate Logistics (-> fixed Standard)':dict(ablate_logistics=True),
        'Ablate Production (-> FIFO)':         dict(ablate_production=True),
    }

    results = {}
    for label, flags in configs.items():
        sls, costs, carbons, shippings = [], [], [], []
        for i in range(n_runs):
            r = run_agentic_simulation_ablated(
                RANDOM_SEED_BASE + i, rl_agent, uncertainty_level, **flags)
            sls.append(r['service_level'] * 100)
            costs.append(r['total_cost'])
            carbons.append(r['carbon_kgco2e'])
            shippings.append(r.get('shipping_cost_total', 0))
        results[label] = {
            'sl_mean':       round(float(np.mean(sls)),      2),
            'sl_std':        round(float(np.std(sls, ddof=1)), 2),
            'cost_mean':     round(float(np.mean(costs)),     0),
            'carbon_mean':   round(float(np.mean(carbons)),   1),
            'shipping_mean': round(float(np.mean(shippings)), 0),
            'n_runs':        n_runs,
        }

    full = results['Full Agentic (all agents active)']
    for label, res in results.items():
        res['marginal_sl_pp']      = round(res['sl_mean']     - full['sl_mean'],     2)
        res['marginal_cost_gbp']   = round(full['cost_mean']  - res['cost_mean'],    0)
        res['marginal_carbon_kg']  = round(full['carbon_mean']- res['carbon_mean'],  1)
        res['marginal_ship_gbp']   = round(full['shipping_mean'] - res['shipping_mean'], 0)

    return results


# ── Multi-seed training stability ─────────────────────────────────────────

def run_multi_seed_stability(n_seeds: int = 5,
                             n_eval_runs: int = 10) -> dict:
    """
    Train n_seeds independent agents and report consistency of best checkpoint SL.

    Uses RANDOM_SEED_BASE + seed_idx * 1000 to ensure well-separated RNG states.
    Evaluates each on n_eval_runs paired seeds from the primary evaluation set.

    Returns: mean/std of best checkpoint val_SL across seeds, plus per-seed detail.
    """
    import warnings; warnings.filterwarnings('ignore')
    from scipy import stats as _stats

    seed_results = []
    for seed_idx in range(n_seeds):
        agent = QLearningInventoryAgent(seed=RANDOM_SEED_BASE + seed_idx * 1000)
        agent.train(lambda ul: StochasticEnvironment(ul, seed=RANDOM_SEED_BASE),
                    n_episodes=RL_EPISODES, verbose=False)

        # Evaluate best checkpoint on a small set of evaluation seeds
        sls = [
            run_agentic_simulation(RANDOM_SEED_BASE + i, agent, 'base', 'poisson')
            ['service_level'] * 100
            for i in range(n_eval_runs)
        ]
        seed_results.append({
            'seed_offset':      seed_idx * 1000,
            'best_val_sl':      round(agent._best_val_sl * 100, 3),
            'best_checkpoint':  agent.convergence_episode,
            'eval_sl_mean':     round(float(np.mean(sls)), 2),
            'eval_sl_std':      round(float(np.std(sls, ddof=1)), 2),
        })

    val_sls  = [r['best_val_sl']  for r in seed_results]
    eval_sls = [r['eval_sl_mean'] for r in seed_results]

    return {
        'n_seeds':          n_seeds,
        'val_sl_mean':      round(float(np.mean(val_sls)),  2),
        'val_sl_std':       round(float(np.std(val_sls, ddof=1)), 2),
        'eval_sl_mean':     round(float(np.mean(eval_sls)), 2),
        'eval_sl_std':      round(float(np.std(eval_sls, ddof=1)), 2),
        'min_val_sl':       round(min(val_sls),  2),
        'max_val_sl':       round(max(val_sls),  2),
        'per_seed':         seed_results,
        'interpretation':   (
            'Std <= 1.0pp confirms policy robustness to initialisation. '
            'Std > 2.0pp suggests need for ensembling or more training episodes.'
        ),
    }


# ── Simulation-optimised (s,S) on held-out validation seeds ──────────────

def find_optimal_sS_simulation(uncertainty_level: str,
                                demand_distribution: str = 'poisson',
                                validation_seeds: list = None,
                                s_range: range = None,
                                S_offset_range: range = None) -> dict:
    """
    Grid-search (s,S) parameters using HELD-OUT validation seeds only.

    Critical: validation_seeds must be DIFFERENT from the 30 evaluation seeds
    (RANDOM_SEED_BASE + 0..29) to prevent test-set leakage.
    Default validation seeds: 8000-8014 (15 seeds, separate pool).

    Returns best (s, S) and achieved SL on held-out set.
    """
    import warnings; warnings.filterwarnings('ignore')

    if validation_seeds is None:
        validation_seeds = list(range(8000, 8015))  # 15 held-out seeds
    if s_range is None:
        s_range = range(50, 275, 25)
    if S_offset_range is None:
        S_offset_range = range(50, 425, 50)

    best_sl = 0.0
    best_s, best_S = REORDER_POINT_BASELINE, REORDER_POINT_BASELINE + ORDER_QUANTITY_EOQ
    grid_results = []

    for s_val in s_range:
        for S_off in S_offset_range:
            S_val = s_val + S_off
            sls = [
                run_sS_simulation(seed, uncertainty_level, demand_distribution,
                                  s=s_val, S=S_val)['service_level'] * 100
                for seed in validation_seeds
            ]
            mean_sl = float(np.mean(sls))
            grid_results.append({'s': s_val, 'S': S_val, 'val_sl': round(mean_sl, 2)})
            if mean_sl > best_sl:
                best_sl, best_s, best_S = mean_sl, s_val, S_val

    return {
        'best_s':              best_s,
        'best_S':              best_S,
        'val_sl':              round(best_sl, 2),
        'validation_seeds':    validation_seeds,
        'n_grid_points':       len(grid_results),
        'top5':                sorted(grid_results, key=lambda x: -x['val_sl'])[:5],
    }


# ── Seasonal condition C5 ────────────────────────────────────────────────

def run_seasonal_condition(rl_agent: QLearningInventoryAgent,
                           n_runs: int = 30) -> dict:
    """
    C5 seasonal demand condition: weekly cycle + Q4 uplift.

    Tests agentic system generalisation beyond stationary demand.
    Uses same paired-seed design as primary conditions.
    Seasonal model: lambda * (1 + 0.30*sin(2*pi*day/7)) * Q4_factor(1.2).
    """
    import warnings; warnings.filterwarnings('ignore')
    from scipy import stats as _stats

    b_sls, a_sls, b_profits, a_profits = [], [], [], []
    for i in range(n_runs):
        seed = RANDOM_SEED_BASE + i
        env_b = StochasticEnvironment('base', seed=seed, demand_distribution='seasonal')
        env_a = StochasticEnvironment('base', seed=seed, demand_distribution='seasonal')

        # Run baseline using standard simulation but with seasonal env
        b = run_baseline_simulation(seed, 'base', 'seasonal')
        a = run_agentic_simulation(seed, rl_agent, 'base', 'seasonal')
        b_sls.append(b['service_level'] * 100)
        a_sls.append(a['service_level'] * 100)
        b_profits.append(b['profit'])
        a_profits.append(a['profit'])

    diff = np.array(a_sls) - np.array(b_sls)
    try:
        _stat, p = _stats.wilcoxon(a_sls, b_sls)
    except Exception:
        p = 1.0

    return {
        'condition':      'C5_seasonal_poisson',
        'description':    'Weekly cycle (amplitude 30%) + Q4 uplift (20%), Poisson base',
        'baseline_sl':    round(float(np.mean(b_sls)), 2),
        'agentic_sl':     round(float(np.mean(a_sls)), 2),
        'delta_sl':       round(float(np.mean(diff)),  2),
        'cohens_d':       round(float(np.mean(diff) / np.std(diff, ddof=1)), 2),
        'p_value':        float(p),
        'wins':           int(np.sum(diff > 0)),
        'n_runs':         n_runs,
        'baseline_profit':round(float(np.mean(b_profits)), 0),
        'agentic_profit': round(float(np.mean(a_profits)), 0),
    }


# ── Per-SKU forecaster comparison: ARIMA vs Holt ─────────────────────────

def run_forecaster_comparison(rl_agent: QLearningInventoryAgent,
                               n_runs: int = 20) -> dict:
    """
    Compare Holt vs ARIMA forecaster effect on service level per SKU.

    For each SKU, runs the full agentic simulation with:
      - AdaptiveForecaster (Holt double-exponential smoothing)
      - ARIMAForecaster    (AR(3) with first differencing)

    Reports SL difference. Expected: ARIMA ~ Holt under stationary demand;
    ARIMA gains materialise under non-stationary/trended conditions.
    """
    import warnings; warnings.filterwarnings('ignore')
    from forecasting_agent import AdaptiveForecaster, ARIMAForecaster

    def run_with_forecaster(seed, forecaster_cls, sku_params):
        """Run one simulation with a specific forecaster class injected."""
        from config import (SIMULATION_DAYS, INITIAL_INVENTORY, ORDER_ACTIONS,
                            GOVERNANCE_ORDER_CAP, HOLT_SS_SERVICE_FACTOR,
                            DRIFT_MONITOR_ACTIVE, PROD_SCHED_SEED_OFFSET,
                            DEMAND_RATE_WINDOW, TARDINESS_PENALTY_RATE,
                            DRIFT_SAFETY_UPLIFT, ORDER_FIXED_COST,
                            HOLDING_COST_RATE, STOCKOUT_COST_RATE, DAYS_PER_YEAR)

        env = StochasticEnvironment('base', seed=seed,
                                    demand_lambda=sku_params['demand_lambda'],
                                    lead_time_mean=sku_params['lead_time_mean'])
        forecaster = forecaster_cls()
        kpi = KPITracker()
        inv = float(sku_params['initial_inventory'])
        pending = {}
        demand_hist = []
        rng = np.random.default_rng(seed + PROD_SCHED_SEED_OFFSET)

        for _ in range(7):
            d = float(env.sample_demand())
            forecaster.update(d)
            demand_hist.append(d)

        for day in range(SIMULATION_DAYS):
            env.advance_day()
            inv += pending.pop(day, 0)
            demand = float(env.sample_demand())
            forecaster.update(demand)
            demand_hist.append(demand)
            if len(demand_hist) > DEMAND_RATE_WINDOW:
                demand_hist.pop(0)
            demand_rate = float(np.mean(demand_hist))
            demand_fc = forecaster.forecast()
            pending_qty = sum(pending.values())

            demand_var = env.demand_variance
            dyn_ss = HOLT_SS_SERVICE_FACTOR * float(np.sqrt(demand_var * env.lt_mean))

            order_qty, _ = rl_agent.decide(inv, demand_fc, pending_qty,
                                           env.disruption_active, day,
                                           demand_rate=demand_rate)

            eff_stock = inv + pending_qty
            if eff_stock < dyn_ss and order_qty == 0:
                order_qty = max(ORDER_ACTIONS[1], int(dyn_ss - eff_stock))

            order_qty = min(order_qty, GOVERNANCE_ORDER_CAP)

            ship_cost = 0.0
            carbon = 0.0
            if order_qty > 0:
                lt = env.get_effective_lead_time()
                pending[day + lt] = pending.get(day + lt, 0) + order_qty
                kpi.n_orders += 1
                from config import SHIPPING_COSTS, CARBON_FACTORS
                ship_cost = SHIPPING_COSTS['Second Class'] * order_qty
                carbon = CARBON_FACTORS['Second Class'] * order_qty

            units_sold = min(inv, demand)
            inv = max(0.0, inv - demand)

            oc = ORDER_FIXED_COST if order_qty > 0 else 0.0
            hc = (inv * sku_params.get('unit_cost', 141.23) * HOLDING_COST_RATE) / DAYS_PER_YEAR
            sc = max(0.0, demand - units_sold) * sku_params.get('unit_cost', 141.23) * STOCKOUT_COST_RATE

            kpi.record_day(day, inv, demand, units_sold, oc, hc, sc, ship_cost, carbon, 0.0)
            kpi.total_demand += demand
            kpi.total_fulfilled += units_sold

        return kpi.compute_kpis()

    from config import PRODUCTS
    results = {}

    for sku_name, sku_params in PRODUCTS.items():
        holt_sls = []
        arima_sls = []
        for i in range(n_runs):
            seed = RANDOM_SEED_BASE + i
            holt_sls.append( run_with_forecaster(seed, AdaptiveForecaster, sku_params)['service_level'] * 100)
            arima_sls.append(run_with_forecaster(seed, ARIMAForecaster,    sku_params)['service_level'] * 100)

        diff = np.array(arima_sls) - np.array(holt_sls)
        results[sku_name] = {
            'holt_sl_mean':   round(float(np.mean(holt_sls)),  2),
            'arima_sl_mean':  round(float(np.mean(arima_sls)), 2),
            'delta_sl_pp':    round(float(np.mean(diff)),       2),
            'demand_lambda':  sku_params['demand_lambda'],
            'n_runs':         n_runs,
            'interpretation': (
                'ARIMA ~ Holt under stationary demand (expected).'
                if abs(np.mean(diff)) < 0.5 else
                f'ARIMA {"better" if np.mean(diff) > 0 else "worse"} than Holt by {abs(np.mean(diff)):.1f}pp.'
            ),
        }

    return results




# ── Per-SKU forecaster comparison: ARIMA vs Holt linked to SL ─────────────

def run_per_sku_forecaster_comparison(rl_agent,
                                       n_runs: int = 20) -> dict:
    """
    Compare Holt vs ARIMA forecaster effect on SL per SKU (linked to SL outcome).

    Runs agentic simulation with identical RL policy but different forecasters.
    Expected: ARIMA ~ Holt under stationary i.i.d. demand; ARIMA gains materialise
    under non-stationary/trended conditions (bicycles, golf with seasonal patterns).

    Each SKU uses its own demand_lambda so low/high demand SKUs are compared fairly.
    """
    import warnings; warnings.filterwarnings('ignore')
    from forecasting_agent import AdaptiveForecaster, ARIMAForecaster
    from config import (PRODUCTS, SIMULATION_DAYS, ORDER_ACTIONS, GOVERNANCE_ORDER_CAP,
                        HOLT_SS_SERVICE_FACTOR, ORDER_FIXED_COST, HOLDING_COST_RATE,
                        STOCKOUT_COST_RATE, DAYS_PER_YEAR, DEMAND_RATE_WINDOW,
                        PROD_SCHED_SEED_OFFSET, SHIPPING_COSTS, CARBON_FACTORS)

    def _sim_with_forecaster(seed, fc_cls, sku_p):
        env = StochasticEnvironment('base', seed=seed,
                                    demand_lambda=sku_p['demand_lambda'],
                                    lead_time_mean=sku_p['lead_time_mean'])
        fc  = fc_cls()
        kpi = KPITracker()
        inv = float(sku_p['initial_inventory'])
        pending = {}
        d_hist  = []

        for _ in range(7):
            d = float(env.sample_demand())
            fc.update(d); d_hist.append(d)

        for day in range(SIMULATION_DAYS):
            env.advance_day()
            inv += pending.pop(day, 0)
            demand = float(env.sample_demand())
            fc.update(demand); d_hist.append(demand)
            if len(d_hist) > DEMAND_RATE_WINDOW: d_hist.pop(0)
            dr     = float(np.mean(d_hist))
            fc_val = fc.forecast()
            pq     = sum(pending.values())

            dyn_ss   = HOLT_SS_SERVICE_FACTOR * float(np.sqrt(env.demand_variance * env.lt_mean))
            oq, _    = rl_agent.decide(inv, fc_val, pq, env.disruption_active, day, demand_rate=dr)
            if (inv + pq) < dyn_ss and oq == 0:
                oq = max(ORDER_ACTIONS[1], int(dyn_ss - (inv + pq)))
            oq = min(oq, GOVERNANCE_ORDER_CAP)

            sc = 0.0; co = 0.0
            if oq > 0:
                lt = env.get_effective_lead_time()
                pending[day + lt] = pending.get(day + lt, 0) + oq
                kpi.n_orders += 1
                sc = SHIPPING_COSTS.get('Second Class', 12.0) * oq
                co = CARBON_FACTORS.get('Second Class', 0.45) * oq

            sold = min(inv, demand); inv = max(0.0, inv - demand)
            uc   = sku_p.get('unit_cost', 141.23)
            kpi.record_day(day, inv, demand, sold,
                           ORDER_FIXED_COST if oq > 0 else 0.0,
                           (inv * uc * HOLDING_COST_RATE) / DAYS_PER_YEAR,
                           max(0.0, demand - sold) * uc * STOCKOUT_COST_RATE,
                           sc, co, 0.0)
            kpi.total_demand += demand; kpi.total_fulfilled += sold

        return kpi.compute_kpis()

    results = {}
    for sku_name, sku_p in PRODUCTS.items():
        h_sls, a_sls = [], []
        for i in range(n_runs):
            seed = RANDOM_SEED_BASE + i
            h_sls.append(_sim_with_forecaster(seed, AdaptiveForecaster, sku_p)['service_level'] * 100)
            a_sls.append(_sim_with_forecaster(seed, ARIMAForecaster,    sku_p)['service_level'] * 100)

        diff = np.array(a_sls) - np.array(h_sls)
        interp = ('ARIMA ~ Holt (stationary demand, difference < 0.5pp)'
                  if abs(float(np.mean(diff))) < 0.5 else
                  f'ARIMA {"better" if float(np.mean(diff)) > 0 else "worse"} by {abs(float(np.mean(diff))):.1f}pp')
        results[sku_name] = {
            'holt_sl':    round(float(np.mean(h_sls)), 2),
            'arima_sl':   round(float(np.mean(a_sls)), 2),
            'delta_pp':   round(float(np.mean(diff)),  2),
            'lambda':     sku_p['demand_lambda'],
            'n_runs':     n_runs,
            'interpretation': interp,
        }
    return results

def run_demand_rate_bin_ablation(rl_agent_with: object,
                                 uncertainty_level: str = 'base',
                                 n_runs: int = 20) -> dict:
    """
    Tier 1 ablation: compare agentic system WITH vs WITHOUT demand_rate_bin.
    The 'without' agent is the same trained Q-table but the 5th state dimension
    is forced to bin 1 (mid-regime) for every decision, effectively disabling
    the regime-aware encoding.

    Returns dict of {lambda: {with_sl, without_sl, delta}} for key lambda values.
    """
    results = {}
    test_lambdas = [6, 8, 10, 15, 25]

    for lam in test_lambdas:
        with_sls, without_sls = [], []
        for i in range(n_runs):
            seed = RANDOM_SEED_BASE + i
            # WITH rate_bin: normal decide()
            a_with = run_agentic_simulation(seed, rl_agent_with,
                                            uncertainty_level, 'poisson',
                                            demand_lambda=lam)
            with_sls.append(a_with['service_level'] * 100)

            # WITHOUT rate_bin: override demand_rate=None forces bin 1 (mid)
            # by passing demand_rate=9.0 (boundary between low and mid bins),
            # ensuring rate_bin=1 always -- equivalent to no regime awareness
            env_wo = StochasticEnvironment(uncertainty_level, seed=seed,
                                           demand_lambda=lam)
            fc_wo  = __import__('forecasting_agent').AdaptiveForecaster()
            inv_wo = float(INITIAL_INVENTORY)
            po_wo  = {}
            dem_wo = ful_wo = 0.0
            hist_wo = []

            for _ in range(7):
                dw = float(env_wo.sample_demand())
                fc_wo.update(dw); hist_wo.append(dw)

            for day in range(SIMULATION_DAYS):
                env_wo.advance_day()
                inv_wo += po_wo.pop(day, 0)
                d = float(env_wo.sample_demand())
                fc_wo.update(d); hist_wo.append(d)
                if len(hist_wo) > DEMAND_RATE_WINDOW: hist_wo.pop(0)
                pq = sum(po_wo.values())
                # Force demand_rate=None -> default mid bin (rate_bin=1)
                oq, _ = rl_agent_with.decide(
                    inv_wo, fc_wo.forecast(), pq,
                    env_wo.disruption_active, day,
                    demand_rate=None)  # None -> default mid bin
                oq = min(oq, GOVERNANCE_ORDER_CAP)
                if oq > 0:
                    lt = env_wo.get_effective_lead_time()
                    po_wo[day + lt] = po_wo.get(day + lt, 0) + oq
                ful_wo += min(inv_wo, d)
                inv_wo  = max(0.0, inv_wo - d)
                dem_wo += d

            sl_wo = ful_wo / dem_wo * 100 if dem_wo > 0 else 100.0
            without_sls.append(sl_wo)

        import numpy as _np
        from scipy.stats import wilcoxon as _wil
        delta = _np.mean(with_sls) - _np.mean(without_sls)
        try:
            _, p = _wil(with_sls, without_sls)
        except Exception:
            p = 1.0
        results[lam] = {
            'lambda':       lam,
            'with_sl':      round(float(_np.mean(with_sls)), 2),
            'without_sl':   round(float(_np.mean(without_sls)), 2),
            'delta_pp':     round(float(delta), 2),
            'p_value':      float(p),
            'n_runs':       n_runs,
        }
    return results

def run_extended_lambda_sweep(rl_agent: object, n_runs: int = 20) -> dict:
    """
    Tier 1: Extended OOD sweep from lambda=3 to lambda=32.
    Returns per-lambda results including in_training_range flag.
    """
    import numpy as _np
    from scipy.stats import wilcoxon as _wil

    lambdas = [3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 27, 29, 32]
    results = {}
    for lam in lambdas:
        b_s, a_s = [], []
        for i in range(n_runs):
            seed = RANDOM_SEED_BASE + i
            b = run_baseline_simulation(seed, 'base', 'poisson', demand_lambda=lam)
            a = run_agentic_simulation(seed, rl_agent, 'base', 'poisson', demand_lambda=lam)
            b_s.append(b['service_level']*100)
            a_s.append(a['service_level']*100)
        diff = _np.array(a_s) - _np.array(b_s)
        try:
            _, p = _wil(a_s, b_s)
        except Exception:
            p = 1.0
        d = float(_np.mean(diff)/_np.std(diff, ddof=1)) if _np.std(diff, ddof=1) > 0 else 0.0
        in_range = DR_DEMAND_LAMBDA_RANGE[0] <= lam <= DR_DEMAND_LAMBDA_RANGE[1]
        ceiling  = float(_np.mean(b_s)) >= 99.5
        results[lam] = {
            'lambda':            lam,
            'in_training_range': in_range,
            'ceiling_effect':    ceiling,
            'baseline_sl':       round(float(_np.mean(b_s)), 2),
            'agentic_sl':        round(float(_np.mean(a_s)), 2),
            'delta_sl_pp':       round(float(_np.mean(diff)), 2),
            'cohens_d':          round(d, 2),
            'p_value':           float(p),
            'wins':              int(_np.sum(diff > 0)),
            'n_runs':            n_runs,
        }
    return results
