"""
Carbon-Profit Pareto Frontier Analysis -- 

Generates the multi-objective Pareto frontier between annual profit and
carbon footprint by sweeping the logistics agent's carbon penalty weight.
This directly addresses the EU Corporate Sustainability Reporting
Directive (CSRD, 2023) requirement for explicit carbon-cost trade-off
quantification in supply chain operations.

Method: Multi-objective parametric sweep
  For each carbon penalty weight w ∈ PARETO_CARBON_PENALTIES:
    1. Instantiate LogisticsAgent with CARBON_PENALTY = w
    2. Run PARETO_RUNS_PER_POINT simulation runs (same seed base)
    3. Record mean (profit, carbon) across runs
  
  Pareto-optimal points are those where no alternative achieves
  both higher profit AND lower carbon simultaneously.

Reference: Marler, R.T., Arora, J.S. (2004). Survey of multi-objective
optimisation methods for engineering. Structural and Multidisciplinary
Optimisation, 26(6), 369-395.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from typing import List, Dict, Tuple
from config import *

def run_pareto_sweep(rl_agent, carbon_penalties: List[float],
                     n_runs: int = 10,
                     uncertainty_level: str = 'base',
                     verbose: bool = True) -> List[Dict]:
    """
    Sweep carbon penalty weights to generate Pareto frontier data.

    For each carbon_penalty value:
      - Logistics agent uses that weight in its composite cost function
      - Inventory agent and all other components remain unchanged
      - Results averaged over n_runs matched simulation runs

    Returns list of result dicts, one per penalty weight.
    """
    from stochastic_env import StochasticEnvironment
    from forecasting_agent import AdaptiveForecaster
    from logistics_agent import OptimizedLogisticsAgent, create_shipment_request
    from simulation_runner import KPITracker
    from config import (SIMULATION_DAYS, RANDOM_SEED_BASE, INITIAL_INVENTORY,
                        ORDER_FIXED_COST, UNIT_COST, UNIT_PRICE,
                        HOLDING_COST_RATE, STOCKOUT_COST_RATE, SHIPPING_COSTS,
                        SHIPPING_DAYS, CARBON_FACTORS)

    results = []

    for penalty_idx, carbon_penalty in enumerate(carbon_penalties):
        run_profits  = []
        run_carbons  = []
        run_sls      = []
        run_modes    = []

        for run in range(n_runs):
            seed = RANDOM_SEED_BASE + run
            env  = StochasticEnvironment(uncertainty_level=uncertainty_level,
                                         seed=seed)
            forecaster = AdaptiveForecaster()

            # Custom logistics agent with this carbon penalty
            log_agent = OptimizedLogisticsAgent(carbon_penalty=carbon_penalty)
            kpi       = KPITracker()

            inventory      = float(INITIAL_INVENTORY)
            pending_orders = {}
            ship_req_id    = 0

            for _ in range(7):
                forecaster.update(float(env.sample_demand()))

            for day in range(SIMULATION_DAYS):
                disruption = env.advance_day()
                inventory += pending_orders.pop(day, 0)
                demand = float(env.sample_demand())
                forecaster.update(demand)
                demand_forecast = forecaster.forecast()
                pending_qty = sum(pending_orders.values())

                order_qty, _ = rl_agent.decide(
                    inventory, demand_forecast, pending_qty, disruption, day)

                shipping_cost = 0.0
                carbon        = 0.0
                shipping_mode = None
                if order_qty > 0:
                    lt = env.get_effective_lead_time()
                    pending_orders[day + lt] = (
                        pending_orders.get(day + lt, 0) + order_qty)
                    req = create_shipment_request(day, order_qty, inventory,
                                                  demand_forecast, ship_req_id)
                    ship_req_id += 1
                    req           = log_agent.route(req)
                    shipping_cost = req.cost
                    carbon        = req.carbon
                    shipping_mode = req.selected_mode

                units_sold = min(inventory, demand)
                inventory  = max(0.0, inventory - demand)

                order_cost    = ORDER_FIXED_COST if order_qty > 0 else 0.0
                holding_cost  = (inventory * UNIT_COST * HOLDING_COST_RATE) / DAYS_PER_YEAR
                unmet         = max(0.0, demand - units_sold)
                stockout_cost = unmet * UNIT_COST * STOCKOUT_COST_RATE

                kpi.record_day(day, inventory, demand, units_sold,
                               order_cost, holding_cost, stockout_cost,
                               shipping_cost, carbon, 0.0, shipping_mode)

            kpis = kpi.compute_kpis()
            run_profits.append(kpis['profit'])
            run_carbons.append(kpis['carbon_kgco2e'])
            run_sls.append(kpis['service_level'])

            # Collect mode distribution
            modes = kpis.get('shipping_mode_counts', {})
            run_modes.append(modes)

        # Aggregate mode distribution across runs
        agg_modes: Dict[str, int] = {}
        for md in run_modes:
            for m, c in md.items():
                agg_modes[m] = agg_modes.get(m, 0) + c

        result = {
            'carbon_penalty':    carbon_penalty,
            'mean_profit':       float(np.mean(run_profits)),
            'std_profit':        float(np.std(run_profits, ddof=1)),
            'mean_carbon':       float(np.mean(run_carbons)),
            'std_carbon':        float(np.std(run_carbons, ddof=1)),
            'mean_sl':           float(np.mean(run_sls)),
            'n_runs':            n_runs,
            'shipping_modes':    agg_modes,
        }
        results.append(result)

        if verbose:
            mode_str = ', '.join(f"{m}:{c}" for m, c in agg_modes.items())
            print(f"  [Pareto] CP={carbon_penalty:6.2f} | "
                  f"Profit=£{result['mean_profit']:>10,.0f} | "
                  f"Carbon={result['mean_carbon']:>6,.0f} kg | "
                  f"SL={result['mean_sl']:.1%} | "
                  f"Modes: {mode_str}")

    return results

def identify_pareto_front(results: List[Dict]) -> List[Dict]:
    """
    Identify Pareto-optimal points: no other point achieves both
    higher profit AND lower carbon simultaneously.
    """
    pareto = []
    for i, r in enumerate(results):
        dominated = False
        for j, s in enumerate(results):
            if i == j:
                continue
            # s dominates r if s has higher profit AND lower carbon
            if (s['mean_profit'] >= r['mean_profit'] and
                    s['mean_carbon'] <= r['mean_carbon'] and
                    (s['mean_profit'] > r['mean_profit'] or
                     s['mean_carbon'] < r['mean_carbon'])):
                dominated = True
                break
        if not dominated:
            pareto.append({**r, 'pareto_optimal': True})
        else:
            pareto.append({**r, 'pareto_optimal': False})
    return pareto

def compute_pareto_metrics(results: List[Dict]) -> Dict:
    """
    Summary metrics for the Pareto frontier.
    Includes trade-off rate: £ profit sacrificed per kg CO₂e saved.
    """
    pareto_pts = [r for r in results if r.get('pareto_optimal')]
    if len(pareto_pts) < 2:
        return {
            'n_pareto_points': len(pareto_pts),
            'trade_off_rate_gbp_per_kg': None,
            'carbon_range_kg': None,
            'profit_range_gbp': None,
            'max_profit_point': pareto_pts[0] if pareto_pts else None,
            'min_carbon_point': pareto_pts[0] if pareto_pts else None,
        }

    # Sort by carbon (ascending)
    sorted_p = sorted(pareto_pts, key=lambda r: r['mean_carbon'])

    # Trade-off rate between first and last Pareto point
    delta_profit = sorted_p[-1]['mean_profit'] - sorted_p[0]['mean_profit']
    delta_carbon = sorted_p[-1]['mean_carbon'] - sorted_p[0]['mean_carbon']
    trade_off    = abs(delta_profit / delta_carbon) if delta_carbon != 0 else 0.0

    return {
        'n_pareto_points':    len(pareto_pts),
        'max_profit_point':   max(pareto_pts, key=lambda r: r['mean_profit']),
        'min_carbon_point':   min(pareto_pts, key=lambda r: r['mean_carbon']),
        'trade_off_rate_gbp_per_kg': round(trade_off, 2),
        'carbon_range_kg':    round(sorted_p[-1]['mean_carbon'] - sorted_p[0]['mean_carbon'], 1),
        'profit_range_gbp':   round(abs(delta_profit), 1),
    }
