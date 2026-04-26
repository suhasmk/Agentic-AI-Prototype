"""
Portfolio Manager -- 
Agentic AI Prototype v2

Orchestrates simultaneous inventory management across all 5 DataCo SKUs,
integrating all gap extensions in a single portfolio simulation:

  
  
  
  
  
  

This is architecturally equivalent to a real warehouse management system
(WMS) coordinating replenishment across a product portfolio, and
directly addresses the "single-SKU limitation" identified in S.8.4
of the v10 report.

Scope note: The portfolio simulation extends but does not replace the
primary single-SKU experiment (Objectives 1-8). It provides portfolio-
level evidence for the broader applicability of the agentic AI approach.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from typing import Dict, List
from config import (PRODUCTS, SIMULATION_DAYS, RANDOM_SEED_BASE,
                    ORDER_FIXED_COST, HOLDING_COST_RATE, STOCKOUT_COST_RATE,
                    SKU_DEMAND_BINS, SIMULATION_RUNS, PORTFOLIO_RL_EPISODES,
                    DAYS_PER_YEAR)
from stochastic_env import StochasticEnvironment
from forecasting_agent import select_forecaster
from inventory_agent import QLearningInventoryAgent, FixedReorderAgent, InventoryState
from logistics_agent import (OptimizedLogisticsAgent, DefaultLogisticsAgent,
                              create_shipment_request)
from governance_agent import GovernanceAgent
from consolidated_shipping import ConsolidatedShippingManager, PendingOrder
from human_approval import HumanApprovalGateway
from drift_monitor import PortfolioDriftMonitor
from supplier_reliability import SupplierReliabilityFeed

class SKUSimulator:
    """Per-SKU simulation state container."""
    def __init__(self, sku_id: str, params: dict, seed: int,
                 rl_agent=None, is_agentic: bool = True):
        self.sku_id   = sku_id
        self.params   = params
        self.is_agentic = is_agentic

        # Environment
        self.env = StochasticEnvironment(
            uncertainty_level='base', seed=seed,
            unit_cost=params['unit_cost'],
            demand_lambda=params['demand_lambda'],
            lead_time_mean=params['lead_time_mean'])

        # Forecaster (
        self.forecaster = select_forecaster(params)

        # Inventory agent
        if is_agentic:
            self.inv_agent = rl_agent  # shared or dedicated Q-table
        else:
            self.inv_agent = FixedReorderAgent()
            self.inv_agent.reorder_point  = params['reorder_point']
            self.inv_agent.order_quantity = params['order_quantity']

        # Logistics agent
        if is_agentic:
            self.log_agent = OptimizedLogisticsAgent()
        else:
            self.log_agent = DefaultLogisticsAgent()

        # Human approval gateway 
        self.hitl = HumanApprovalGateway(
            sku=sku_id, eoq=params['order_quantity'])

        # State
        self.inventory      = float(params['initial_inventory'])
        self.pending_orders: Dict[int, float] = {}
        self.ship_req_id    = 0

        # KPI accumulators
        self.total_demand    = 0.0
        self.total_fulfilled = 0.0
        self.total_revenue   = 0.0
        self.total_cost      = 0.0
        self.total_carbon    = 0.0
        self.n_orders        = 0
        self.stockout_days   = 0
        self.daily_records: List[dict] = []
        self.mode_counts: Dict[str, int] = {}

    def step(self, day: int, disruption: bool,
             reliability_feed: SupplierReliabilityFeed,
             shipping_manager: ConsolidatedShippingManager,
             drift_monitor: PortfolioDriftMonitor,
             gov_agent: GovernanceAgent) -> dict:
        """Advance simulation by one day for this SKU."""

        # Apply lead-time-in-transit arrivals
        self.inventory += self.pending_orders.pop(day, 0)

        # Sample demand
        demand          = float(self.env.sample_demand())
        self.forecaster.update(demand)
        demand_forecast = self.forecaster.forecast()
        pending_qty     = sum(self.pending_orders.values())

        # Get supplier lead time adjusted for reliability 
        region     = self.params['supplier_region']
        lt_mult    = reliability_feed.get_lead_time_multiplier(region)

        # Inventory decision
        if self.is_agentic:
            order_qty, audit = self.inv_agent.decide(
                self.inventory, demand_forecast, pending_qty, disruption, day)
        else:
            order_qty, audit = self.inv_agent.decide(
                self.inventory, demand_forecast, pending_qty, disruption, day)

        # Human approval gateway 
        unit_cost = self.params['unit_cost']
        approval  = self.hitl.evaluate(
            day, order_qty, disruption,
            self.inventory, demand_forecast,
            agent_name=getattr(self.inv_agent, 'name', 'Agent'),
            unit_cost=unit_cost)
        final_qty = approval.final_qty

        shipping_cost = 0.0
        carbon        = 0.0
        shipping_mode = None

        if final_qty > 0:
            lt = self.env.get_effective_lead_time()
            # Apply supplier reliability adjustment to lead time
            lt = max(1, int(lt * lt_mult))
            arrival = day + lt
            self.pending_orders[arrival] = (
                self.pending_orders.get(arrival, 0) + final_qty)
            self.n_orders += 1

            req = create_shipment_request(
                day, final_qty, self.inventory,
                demand_forecast, self.ship_req_id)
            self.ship_req_id += 1
            req = self.log_agent.route(req)

            # Register with consolidated shipping manager 
            pending = PendingOrder(
                sku             = self.sku_id,
                day             = day,
                order_qty       = final_qty,
                order_value     = final_qty * unit_cost,
                supplier_region = region,
                shipping_mode   = req.selected_mode,
                base_cost       = req.cost,
                base_carbon     = req.carbon,
                weight_kg       = self.params['weight_kg'] * final_qty,
                urgency_days    = req.delivery_days,
                req_id          = self.ship_req_id,
            )
            shipping_manager.register_order(pending)
            # Actual costs updated after consolidation; use base for now
            shipping_cost = req.cost
            carbon        = req.carbon
            shipping_mode = req.selected_mode
            self.mode_counts[shipping_mode] = (
                self.mode_counts.get(shipping_mode, 0) + 1)

        # Fulfil demand
        units_sold     = min(self.inventory, demand)
        self.inventory = max(0.0, self.inventory - demand)

        # Cost accounting
        order_cost    = ORDER_FIXED_COST if final_qty > 0 else 0.0
        holding_cost  = (self.inventory * unit_cost * HOLDING_COST_RATE) / DAYS_PER_YEAR
        unmet         = max(0.0, demand - units_sold)
        stockout_cost = unmet * unit_cost * STOCKOUT_COST_RATE
        revenue       = units_sold * self.params['unit_price']
        cogs          = units_sold * unit_cost
        day_cost      = order_cost + holding_cost + stockout_cost + shipping_cost
        profit_today  = revenue - cogs - day_cost

        if units_sold < demand:
            self.stockout_days += 1

        # Drift monitor update 
        drift_event = drift_monitor.update(
            self.sku_id, day, units_sold, demand)
        if drift_event:
            gov_agent.log_decision(
                day, 'DriftMonitor', 'drift_detected',
                {'sku': self.sku_id, 'rolling_sl': drift_event.sl_current},
                0, drift_event.message)

        # Accumulate KPIs
        self.total_demand    += demand
        self.total_fulfilled += units_sold
        self.total_revenue   += revenue
        self.total_cost      += day_cost
        self.total_carbon    += carbon

        self.daily_records.append({
            'day':          day,
            'inventory':    round(self.inventory, 2),
            'demand':       demand,
            'units_sold':   units_sold,
            'order_qty':    final_qty,
            'shipping_mode': shipping_mode,
            'profit':       round(profit_today, 2),
        })

        return {
            'demand': demand, 'units_sold': units_sold,
            'order_qty': final_qty, 'shipping_mode': shipping_mode,
            'shipping_cost': shipping_cost, 'carbon': carbon,
        }

    def kpis(self) -> dict:
        """Compute final SKU-level KPIs."""
        sl      = (self.total_fulfilled / self.total_demand
                   if self.total_demand > 0 else 0.0)
        cogs    = self.total_fulfilled * self.params['unit_cost']
        profit  = self.total_revenue - cogs - self.total_cost
        return {
            'sku':              self.sku_id,
            'service_level':    sl,
            'profit':           profit,
            'total_cost':       self.total_cost,
            'carbon_kgco2e':    self.total_carbon,
            'stockout_days':    self.stockout_days,
            'n_orders':         self.n_orders,
            'shipping_modes':   dict(self.mode_counts),
            'hitl_summary':     self.hitl.summary(),
        }

def run_portfolio_simulation(rl_agents: Dict[str, QLearningInventoryAgent],
                              seed: int,
                              is_agentic: bool = True,
                              verbose: bool = False) -> dict:
    """
    Run one complete portfolio simulation across all 5 SKUs.

    Integrations active:
      
      
      
      
      
      

    Returns portfolio-level KPI dict.
    """
    from config import PRODUCTS

    gov_agent      = GovernanceAgent("Portfolio-Agentic" if is_agentic else "Portfolio-Baseline")
    shipping_mgr   = ConsolidatedShippingManager()
    drift_mon      = PortfolioDriftMonitor()
    reliability    = SupplierReliabilityFeed(seed=seed)

    # Register all SKUs with drift monitor
    for sku_id in PRODUCTS:
        drift_mon.register_sku(sku_id)

    # Instantiate per-SKU simulators
    simulators: Dict[str, SKUSimulator] = {}
    for i, (sku_id, params) in enumerate(PRODUCTS.items()):
        agent = rl_agents.get(sku_id) if is_agentic else None
        sim   = SKUSimulator(sku_id, params, seed=seed + i * 10,
                             rl_agent=agent, is_agentic=is_agentic)
        # Warm-up forecaster
        for _ in range(7):
            sim.forecaster.update(float(sim.env.sample_demand()))
        simulators[sku_id] = sim

    # Main simulation loop
    for day in range(SIMULATION_DAYS):
        # Advance supplier reliability 
        rel_scores = reliability.advance_day(day)

        # Determine disruption for each SKU (partially correlated via env)
        for sku_id, sim in simulators.items():
            disruption = sim.env.advance_day()
            # Supplier reliability adds extra disruption probability
            if not disruption:
                region   = sim.params['supplier_region']
                extra_dp = reliability.get_disruption_adjustment(region)
                # Use the SKU's own env rng for consistency with the simulation stream
                disruption = float(sim.env.rng.random()) < extra_dp

            sim.step(day, disruption, reliability, shipping_mgr,
                     drift_mon, gov_agent)

        # Dispatch consolidated shipments for today 
        shipping_mgr.dispatch_day(day)

    # Collect results
    sku_kpis = {sku_id: sim.kpis() for sku_id, sim in simulators.items()}

    # Portfolio aggregates
    total_profit   = sum(k['profit']       for k in sku_kpis.values())
    total_carbon   = sum(k['carbon_kgco2e'] for k in sku_kpis.values())
    total_cost     = sum(k['total_cost']   for k in sku_kpis.values())
    total_stockout = sum(k['stockout_days'] for k in sku_kpis.values())
    portfolio_sl   = np.mean([k['service_level'] for k in sku_kpis.values()])

    consolidation  = shipping_mgr.summary()
    drift_summary  = drift_mon.portfolio_summary()

    return {
        'per_sku':             sku_kpis,
        'portfolio_profit':    total_profit,
        'portfolio_carbon':    total_carbon,
        'portfolio_cost':      total_cost,
        'portfolio_sl':        portfolio_sl,
        'portfolio_stockout':  total_stockout,
        'consolidation':       consolidation,
        'drift':               drift_summary,
        'supplier_reliability': reliability.summary(),
        'is_agentic':          is_agentic,
    }

def run_portfolio_experiment(rl_agents: Dict[str, QLearningInventoryAgent] = None,
                              n_runs: int = SIMULATION_RUNS,
                              verbose: bool = True) -> tuple:
    """
    Paired portfolio experiment: 30 runs baseline vs agentic.
    Returns (baseline_results, agentic_results).
    """
    baseline_results = []
    agentic_results  = []

    # Train per-SKU agents if none provided
    if rl_agents is None:
        from config import PRODUCTS as _PROD
        from stochastic_env import StochasticEnvironment as _SE
        rl_agents = {}
        for _sku, _p in _PROD.items():
            _bins = SKU_DEMAND_BINS.get(_sku, SKU_DEMAND_BINS['default'])
            _a = QLearningInventoryAgent(seed=RANDOM_SEED_BASE, demand_bins=_bins)
            def _f(p=_p):
                def _ef(u='base'): return _SE(u, demand_lambda=p['demand_lambda'], lead_time_mean=p['lead_time_mean'])
                return _ef
            _a.train(_f(), n_episodes=PORTFOLIO_RL_EPISODES, verbose=False,
                     unit_cost=_p["unit_cost"], unit_price=_p["unit_price"])
            rl_agents[_sku] = _a

    for run in range(n_runs):
        seed = RANDOM_SEED_BASE + run
        b = run_portfolio_simulation(rl_agents, seed, is_agentic=False)
        a = run_portfolio_simulation(rl_agents, seed, is_agentic=True)
        baseline_results.append(b)
        agentic_results.append(a)

        if verbose and (run + 1) % 5 == 0:
            b_sl = b['portfolio_sl']
            a_sl = a['portfolio_sl']
            b_p  = b['portfolio_profit']
            a_p  = a['portfolio_profit']
            print(f"  [Portfolio] Run {run+1:2d}/{n_runs} | "
                  f"Baseline SL={b_sl:.1%} P=£{b_p:,.0f} | "
                  f"Agentic  SL={a_sl:.1%} P=£{a_p:,.0f}")

    return baseline_results, agentic_results
