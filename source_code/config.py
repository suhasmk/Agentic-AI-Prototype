"""
Configuration and Constants -- Agentic AI Inventory Management Prototype
"""
# ── Simulation ─────────────────────────────────────────────────────────────
SIMULATION_RUNS  = 30
SIMULATION_DAYS  = 365
RANDOM_SEED_BASE = 42
DEMAND_LAMBDA_BASE   = 15
DEMAND_LAMBDA_HIGH   = 25
LEAD_TIME_MEAN       = 5
LEAD_TIME_STD_BASE   = 1.0
LEAD_TIME_STD_HIGH   = 2.5
DISRUPTION_PROB_BASE     = 0.02
DISRUPTION_PROB_HIGH     = 0.08
DISRUPTION_DURATION_MEAN = 3

# ── Single-product parameters ──────────────────────────────────────────────
HOLDING_COST_RATE   = 0.25
STOCKOUT_COST_RATE  = 2.0
ORDER_FIXED_COST    = 50.0
UNIT_COST           = 141.23
UNIT_PRICE          = 180.00
REORDER_POINT_BASELINE = 75
ORDER_QUANTITY_EOQ     = 100
INITIAL_INVENTORY      = 150

# ── Multi-SKU product catalogue ────────────────────────────────────────────
PRODUCTS = {
    "SKU_FIELD_SPORT": {
        "name": "Field & Stream Sportsman 16 Gun Fire Safe",
        "category": "Field & Stream",
        "unit_cost": 141.23, "unit_price": 180.00,
        "demand_lambda": 15, "lead_time_mean": 5,
        "reorder_point": 75, "order_quantity": 100,
        "initial_inventory": 150, "holding_rate": 0.25,
        "stockout_rate": 2.0, "order_fixed_cost": 50.0,
        "supplier_region": "US_EAST", "weight_kg": 45.0,
    },
    "SKU_FITNESS": {
        "name": "Perfect Fitness Perfect Rip Deck",
        "category": "Fitness",
        "unit_cost": 89.50, "unit_price": 120.00,
        "demand_lambda": 22, "lead_time_mean": 4,
        "reorder_point": 60, "order_quantity": 80,
        "initial_inventory": 120, "holding_rate": 0.20,
        "stockout_rate": 2.0, "order_fixed_cost": 40.0,
        "supplier_region": "US_EAST", "weight_kg": 1.5,
    },
    "SKU_BICYCLE": {
        "name": "Diamondback Women's Serene Classic Comfort Bike",
        "category": "Bicycles",
        "unit_cost": 220.00, "unit_price": 299.00,
        "demand_lambda": 8, "lead_time_mean": 7,
        "reorder_point": 40, "order_quantity": 50,
        "initial_inventory": 80, "holding_rate": 0.20,
        "stockout_rate": 2.5, "order_fixed_cost": 75.0,
        "supplier_region": "US_WEST", "weight_kg": 14.5,
    },
    "SKU_WATCH": {
        "name": "Nautica Men's Voyage N83 Watch",
        "category": "Accessories",
        "unit_cost": 52.00, "unit_price": 75.00,
        "demand_lambda": 30, "lead_time_mean": 3,
        "reorder_point": 90, "order_quantity": 150,
        "initial_inventory": 200, "holding_rate": 0.30,
        "stockout_rate": 2.0, "order_fixed_cost": 30.0,
        "supplier_region": "ASIA", "weight_kg": 0.2,
    },
    "SKU_GOLF": {
        "name": "Nike Men's Comfort 2 Golf Shoe",
        "category": "Golf",
        "unit_cost": 95.00, "unit_price": 130.00,
        "demand_lambda": 12, "lead_time_mean": 6,
        "reorder_point": 50, "order_quantity": 75,
        "initial_inventory": 100, "holding_rate": 0.22,
        "stockout_rate": 2.0, "order_fixed_cost": 45.0,
        "supplier_region": "ASIA", "weight_kg": 1.0,
    },
}
DEFAULT_SKU = "SKU_FIELD_SPORT"

# ── Logistics ──────────────────────────────────────────────────────────────
SHIPPING_COSTS = {"Standard Class": 8.0,"First Class": 18.0,"Second Class": 12.0,"Same Day": 35.0}
SHIPPING_DAYS  = {"Standard Class": 5,  "First Class": 2,  "Second Class": 3,   "Same Day": 1}
CARBON_FACTORS = {"Standard Class": 0.15,"First Class": 0.85,"Second Class": 0.45,"Same Day": 1.20}
CONSOLIDATION_DISCOUNT      = 0.15
CONSOLIDATION_CARBON_SAVING = 0.10

# ── Reinforcement learning ─────────────────────────────────────────────────
RL_ALPHA         = 0.1
RL_GAMMA         = 0.95
RL_EPSILON_START = 1.0
RL_EPSILON_MIN   = 0.05
RL_EPSILON_DECAY = 0.9936
RL_EPISODES      = 2000
RL_CONVERGENCE_WINDOW   = 100
RL_CONVERGENCE_DELTA    = 0.005
RL_CONVERGENCE_PATIENCE = 50
VALIDATION_INTERVAL    = 100
VALIDATION_SEEDS       = list(range(9000, 9010))
VALIDATION_SL_DELTA    = 0.002
VALIDATION_SL_PATIENCE = 3
OPTIMISTIC_Q_INIT      = 50.0
CARBON_PENALTY_PER_KGCO2 = 0.50
HOLT_SS_SERVICE_FACTOR    = 2.33
ABLATION_RUNS = 30

# ── State space ────────────────────────────────────────────────────────────
INVENTORY_BINS     = [0, 25, 50, 75, 100, 150, 200, 300, float('inf')]
DEMAND_BINS        = [0, 3, 6, 9, 12, 18, 25, float('inf')]
PENDING_ORDER_BINS = [0, 50, 100, 200, float('inf')]
# Demand rate regime bins (3 regimes: low / mid / high).
# 7-day rolling mean encodes which ordering regime the agent is in.
# Eliminates negative transfer at lambda<10 by giving regime-specific policy slots.
DEMAND_RATE_BINS   = [0, 9, 18, float('inf')]
DEMAND_RATE_WINDOW = 7
SKU_DEMAND_BINS    = {
    'SKU_BICYCLE': [0, 2, 4, 6, 8, 10, 14, float('inf')],
    'SKU_GOLF':    [0, 3, 6, 9, 12, 16, float('inf')],
    'default':     DEMAND_BINS,
}

# ── Action space and derived governance thresholds ─────────────────────────
ORDER_ACTIONS        = [0, 25, 50, 75, 100, 150, 200, 250]
TARGET_SERVICE_LEVEL = 0.95
_sorted_actions = sorted(ORDER_ACTIONS)
_n_actions      = len(_sorted_actions)
_gov_idx             = int((87.5 / 100.0) * (_n_actions - 1))
GOVERNANCE_ORDER_CAP = _sorted_actions[_gov_idx]          # 200u
_p75_idx = int(0.75 * (_n_actions - 1))
_p50_idx = int(0.50 * (_n_actions - 1))
HITL_ORDER_QTY_THRESHOLD   = _sorted_actions[_p75_idx]    # 150u
HITL_QTY_CAP               = _sorted_actions[_p50_idx]    # 75u
HITL_ORDER_VALUE_THRESHOLD  = HITL_ORDER_QTY_THRESHOLD * UNIT_COST
HITL_DISRUPTION_FLAG        = True
HITL_DISRUPTION_UPLIFT      = 0.10

# ── Operational constants ──────────────────────────────────────────────────
DAYS_PER_YEAR = 365
# Portfolio trains for 500 episodes so epsilon reaches minimum and policy stabilises.
PORTFOLIO_RL_EPISODES      = 1500  # Raised from 500: allows 1,033 post-exploration episodes
DISRUPTION_VALIDATION_DAYS = 200
HOLT_ALPHA = 0.3
HOLT_BETA  = 0.1
TARDINESS_PENALTY_RATE  = 0.1
GOVERNANCE_EXPORT_LIMIT = 100

# ── Production ─────────────────────────────────────────────────────────────
PRODUCTION_CAPACITY         = 50
SETUP_COST                  = 200.0
PRODUCTS_TO_SCHEDULE        = 5
PRODUCTION_BUFFER_MAX       = 30
PRODUCTION_FILL_RATE        = 0.20
PRODUCTION_UNIT_COST_FACTOR = 0.85

# ── Drift monitoring ───────────────────────────────────────────────────────
# Calibrated for agentic performance floor (~97%+ SL).
# The baseline (86% SL) would also trigger the 90% alert -- this monitor
# is designed for detecting agentic policy degradation only.
DRIFT_WINDOW_DAYS     = 30
DRIFT_ALERT_THRESHOLD = 0.90
DRIFT_RETRAIN_TRIGGER = 0.85
DRIFT_SAFETY_UPLIFT   = 0.20
DRIFT_MONITOR_ACTIVE  = True

# ── Pareto frontier sweep ──────────────────────────────────────────────────
PARETO_CARBON_PENALTIES = [0, 0.10, 0.25, 0.50, 1.0, 2.0, 5.0, 10.0, 20.0]
PARETO_RUNS_PER_POINT   = 10

# ── Supplier reliability ───────────────────────────────────────────────────
SUPPLIER_RELIABILITY = {"US_EAST": 0.92,"US_WEST": 0.88,"ASIA": 0.78}
SUPPLIER_RELIABILITY_IMPACT = 0.5
ACTIVE_SUPPLIER       = "US_EAST"
RELIABILITY_LT_IMPACT = 0.8
DISRUPTION_LT_MULTIPLIER  = 2.5
DISRUPTION_STD_MULTIPLIER = 2.0
PROD_SCHED_SEED_OFFSET    = 500
ALPHA_DECAY = 0.997
ALPHA_MIN   = 0.01

# ── Domain randomisation ───────────────────────────────────────────────────
DR_ENABLED                = True
DR_DEMAND_LAMBDA_RANGE    = (6.0, 26.0)
DR_LEAD_TIME_MEAN_RANGE   = (3.5,  7.0)
DR_DISRUPTION_PROB_RANGE  = (0.01, 0.10)  # Extended from (0.01,0.06) to cover DISRUPTION_PROB_HIGH=0.08 # so high-uncertainty evaluation conditions fall within DR training support.
OOD_LAMBDA_LOW  = DR_DEMAND_LAMBDA_RANGE[0] * 0.70
OOD_LAMBDA_HIGH = DR_DEMAND_LAMBDA_RANGE[1] * 1.10

# ── Statistical analysis ───────────────────────────────────────────────────
STAT_ALPHA   = 0.05
MIN_SIG_KPIS = 3
NEGBIN_R     = 5
ARIMA_AR_ORDER    = 3
ARIMA_REFIT_EVERY = 30
