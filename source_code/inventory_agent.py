"""
Inventory Control Agent - Q-Learning (Reinforcement Learning)
Implements 

State : (inventory_bin, demand_trend_bin, pending_orders_bin, disruption_flag)
Action: Order quantity from discrete action set
Reward: revenue_margin - holding_cost - stockout_cost - order_fixed_cost
        - estimated_shipping_cost

The reward function explicitly includes an estimated shipping cost term
to align the training objective with the full economic KPI evaluated at
test time. Shipping is estimated using Standard Class rates during
training (conservative lower bound), ensuring the agent learns to
balance service-level performance against total operational cost.
"""

import numpy as np
import pickle
from typing import Tuple, List, Dict
from config import *

def discretize(value: float, bins: list) -> int:
    """Map continuous value to bin index."""
    for i, b in enumerate(bins[:-1]):
        if value < bins[i + 1]:
            return i
    return len(bins) - 2

class InventoryState:
    """Encodes environment state into discrete Q-table indices."""

    @staticmethod
    def encode(inventory: float, demand_forecast: float,
               pending_orders: float, disruption: bool,
               demand_bins: list = None,
               demand_rate: float = None) -> tuple:
        """
        Encode continuous state variables into discrete bin indices.

        State = (inv_bin, dem_bin, pend_bin, dis_flag, rate_bin)

        demand_rate: 7-day rolling mean demand. Used to detect regime
                     (low/mid/high) so the agent can learn regime-specific
                     policies. Resolves negative transfer at low-lambda
                     environments (lambda<10) by giving the agent a signal that
                     it is in a low-demand regime and should order less.
                     If None, defaults to bin 1 (mid-regime) -- neutral.
        """
        d_bins   = demand_bins if demand_bins is not None else DEMAND_BINS
        inv_bin  = discretize(inventory,       INVENTORY_BINS)
        dem_bin  = discretize(demand_forecast, d_bins)
        pend_bin = discretize(pending_orders,  PENDING_ORDER_BINS)
        dis_flag = int(disruption)
        # Demand regime bin -- defaults to mid-regime (1) if no rate provided
        if demand_rate is not None:
            rate_bin = discretize(demand_rate, DEMAND_RATE_BINS)
        else:
            rate_bin = 1
        return (inv_bin, dem_bin, pend_bin, dis_flag, rate_bin)

    @staticmethod
    def state_space_size(demand_bins: list = None) -> tuple:
        """Returns state space dimensions including demand rate bin."""
        d_bins = demand_bins if demand_bins is not None else DEMAND_BINS
        return (len(INVENTORY_BINS)     - 1,
                len(d_bins)             - 1,
                len(PENDING_ORDER_BINS) - 1,
                2,
                len(DEMAND_RATE_BINS)   - 1)

class QLearningInventoryAgent:
    """
    Q-Learning agent for inventory replenishment decisions.

    
    - Tabular Q-learning with epsilon-greedy exploration
    - Adaptive reward shaping covering all five operational cost components
    - Policy convergence tracking across training episodes
    - Full decision audit trail for 

    State space: 8 * 7 * 4 * 2 * 3 = 1,344 discrete states (default DEMAND_BINS).
    All SKUs share the same demand_rate_bin dimension (3 levels).
    Action space: 8 discrete order quantities {0, 25, 50, 75, 100, 150, 200, 250}
    """

    def __init__(self, seed: int = RANDOM_SEED_BASE, demand_bins: list = None):
        """
        Args:
            seed:        RNG seed for reproducibility.
            demand_bins: Per-SKU demand discretisation bins.
                         If None, uses global DEMAND_BINS from config.
                         Must be consistent between training and inference.
        """
        self.rng       = np.random.default_rng(seed)
        self.actions   = ORDER_ACTIONS
        self.n_actions = len(self.actions)

        # Store per-SKU bins -- used throughout train() and decide()
        self.demand_bins = demand_bins  # None -> global DEMAND_BINS

        # Q-table: shape = state_dims * n_actions
        # Optimistic init ensures exploration of all states
        # on unvisited states is non-zero, seeding exploration and preventing
        # the all-zero -> action[0]=ORDER=0 default that causes stockouts.
        s = InventoryState.state_space_size(demand_bins)
        self.q_table = np.full((*s, self.n_actions), OPTIMISTIC_Q_INIT)

        # Hyperparameters (from config)
        self.alpha         = RL_ALPHA
        self.alpha_decay   = ALPHA_DECAY
        self.alpha_min     = ALPHA_MIN
        self.gamma         = RL_GAMMA
        self.epsilon       = RL_EPSILON_START
        self.epsilon_min   = RL_EPSILON_MIN
        self.epsilon_decay = RL_EPSILON_DECAY

        self.episode_rewards:       List[float] = []
        self.policy_changes:        List[int]   = []
        self.convergence_episode:   int         = None
        self.is_trained             = False
        self._conv_patience_count:  int         = 0
        self.validation_sl_history: List[dict]  = []
        self._val_patience_count:   int         = 0
        # POLICY CHECKPOINTING: save Q-table snapshot at best validation SL.
        # Tabular Q-learning with continuous DR does not converge to a stable
        # policy -- the table oscillates as new DR episodes push it in different
        # directions. We checkpoint the best greedy policy seen during training
        # and restore it at the end. This is standard practice in RL
        # (Mnih et al., 2015, DQN; Lillicrap et al., 2016, DDPG).
        self._best_val_sl:    float     = -1.0
        self._best_q_table:   np.ndarray = None
        self._best_val_ep:    int        = -1
        self.name = "Q-Learning RL Agent"

        # Decision audit log 
        self.decision_log: List[Dict] = []

    def select_action(self, state: tuple, greedy: bool = False) -> Tuple[int, int]:
        """
        epsilon-greedy action selection.
        Returns (action_index, order_quantity).
        greedy=True forces exploitation (used at evaluation time).
        """
        if not greedy and self.rng.random() < self.epsilon:
            idx = int(self.rng.integers(0, self.n_actions))
        else:
            idx = int(np.argmax(self.q_table[state]))
        return idx, self.actions[idx]

    def update(self, state: tuple, action_idx: int,
               reward: float, next_state: tuple):
        """Standard Bellman Q-update."""
        current_q  = self.q_table[state][action_idx]
        max_next_q = np.max(self.q_table[next_state])
        td_target  = reward + self.gamma * max_next_q
        self.q_table[state][action_idx] += self.alpha * (td_target - current_q)

    def decay_epsilon(self):
        """Reduce exploration rate and learning rate after each training episode.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.alpha   = max(self.alpha_min,   self.alpha   * self.alpha_decay)

    def compute_reward(self, inventory: float, demand: float,
                       units_sold: float, order_qty: int,
                       ordered: bool, shipping_cost: float = 0.0,
                       unit_cost: float = None,
                       unit_price: float = None,
                       carbon_kgco2e: float = 0.0) -> float:
        """
        Composite reward = revenue_margin
                         − holding_cost
                         − stockout_cost
                         − order_fixed_cost
                         − shipping_cost
                         − carbon_penalty          

        carbon_penalty = CARBON_PENALTY_PER_KGCO2 * carbon_kgco2e
        Penalises high-carbon shipping modes (air > road) during training,
        so the learnt policy trades service level against emissions at
        training time rather than ignoring carbon entirely.
        At £0.50/kgCO2e (UK ETS shadow price), a First Class (air) shipment
        of 100u incurs ~£18 carbon penalty vs £3 for Standard Class (road).
        """
        uc = unit_cost if unit_cost is not None else UNIT_COST
        up = unit_price if unit_price is not None else UNIT_PRICE

        revenue    = units_sold * (up - uc)
        holding    = (inventory * uc * HOLDING_COST_RATE) / DAYS_PER_YEAR
        unmet      = max(0.0, demand - units_sold)
        stockout   = unmet * uc * STOCKOUT_COST_RATE
        order_cost = ORDER_FIXED_COST if ordered else 0.0
        carbon_pen = CARBON_PENALTY_PER_KGCO2 * carbon_kgco2e

        return revenue - holding - stockout - order_cost - shipping_cost - carbon_pen

    def train(self, env_factory, n_episodes: int = RL_EPISODES,
              verbose: bool = True, domain_randomise: bool = True,
              unit_cost: float = None, unit_price: float = None):
        """
        Train Q-learning agent using simulated environment episodes.

        Args:
            env_factory      : callable(uncertainty_level) -> StochasticEnvironment
                               CRITICAL: must return an env with the correct
                               demand_lambda/lead_time_mean for this SKU.
                               Per-SKU portfolio agents must pass a factory
                               that overrides these parameters.
            n_episodes       : number of training episodes
            domain_randomise : if True, mix base (60%) and high (40%)
                               uncertainty episodes for a robust policy
                               (
            unit_cost        : per-SKU unit cost for reward shaping.
            unit_price       : per-SKU unit price for reward shaping.
                               Defaults to global UNIT_PRICE if None.
                               Must be set for multi-SKU portfolio agents.
                               If None, falls back to global UNIT_COST.

        Convergence criterion: reward-stability.
        Convergence is declared when the fractional improvement in rolling
        mean episode reward is < RL_CONVERGENCE_DELTA for
        RL_CONVERGENCE_PATIENCE consecutive episodes.
        This is tighter than the old policy-change criterion (which fired at
        ep 50 despite the Q-table changing massively through ep 1000).
        """
        from forecasting_agent import AdaptiveForecaster

        _unit_cost  = unit_cost  if unit_cost  is not None else UNIT_COST
        _unit_price = unit_price if unit_price is not None else UNIT_PRICE

        prev_policy = None

        from logistics_agent import OptimizedLogisticsAgent, create_shipment_request as _make_req

        for ep in range(n_episodes):
            # from a continuous distribution each episode rather than a two-point mixture.
            # Implements Tobin et al. (2017) domain randomisation for genuine DRO coverage.
            if domain_randomise and DR_ENABLED:
                dr_lambda  = float(self.rng.uniform(*DR_DEMAND_LAMBDA_RANGE))
                dr_lt_mean = float(self.rng.uniform(*DR_LEAD_TIME_MEAN_RANGE))
                dr_dis_p   = float(self.rng.uniform(*DR_DISRUPTION_PROB_RANGE))
                env = env_factory('base')
                # uses self.lt_mean; setting only lead_time_mean had zero effect on sampling.
                env.demand_lambda   = dr_lambda
                env.lt_mean         = dr_lt_mean
                env.lead_time_mean  = dr_lt_mean   # keep in sync for consistency
                env.disruption_prob = dr_dis_p
                env._recompute_lognormal_params()
            else:
                uncertainty = 'high' if (ep % 5 >= 3) else 'base'
                if not domain_randomise:
                    uncertainty = 'base'
                env = env_factory(uncertainty)

            # disruption) from a single deterministic episode seed. Previously
            # env.rng = new_rng only replaced the demand_rng alias, leaving lt_rng
            # and disruption_rng in an indeterminate state across episodes.
            env.reset(seed=ep + 1000)

            _log_agent = OptimizedLogisticsAgent()

            forecaster    = AdaptiveForecaster()
            inventory     = float(INITIAL_INVENTORY)
            pending_orders: Dict[int, float] = {}
            total_reward  = 0.0
            _ship_req_id  = 0
            demand_history: List[float] = []   # for rolling demand rate (regime detection)

            # Warm-up: 7 days of demand history before episode begins
            for _ in range(7):
                d_warmup = float(env.sample_demand())
                forecaster.update(d_warmup)
                demand_history.append(d_warmup)

            for day in range(SIMULATION_DAYS):
                disruption = env.advance_day()

                # Receive pending orders arriving today
                inventory += pending_orders.pop(day, 0)

                # Observe demand and update forecast
                demand          = env.sample_demand()
                forecaster.update(float(demand))
                demand_forecast = forecaster.forecast()
                # Append day-t demand BEFORE computing demand_rate so that
                # state encoding uses a rate that includes today's observation.
                # This also ensures next_state uses the same (now current) rate,
                # which is the best approximation of s' available without
                # look-ahead. The 7-day window changes slowly enough that
                # state and next_state share the same rate_bin in >95% of steps.
                demand_history.append(float(demand))
                if len(demand_history) > DEMAND_RATE_WINDOW:
                    demand_history.pop(0)
                demand_rate = float(np.mean(demand_history))

                pending_qty = sum(pending_orders.values())

                # Encode state (includes demand regime bin)
                state = InventoryState.encode(
                    inventory, demand_forecast, pending_qty, disruption,
                    self.demand_bins, demand_rate)

                # Select action
                action_idx, order_qty = self.select_action(state)

                # matching the reward signal to evaluation-time conditions.
                ordered       = order_qty > 0
                shipping_cost = 0.0
                if ordered:
                    lt = env.get_effective_lead_time()
                    arrival_day = day + lt
                    pending_orders[arrival_day] = (
                        pending_orders.get(arrival_day, 0) + order_qty)
                    req = _make_req(day, order_qty, inventory, demand_forecast, _ship_req_id)
                    req = _log_agent.route(req)
                    shipping_cost = req.cost
                    carbon_train  = req.carbon
                    _ship_req_id += 1
                else:
                    carbon_train = 0.0

                # Fulfil demand
                units_sold = min(inventory, float(demand))
                inventory  = max(0.0, inventory - float(demand))

                # Compute reward
                reward = self.compute_reward(
                    inventory, demand, units_sold, order_qty,
                    ordered, shipping_cost,
                    unit_cost=_unit_cost,
                    unit_price=_unit_price,
                    carbon_kgco2e=carbon_train)
                total_reward += reward

                # Observe next state (includes demand regime)
                next_pending = sum(pending_orders.values())
                next_state   = InventoryState.encode(
                    inventory, demand_forecast, next_pending, disruption,
                    self.demand_bins, demand_rate)

                # Q-table update
                self.update(state, action_idx, reward, next_state)

            self.episode_rewards.append(total_reward)
            self.decay_epsilon()

            # HELD-OUT VALIDATION
            # Every VALIDATION_INTERVAL episodes, run the GREEDY policy on
            # VALIDATION_SEEDS (held-out, never used in training) and measure SL.
            # Convergence = validation SL improves by < VALIDATION_SL_DELTA for
            # VALIDATION_SL_PATIENCE consecutive checks.
            # This is a genuine out-of-sample convergence test -- not reward noise.
            if (ep + 1) % VALIDATION_INTERVAL == 0:
                val_sls = []
                # Validation across BOTH base and high uncertainty conditions.
                # Using only base ('base') caused ep800 to be selected as "best"
                # even though it performs poorly on high-demand (lambda=25) conditions.
                # Mixed validation ensures the checkpoint generalises across all conditions.
                _val_conditions = [('base', vseed) for vseed in VALIDATION_SEEDS] +                                   [('high', vseed + 500) for vseed in VALIDATION_SEEDS[:5]]
                for _ul, vseed in _val_conditions:
                    val_env   = env_factory(_ul)
                    val_env.reset(seed=vseed)
                    val_inv   = float(INITIAL_INVENTORY)
                    val_po    = {}
                    val_dem   = 0.0
                    val_ful   = 0.0
                    val_fc    = AdaptiveForecaster()
                    val_hist_v: List[float] = []
                    for _ in range(7):
                        dw = float(val_env.sample_demand())
                        val_fc.update(dw)
                        val_hist_v.append(dw)
                    for vday in range(SIMULATION_DAYS):
                        val_env.advance_day()
                        val_inv += val_po.pop(vday, 0)
                        vd = float(val_env.sample_demand())
                        val_fc.update(vd)
                        val_hist_v.append(vd)
                        if len(val_hist_v) > DEMAND_RATE_WINDOW:
                            val_hist_v.pop(0)
                        vrate = float(np.mean(val_hist_v))
                        vpq = sum(val_po.values())
                        vstate = InventoryState.encode(
                            val_inv, val_fc.forecast(), vpq,
                            val_env.disruption_active, self.demand_bins, vrate)
                        _, voq = self.select_action(vstate, greedy=True)
                        if voq > 0:
                            vlt = val_env.get_effective_lead_time()
                            val_po[vday + vlt] = val_po.get(vday + vlt, 0) + voq
                        vus = min(val_inv, vd)
                        val_inv = max(0.0, val_inv - vd)
                        val_dem += vd
                        val_ful += vus
                    val_sls.append(val_ful / val_dem if val_dem > 0 else 1.0)

                mean_val_sl = float(np.mean(val_sls))
                check = {'ep': ep + 1, 'val_sl': round(mean_val_sl, 5),
                         'val_sl_std': round(float(np.std(val_sls, ddof=1)), 5)}
                self.validation_sl_history.append(check)

                # CHECKPOINT: save best Q-table by validation SL
                if mean_val_sl > self._best_val_sl:
                    self._best_val_sl   = mean_val_sl
                    self._best_q_table  = self.q_table.copy()
                    self._best_val_ep   = ep + 1

                # Check convergence on validation SL
                if (len(self.validation_sl_history) >= 2
                        and self.convergence_episode is None):
                    prev_sl = self.validation_sl_history[-2]['val_sl']
                    curr_sl = mean_val_sl
                    if abs(curr_sl - prev_sl) < VALIDATION_SL_DELTA:
                        self._val_patience_count += 1
                        if self._val_patience_count >= VALIDATION_SL_PATIENCE:
                            self.convergence_episode = ep + 1
                            if verbose:
                                print(f"  [QLearning] Validation SL converged at ep "
                                      f"{self.convergence_episode} "
                                      f"(val_SL={mean_val_sl*100:.2f}%)")
                    else:
                        self._val_patience_count = 0

                if verbose:
                    avg_r = (np.mean(self.episode_rewards[-50:])
                             if ep >= 50 else total_reward)
                    print(f"  [QLearning] Ep {ep+1:4d}/{n_episodes}  "
                          f"epsilon={self.epsilon:.3f}  avg_reward={avg_r:.1f}  "
                          f"val_SL={mean_val_sl*100:.2f}%")

            # Track policy change count (diagnostic only)
            current_policy = np.argmax(self.q_table, axis=-1).copy()
            if prev_policy is not None:
                changes = int(np.sum(current_policy != prev_policy))
                self.policy_changes.append(changes)
            prev_policy = current_policy

        self.is_trained = True

        # RESTORE BEST CHECKPOINT: use Q-table from episode with highest validation SL.
        # This eliminates the "oscillating policy" problem -- we evaluate the best
        # greedy policy seen during training, not the final (potentially degraded) one.
        if self._best_q_table is not None:
            self.q_table = self._best_q_table
            self.convergence_episode = self._best_val_ep
        else:
            self.convergence_episode = n_episodes

        final_val_sl = self._best_val_sl * 100
        print(f"  [QLearning] Training complete. "
              f"Best checkpoint: ep {self.convergence_episode}  "
              f"Best val_SL: {final_val_sl:.2f}%  "
              f"(trained {n_episodes} episodes total)")

    def decide(self, inventory: float, demand_forecast: float,
               pending_qty: float, disruption: bool,
               day: int, context: dict = None,
               demand_rate: float = None) -> Tuple[int, dict]:
        """
        Greedy policy decision for evaluation (post-training).
        Returns (order_qty, audit_record) with full audit record.

        demand_rate: 7-day rolling mean demand. Used to determine the demand
                     regime bin (low/mid/high) in the Q-table state encoding.
                     If None, defaults to mid-regime (bin 1) -- neutral fallback.

        Unvisited-state fallback: if state was never visited in training
        (all Q-values still equal OPTIMISTIC_Q_INIT), fall back to the same
        ROP heuristic used by the baseline agent. This is conservative and
        prevents the agent from appearing better than it is on novel states.
        """
        state = InventoryState.encode(
            inventory, demand_forecast, pending_qty, disruption,
            self.demand_bins, demand_rate)

        q_row   = self.q_table[state]
        visited = not np.allclose(q_row, OPTIMISTIC_Q_INIT)

        if visited:
            _, order_qty = self.select_action(state, greedy=True)
            policy_source = 'rl_learned'
        else:
            # Fallback: ROP heuristic identical to baseline agent
            effective = inventory + pending_qty
            order_qty = ORDER_QUANTITY_EOQ if effective <= REORDER_POINT_BASELINE else 0
            policy_source = 'rop_fallback_unvisited'

        q_values = self.q_table[state].tolist()
        audit = {
            'day': day,
            'agent': self.name,
            'state': {
                'inventory': round(inventory, 2),
                'demand_forecast': round(demand_forecast, 2),
                'pending_orders': round(pending_qty, 2),
                'disruption': disruption,
                'demand_rate': round(demand_rate, 2) if demand_rate is not None else None,
            },
            'encoded_state': state,
            'action': order_qty,
            'q_values': [round(q, 4) for q in q_values],
            'best_q': round(float(max(q_values)), 4),
            'state_visited': visited,
            'policy_source': policy_source,
            'context': context or {}
        }
        self.decision_log.append(audit)
        return order_qty, audit

    def save(self, path: str):
        """Persist Q-table and training metadata for reproducibility."""
        with open(path, 'wb') as f:
            pickle.dump({
                'q_table':             self.q_table,
                'episode_rewards':     self.episode_rewards,
                'policy_changes':      self.policy_changes,
                'convergence_episode': self.convergence_episode
            }, f)

    def load(self, path: str):
        """Load persisted Q-table (sets agent to greedy evaluation mode)."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.q_table             = data['q_table']
        self.episode_rewards     = data['episode_rewards']
        self.policy_changes      = data['policy_changes']
        self.convergence_episode = data['convergence_episode']
        self.is_trained = True
        self.epsilon    = self.epsilon_min  # pure exploitation

class FixedReorderAgent:
    """
    Baseline inventory agent: fixed reorder point (ROP) policy.
    Places an EOQ-sized order whenever effective inventory (on-hand
    + in-transit) falls at or below the reorder point.
    
    """

    def __init__(self):
        self.reorder_point  = REORDER_POINT_BASELINE
        self.order_quantity = ORDER_QUANTITY_EOQ
        self.name           = "Fixed Reorder Point (Baseline)"
        self.decision_log: List[Dict] = []

    def decide(self, inventory: float, demand_forecast: float,
               pending_qty: float, disruption: bool,
               day: int, context: dict = None) -> Tuple[int, dict]:
        """Reorder if inventory + in-transit falls below ROP."""
        effective_inventory = inventory + pending_qty
        order_qty = (self.order_quantity
                     if effective_inventory <= self.reorder_point else 0)

        audit = {
            'day': day,
            'agent': self.name,
            'inventory': round(inventory, 2),
            'pending': round(pending_qty, 2),
            'effective': round(effective_inventory, 2),
            'reorder_point': self.reorder_point,
            'action': order_qty
        }
        self.decision_log.append(audit)
        return order_qty, audit
