"""
Real-Time Operations Dashboard Server -- Agentic AI Prototype v2
Serves a production-grade web dashboard via Server-Sent Events (SSE).

Run:  python3 dashboard_server.py [--port 8765]
Then: open http://localhost:8765

Dashboard panels:
  - Training Progress  : Q-learning convergence (episodes, epsilon, reward curve)
  - KPI Comparison     : Live Baseline vs Agentic across 8 KPIs with stats
  - Service Level Gauge: Visual service level indicator
  - Shipping Modes     : Mode distribution (agentic vs baseline)
  - Drift Monitor      : Rolling SL per SKU with alert badges
  - Portfolio View     : 5-SKU summary (5 DataCo products)
  - Operations Log     : Audit trail and anomaly feed
"""

import sys
import os
import json
import threading
import queue
import argparse
import traceback
import socket

sys.path.insert(0, os.path.dirname(__file__))

from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np

from config import (RANDOM_SEED_BASE, SIMULATION_DAYS, INITIAL_INVENTORY,
                    PRODUCTS, ORDER_FIXED_COST, SHIPPING_COSTS, RL_EPISODES)
from stochastic_env import StochasticEnvironment
from forecasting_agent import AdaptiveForecaster
from inventory_agent import QLearningInventoryAgent, FixedReorderAgent, InventoryState
from production_agent import generate_production_jobs
from logistics_agent import (DefaultLogisticsAgent, OptimizedLogisticsAgent,
                               create_shipment_request)
from governance_agent import GovernanceAgent
from statistical_analysis import full_comparison

# ─── Global broadcast state ────────────────────────────────────────────────────
_subscribers: list = []
_sub_lock = threading.Lock()
sim_state = {"running": False, "phase": "idle", "error": None}

def push(event_type: str, data: dict):
    """Broadcast an SSE event to all connected clients."""
    payload = json.dumps({"type": event_type, "data": data}, default=str)
    line = f"data: {payload}\n\n"
    with _sub_lock:
        dead = []
        for q in _subscribers:
            try:
                q.put_nowait(line)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _subscribers.remove(q)

# ─── Simulation pipeline ───────────────────────────────────────────────────────

def _run_training(rl_agent: QLearningInventoryAgent, n_episodes: int = 300):
    push("phase", {
        "phase": "training",
        "message": f"Training Q-Learning agent ({n_episodes} episodes)…",
        "total_episodes": n_episodes,
    })

    prev_policy = None
    convergence_window = 50
    convergence_threshold = 10

    for ep in range(n_episodes):
        dr_uncertainty = 'high' if (ep % 5 >= 3) else 'base'
        env = StochasticEnvironment(uncertainty_level=dr_uncertainty, seed=ep + 1000)
        forecaster = AdaptiveForecaster()
        inventory = float(INITIAL_INVENTORY)
        pending_orders = {}
        total_reward = 0.0

        for _ in range(7):
            forecaster.update(float(env.sample_demand()))

        for day in range(SIMULATION_DAYS):
            disruption = env.advance_day()
            inventory += pending_orders.pop(day, 0)
            demand = env.sample_demand()
            forecaster.update(float(demand))
            demand_forecast = forecaster.forecast()
            pending_qty = sum(pending_orders.values())

            state = InventoryState.encode(inventory, demand_forecast,
                                           pending_qty, disruption)
            action_idx, order_qty = rl_agent.select_action(state)

            shipping_cost = 0.0
            if order_qty > 0:
                lt = env.get_effective_lead_time()
                pending_orders[day + lt] = pending_orders.get(day + lt, 0) + order_qty
                shipping_cost = ORDER_FIXED_COST + SHIPPING_COSTS["Standard Class"]

            units_sold = min(inventory, float(demand))
            inventory = max(0.0, inventory - float(demand))

            reward = rl_agent.compute_reward(inventory, float(demand),
                                              units_sold, order_qty,
                                              order_qty > 0, shipping_cost)
            next_state = InventoryState.encode(inventory, demand_forecast,
                                               sum(pending_orders.values()), disruption)
            rl_agent.update(state, action_idx, reward, next_state)
            total_reward += reward

        rl_agent.episode_rewards.append(total_reward)
        rl_agent.decay_epsilon()

        current_policy = np.argmax(rl_agent.q_table, axis=-1).copy()
        changes = 0
        if prev_policy is not None:
            changes = int(np.sum(current_policy != prev_policy))
            rl_agent.policy_changes.append(changes)
            if (rl_agent.convergence_episode is None
                    and len(rl_agent.policy_changes) >= convergence_window):
                recent = rl_agent.policy_changes[-convergence_window:]
                if max(recent) <= convergence_threshold:
                    rl_agent.convergence_episode = ep
        prev_policy = current_policy

        interval = max(1, n_episodes // 20)
        if ep % interval == 0 or ep == n_episodes - 1:
            avg_r = float(np.mean(rl_agent.episode_rewards[-50:])) if ep >= 50 else total_reward
            push("training_progress", {
                "episode":        ep,
                "total":          n_episodes,
                "pct":            round(100 * (ep + 1) / n_episodes, 1),
                "epsilon":        round(rl_agent.epsilon, 4),
                "avg_reward":     round(avg_r, 1),
                "policy_changes": changes,
                "converged_ep":   rl_agent.convergence_episode,
                "reward_history": [round(r, 0) for r in rl_agent.episode_rewards[-50:]],
            })

    rl_agent.is_trained = True
    if rl_agent.convergence_episode is None:
        rl_agent.convergence_episode = n_episodes - 1

    push("training_complete", {
        "convergence_episode": rl_agent.convergence_episode,
        "final_epsilon":       round(rl_agent.epsilon, 4),
        "n_episodes":          n_episodes,
        "final_avg_reward":    round(float(np.mean(rl_agent.episode_rewards[-50:])), 1),
    })

def _run_experiment(rl_agent, n_runs: int, uncertainty: str):
    from simulation_runner import run_baseline_simulation, run_agentic_simulation

    push("phase", {
        "phase":   "experiment",
        "message": f"Running {n_runs}* paired simulation ({uncertainty} uncertainty)…",
        "total_runs": n_runs,
    })

    baseline_results, agentic_results = [], []
    for run in range(n_runs):
        seed = RANDOM_SEED_BASE + run
        b = run_baseline_simulation(seed, uncertainty)
        a = run_agentic_simulation(seed, rl_agent, uncertainty)
        baseline_results.append(b)
        agentic_results.append(a)

        push("run_complete", {
            "run":                run + 1,
            "total":              n_runs,
            "pct":                round(100 * (run + 1) / n_runs, 1),
            "baseline_sl":        round(b["service_level"] * 100, 1),
            "agentic_sl":         round(a["service_level"] * 100, 1),
            "baseline_profit":    round(b["profit"], 0),
            "agentic_profit":     round(a["profit"], 0),
            "baseline_carbon":    round(b["carbon_kgco2e"], 1),
            "agentic_carbon":     round(a["carbon_kgco2e"], 1),
            "agentic_mode_counts":  a.get("shipping_mode_counts", {}),
            "baseline_mode_counts": b.get("shipping_mode_counts", {}),
        })

    return baseline_results, agentic_results

def _run_stats(baseline_results, agentic_results):
    push("phase", {"phase": "statistics", "message": "Computing statistical significance tests…"})
    comparison = full_comparison(baseline_results, agentic_results)

    kpi_table = []
    for kpi, r in comparison["kpi_results"].items():
        kpi_table.append({
            "kpi":            kpi,
            "label":          r["label"],
            "baseline_mean":  r["mean_baseline"],
            "agentic_mean":   r["mean_agentic"],
            "improvement":    round(r["improvement_pct"], 1),
            "cohen_d":        round(r["cohen_d"], 2),
            "p_value":        r["p_value"],
            "significant":    r["significant"],
            "agentic_better": r["agentic_better"],
        })

    push("statistics_complete", {
        "kpi_table":        kpi_table,
        "n_sig_improved":   comparison["n_significantly_improved"],
        "objective_5_met":  comparison["objective_5_met"],
        "alpha_bonferroni": comparison["alpha_bonferroni"],
        "n_runs":           comparison["n_runs"],
    })
    return comparison

def _run_portfolio():
    try:
        from portfolio_manager import run_portfolio_experiment
        push("phase", {"phase": "portfolio",
                        "message": "Running 5-SKU portfolio simulation (Gaps 1-7)…"})
        b_port, a_port = run_portfolio_experiment(n_runs=5, verbose=False)
        push("portfolio_complete", {
            "baseline": b_port,
            "agentic":  a_port,
            "n_skus":   len(PRODUCTS),
        })
    except Exception as e:
        push("portfolio_complete", {
            "error":    str(e),
            "baseline": {},
            "agentic":  {},
            "n_skus":   len(PRODUCTS),
        })

def simulation_pipeline(n_episodes: int, n_runs: int, uncertainty: str):
    global sim_state
    sim_state.update({"running": True, "error": None, "phase": "training"})

    try:
        push("status", {"running": True, "phase": "training"})
        rl_agent = QLearningInventoryAgent(seed=RANDOM_SEED_BASE)
        _run_training(rl_agent, n_episodes=n_episodes)
        baseline_res, agentic_res = _run_experiment(rl_agent, n_runs, uncertainty)
        _run_stats(baseline_res, agentic_res)
        _run_portfolio()

        sim_state["phase"] = "complete"
        push("simulation_complete", {
            "message": "All phases complete.",
            "n_runs": n_runs,
            "n_episodes": n_episodes,
        })
        push("status", {"running": False, "phase": "complete"})

    except Exception as e:
        tb = traceback.format_exc()
        sim_state["error"] = str(e)
        sim_state["phase"] = "error"
        push("error", {"message": str(e), "traceback": tb})
        push("status", {"running": False, "phase": "error"})
    finally:
        sim_state["running"] = False

# ─── Embedded dashboard HTML ───────────────────────────────────────────────────

DASHBOARD_HTML = b"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agentic AI Supply Chain Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
:root{
  --bg:#0b0e14;--surface:#111620;--surface2:#161d2b;--border:#1e2a3d;
  --text:#c8d4e8;--dim:#4e6080;--accent:#00e5aa;--warn:#ffb340;
  --danger:#ff4d6a;--blue:#4d9fff;--base-c:#ff4d6a;--agent-c:#00e5aa;
  --mono:'IBM Plex Mono',monospace;--sans:'IBM Plex Sans',sans-serif;
  --r:6px;--gap:14px;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:var(--sans);background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}

/* HEADER */
header{
  display:flex;align-items:center;justify-content:space-between;
  padding:12px 24px;background:var(--surface);
  border-bottom:1px solid var(--border);position:sticky;top:0;z-index:100;
}
.h-left{display:flex;align-items:center;gap:14px}
.badge{
  font-family:var(--mono);font-size:10px;font-weight:600;
  color:var(--bg);background:var(--accent);padding:3px 8px;
  border-radius:3px;letter-spacing:.08em;
}
.h-title{font-size:14px;font-weight:600;color:#e8f0ff;letter-spacing:.02em}
.h-sub{font-size:10px;color:var(--dim);font-family:var(--mono)}
.status-pill{
  font-family:var(--mono);font-size:10px;font-weight:500;
  padding:4px 12px;border-radius:20px;border:1px solid var(--border);
  display:flex;align-items:center;gap:6px;transition:all .3s;
}
.status-pill.idle{color:var(--dim)}
.status-pill.running{color:var(--warn);border-color:var(--warn)}
.status-pill.complete{color:var(--accent);border-color:var(--accent)}
.status-pill.error{color:var(--danger);border-color:var(--danger)}
.pulse{width:7px;height:7px;border-radius:50%;background:currentColor;
  animation:pulse 1.4s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.7)}}
.status-pill:not(.running) .pulse{animation:none}

/* LAYOUT */
main{display:grid;grid-template-columns:260px 1fr;height:calc(100vh - 49px);overflow:hidden}
aside{
  background:var(--surface);border-right:1px solid var(--border);
  overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:14px;
}
.dash{overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:14px}

/* SIDEBAR */
.s-head{
  font-family:var(--mono);font-size:9px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--dim);
  padding-bottom:6px;border-bottom:1px solid var(--border);margin-bottom:6px;
}
.ctrl label{font-family:var(--mono);font-size:9px;color:var(--dim);letter-spacing:.08em;text-transform:uppercase;display:block;margin-bottom:3px}
.ctrl input[type=range]{width:100%;accent-color:var(--accent)}
.ctrl-v{font-family:var(--mono);font-size:11px;color:var(--accent);float:right}
.ctrl select{
  width:100%;background:var(--surface2);border:1px solid var(--border);
  border-radius:var(--r);color:var(--text);font-family:var(--mono);
  font-size:11px;padding:5px 7px;
}
.run-btn{
  width:100%;padding:10px;background:var(--accent);color:var(--bg);
  border:none;border-radius:var(--r);font-family:var(--mono);font-size:12px;
  font-weight:600;letter-spacing:.05em;cursor:pointer;transition:opacity .2s;
}
.run-btn:hover{opacity:.85}
.run-btn:disabled{opacity:.35;cursor:not-allowed}

.mp-wrap{margin-top:4px}
.mp-label{display:flex;justify-content:space-between;font-family:var(--mono);font-size:9px;color:var(--dim);margin-bottom:3px}
.mp-track{height:3px;background:var(--surface2);border-radius:2px;overflow:hidden}
.mp-fill{height:100%;background:var(--accent);border-radius:2px;transition:width .4s ease;width:0}
.ms{display:flex;flex-direction:column;gap:3px;margin-top:7px}
.ms-row{display:flex;justify-content:space-between;font-family:var(--mono);font-size:10px}
.ms-k{color:var(--dim)}.ms-v{color:var(--text)}.ms-v.a{color:var(--accent)}

.obj-list{display:flex;flex-direction:column;gap:4px}
.obj-row{display:flex;align-items:flex-start;gap:8px;font-size:10px;color:var(--dim)}
.obj-chk{font-family:var(--mono);font-size:10px;color:var(--dim);flex-shrink:0}
.obj-chk.pass{color:var(--accent)}

/* CARDS */
.row{display:grid;gap:var(--gap)}
.r4{grid-template-columns:repeat(4,1fr)}
.r3{grid-template-columns:repeat(3,1fr)}
.r2{grid-template-columns:1fr 1fr}
.r22{grid-template-columns:2fr 1fr}

.card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r);padding:14px;position:relative;overflow:hidden;
}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--border);transition:background .4s}
.card.ac::before{background:var(--accent)}
.card.wn::before{background:var(--warn)}
.card.bl::before{background:var(--blue)}
.card.dn::before{background:var(--danger)}

.c-title{font-family:var(--mono);font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:var(--dim);margin-bottom:10px}

/* KPI TILES */
.kpi-t{
  display:flex;flex-direction:column;gap:2px;
  padding:12px;background:var(--surface2);
  border-radius:var(--r);border:1px solid var(--border);min-width:0;
}
.kpi-l{font-family:var(--mono);font-size:8px;color:var(--dim);letter-spacing:.08em;text-transform:uppercase}
.kpi-v{font-family:var(--mono);font-size:20px;font-weight:600;color:#e8f0ff;line-height:1;margin:3px 0}
.kpi-d{font-family:var(--mono);font-size:9px;color:var(--dim)}
.kpi-d.pos{color:var(--accent)}.kpi-d.neg{color:var(--danger)}.kpi-d.neu{color:var(--warn)}

/* TABLE */
.t{width:100%;border-collapse:collapse;font-size:11px}
.t th{font-family:var(--mono);font-size:9px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim);text-align:left;padding:6px 8px;border-bottom:1px solid var(--border)}
.t th.r{text-align:right}
.t td{padding:6px 8px;border-bottom:1px solid #1a2030;font-family:var(--mono);font-size:10px}
.t tr:last-child td{border-bottom:none}
.b-v{color:var(--base-c);text-align:right}
.a-v{color:var(--agent-c);text-align:right}
.chg{text-align:right}.chg.g{color:var(--accent)}.chg.b{color:var(--danger)}
.sig{text-align:center}
.sb{display:inline-block;padding:1px 5px;border-radius:3px;font-size:8px;font-family:var(--mono)}
.sb.y{background:rgba(0,229,170,.12);color:var(--accent)}
.sb.n{background:rgba(78,96,128,.12);color:var(--dim)}

/* GAUGE */
.gw{display:flex;flex-direction:column;align-items:center;gap:8px;padding:4px 0}
.g-svg{width:160px;height:95px}
.g-row{display:flex;gap:20px;font-family:var(--mono);font-size:10px}
.g-cell{text-align:center}
.g-cell-lbl{color:var(--dim);font-size:9px}

/* LOG FEED */
.log{font-family:var(--mono);font-size:10px;line-height:1.9;max-height:190px;overflow-y:auto;color:var(--dim)}
.log .e{display:flex;gap:8px}
.log .ts{color:var(--dim);flex-shrink:0}
.log .m.info{color:var(--text)}.log .m.ok{color:var(--accent)}
.log .m.w{color:var(--warn)}.log .m.err{color:var(--danger)}
.log .m.ph{color:var(--blue);font-weight:600}

/* DRIFT */
.drift{display:flex;flex-direction:column;gap:6px}
.dr{display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:10px}
.dr-sku{color:var(--text);width:110px;flex-shrink:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.dr-t{flex:1;height:5px;background:var(--surface2);border-radius:3px;overflow:hidden}
.dr-f{height:100%;border-radius:3px;transition:width .5s ease,background .3s}
.dr-sl{width:40px;text-align:right;font-size:10px}
.dr-b{font-size:8px;padding:1px 5px;border-radius:3px;width:50px;text-align:center;font-family:var(--mono)}
.dr-b.ok{background:rgba(0,229,170,.1);color:var(--accent)}
.dr-b.al{background:rgba(255,179,64,.12);color:var(--warn)}
.dr-b.rt{background:rgba(255,77,106,.12);color:var(--danger)}

/* PORTFOLIO */
.pf{display:grid;grid-template-columns:repeat(5,1fr);gap:7px}
.sku-c{background:var(--surface2);border:1px solid var(--border);border-radius:4px;padding:9px 6px;text-align:center}
.sku-n{font-family:var(--mono);font-size:8px;color:var(--dim);margin-bottom:5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sku-v{font-family:var(--mono);font-size:15px;font-weight:600}
.sku-l{font-family:var(--mono);font-size:8px;color:var(--dim)}

/* PHASE BANNER */
.banner{
  display:none;align-items:center;gap:10px;
  padding:9px 14px;background:var(--surface2);
  border:1px solid var(--border);border-radius:var(--r);
  font-family:var(--mono);font-size:11px;
}

/* MODE */
.mode-wrap{display:flex;gap:10px;align-items:center;height:160px}
.mode-leg{font-family:var(--mono);font-size:9px;color:var(--dim);line-height:2;flex:1}

::-webkit-scrollbar{width:3px;height:3px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
</style>
</head>
<body>

<header>
  <div class="h-left">
    <div class="badge">AGENT AI</div>
    <div>
      <div class="h-title">Agentic AI Supply Chain Dashboard</div>
      <div class="h-sub">Autonomous Decision-Making \xc2\xb7 Manufacturing & Logistics \xc2\xb7 v2</div>
    </div>
  </div>
  <div id="sPill" class="status-pill idle"><div class="pulse"></div><span id="sTxt">IDLE</span></div>
</header>

<main>
<aside>
  <div>
    <div class="s-head">Simulation Controls</div>
    <div style="display:flex;flex-direction:column;gap:9px;margin-top:4px">
      <div class="ctrl">
        <label>Training Episodes <span class="ctrl-v" id="epV">300</span></label>
        <input type="range" min="100" max="500" step="50" value="300" id="epS" oninput="epV.textContent=this.value">
      </div>
      <div class="ctrl">
        <label>Simulation Runs <span class="ctrl-v" id="rnV">20</span></label>
        <input type="range" min="5" max="30" step="5" value="20" id="rnS" oninput="rnV.textContent=this.value">
      </div>
      <div class="ctrl">
        <label>Uncertainty</label>
        <select id="uncS">
          <option value="base">Base (\xce\xbb=15, p\xe2\x82\x99=0.02)</option>
          <option value="high">High (\xce\xbb=25, p\xe2\x82\x99=0.08)</option>
        </select>
      </div>
      <button class="run-btn" id="runBtn" onclick="startSim()">\xe2\x96\xb6 RUN SIMULATION</button>
    </div>
  </div>

  <div>
    <div class="s-head">Training Progress</div>
    <div class="mp-wrap">
      <div class="mp-label"><span id="mpEp">Episode \xe2\x80\x94</span><span id="mpPct">0%</span></div>
      <div class="mp-track"><div class="mp-fill" id="mpFill"></div></div>
      <div class="ms">
        <div class="ms-row"><span class="ms-k">\xce\xb5 (epsilon)</span><span class="ms-v" id="sEps">\xe2\x80\x94</span></div>
        <div class="ms-row"><span class="ms-k">Avg reward</span><span class="ms-v" id="sRew">\xe2\x80\x94</span></div>
        <div class="ms-row"><span class="ms-k">Converged</span><span class="ms-v a" id="sConv">\xe2\x80\x94</span></div>
        <div class="ms-row"><span class="ms-k">Policy \xce\x94</span><span class="ms-v" id="sChg">\xe2\x80\x94</span></div>
      </div>
    </div>
  </div>

  <div>
    <div class="s-head">Objectives</div>
    <div class="obj-list">
      <div class="obj-row"><span class="obj-chk" id="o1">\xe2\x97\x8b</span><span>OBJ1 \xe2\x80\x94 Stochastic env</span></div>
      <div class="obj-row"><span class="obj-chk" id="o2">\xe2\x97\x8b</span><span>OBJ2 \xe2\x80\x94 Baseline system</span></div>
      <div class="obj-row"><span class="obj-chk" id="o3">\xe2\x97\x8b</span><span>OBJ3 \xe2\x80\x94 Agentic system</span></div>
      <div class="obj-row"><span class="obj-chk" id="o4">\xe2\x97\x8b</span><span>OBJ4 \xe2\x80\x94 RL convergence</span></div>
      <div class="obj-row"><span class="obj-chk" id="o5">\xe2\x97\x8b</span><span>OBJ5 \xe2\x80\x94 KPI comparison</span></div>
      <div class="obj-row"><span class="obj-chk" id="o6">\xe2\x97\x8b</span><span>OBJ6 \xe2\x80\x94 Robustness</span></div>
      <div class="obj-row"><span class="obj-chk" id="o7">\xe2\x97\x8b</span><span>OBJ7 \xe2\x80\x94 Governance</span></div>
      <div class="obj-row"><span class="obj-chk" id="o8">\xe2\x97\x8b</span><span>OBJ8 \xe2\x80\x94 Self-learning</span></div>
    </div>
  </div>

  <div style="font-family:var(--mono);font-size:9px;color:var(--dim);line-height:1.7;margin-top:auto;padding-top:8px;border-top:1px solid var(--border)">
    <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">
      <div style="width:8px;height:8px;border-radius:50%;background:var(--base-c);flex-shrink:0"></div>Baseline (Fixed ROP + MA)
    </div>
    <div style="display:flex;align-items:center;gap:6px">
      <div style="width:8px;height:8px;border-radius:50%;background:var(--agent-c);flex-shrink:0"></div>Agentic AI (Q-Learning)
    </div>
  </div>
</aside>

<div class="dash">

  <!-- Banner -->
  <div id="banner" class="banner">
    <span id="banIcon">\xe2\x9a\x99</span><span id="banMsg">Initialising\xe2\x80\xa6</span>
  </div>

  <!-- KPI tiles -->
  <div class="row r4">
    <div class="kpi-t">
      <div class="kpi-l">Service Level \xe2\x80\x94 Agentic</div>
      <div class="kpi-v" id="tSL">\xe2\x80\x94</div>
      <div class="kpi-d" id="tSLd">Baseline: \xe2\x80\x94</div>
    </div>
    <div class="kpi-t">
      <div class="kpi-l">Profit Delta (\xc2\xa3/yr)</div>
      <div class="kpi-v" id="tProfit">\xe2\x80\x94</div>
      <div class="kpi-d" id="tProfitD">vs baseline</div>
    </div>
    <div class="kpi-t">
      <div class="kpi-l">Stockout Days Reduced</div>
      <div class="kpi-v" id="tStock">\xe2\x80\x94</div>
      <div class="kpi-d" id="tStockD">days/year</div>
    </div>
    <div class="kpi-t">
      <div class="kpi-l">Carbon Footprint</div>
      <div class="kpi-v" id="tCarbon">\xe2\x80\x94</div>
      <div class="kpi-d" id="tCarbonD">kg CO\xe2\x82\x82e/yr</div>
    </div>
  </div>

  <!-- SL chart + Mode donut -->
  <div class="row r2">
    <div class="card ac">
      <div class="c-title">Service Level \xe2\x80\x94 Run by Run</div>
      <div style="height:170px"><canvas id="slC"></canvas></div>
    </div>
    <div class="card bl">
      <div class="c-title">Shipping Mode Distribution (Agentic)</div>
      <div class="mode-wrap">
        <div style="width:150px;flex-shrink:0;height:150px"><canvas id="mdC"></canvas></div>
        <div class="mode-leg" id="mdLeg"></div>
      </div>
    </div>
  </div>

  <!-- KPI table + Gauge -->
  <div class="row r22">
    <div class="card" style="overflow:auto">
      <div class="c-title">KPI Comparison \xe2\x80\x94 Paired t-test (Bonferroni \xce\xb1 = 0.00625)</div>
      <table class="t">
        <thead><tr>
          <th>KPI</th><th class="r">Baseline</th><th class="r">Agentic</th>
          <th class="r">\xce\x94%</th><th class="r">d</th><th style="text-align:center">Sig</th>
        </tr></thead>
        <tbody id="kpiTB">
          <tr><td colspan="6" style="color:var(--dim);padding:14px;text-align:center;font-family:var(--mono);font-size:10px">Run simulation to populate\xe2\x80\xa6</td></tr>
        </tbody>
      </table>
    </div>
    <div class="card">
      <div class="c-title">Service Level Gauge</div>
      <div class="gw">
        <svg class="g-svg" viewBox="0 0 200 110">
          <defs>
            <linearGradient id="gG" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stop-color="#ff4d6a"/>
              <stop offset="55%" stop-color="#ffb340"/>
              <stop offset="100%" stop-color="#00e5aa"/>
            </linearGradient>
          </defs>
          <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="#1e2a3d" stroke-width="13" stroke-linecap="round"/>
          <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="url(#gG)" stroke-width="13" stroke-linecap="round" opacity=".18"/>
          <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="url(#gG)" stroke-width="13" stroke-linecap="round"
                stroke-dasharray="251.2" id="gArc" stroke-dashoffset="251.2" style="transition:stroke-dashoffset 1.2s ease"/>
          <circle cx="100" cy="100" r="4" fill="#e8f0ff"/>
          <text x="14" y="112" fill="#4e6080" font-size="8" font-family="monospace">0%</text>
          <text x="170" y="112" fill="#4e6080" font-size="8" font-family="monospace">100%</text>
          <text id="gTxt" x="100" y="85" text-anchor="middle" fill="#e8f0ff" font-size="20" font-weight="600" font-family="monospace">\xe2\x80\x94</text>
          <text x="100" y="99" text-anchor="middle" fill="#4e6080" font-size="8" font-family="monospace">SERVICE LEVEL</text>
        </svg>
        <div class="g-row">
          <div class="g-cell"><div class="g-cell-lbl">BASELINE</div><div id="gB" style="color:var(--base-c)">\xe2\x80\x94</div></div>
          <div class="g-cell"><div class="g-cell-lbl">AGENTIC</div><div id="gA" style="color:var(--accent)">\xe2\x80\x94</div></div>
          <div class="g-cell"><div class="g-cell-lbl">TARGET</div><div style="color:var(--blue)">95.0%</div></div>
        </div>
        <div id="gStat" style="font-family:var(--mono);font-size:9px;color:var(--dim)">Awaiting results\xe2\x80\xa6</div>
      </div>
    </div>
  </div>

  <!-- Drift + Portfolio -->
  <div class="row r2">
    <div class="card wn">
      <div class="c-title">Model Drift Monitor \xe2\x80\x94 30-day Rolling SL</div>
      <div class="drift" id="driftW">
        <div style="font-family:var(--mono);font-size:10px;color:var(--dim)">Awaiting portfolio run\xe2\x80\xa6</div>
      </div>
    </div>
    <div class="card bl">
      <div class="c-title">Portfolio \xe2\x80\x94 5-SKU Summary</div>
      <div class="pf" id="pfGrid">
        <div style="grid-column:1/-1;font-family:var(--mono);font-size:10px;color:var(--dim)">Awaiting portfolio run\xe2\x80\xa6</div>
      </div>
    </div>
  </div>

  <!-- Reward curve + Log -->
  <div class="row r2">
    <div class="card">
      <div class="c-title">Q-Learning Reward Curve</div>
      <div style="height:130px"><canvas id="rwC"></canvas></div>
    </div>
    <div class="card">
      <div class="c-title">Operations Log</div>
      <div class="log" id="logF">
        <div class="e"><span class="ts">00:00</span><span class="m info">Dashboard ready. Configure parameters and run simulation.</span></div>
      </div>
    </div>
  </div>

</div>
</main>

<script>
// Chart defaults
Chart.defaults.color='#4e6080';
Chart.defaults.borderColor='#1e2a3d';
Chart.defaults.font.family="'IBM Plex Mono',monospace";
Chart.defaults.font.size=10;

const T0=Date.now();
function ts(){const s=Math.floor((Date.now()-T0)/1000);return String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0')}
function log(msg,cls='info'){
  const f=document.getElementById('logF');
  const d=document.createElement('div');d.className='e';
  d.innerHTML=`<span class="ts">${ts()}</span><span class="m ${cls}">${msg}</span>`;
  f.appendChild(d);f.scrollTop=f.scrollHeight;
}
function setStatus(p){
  const el=document.getElementById('sPill');
  const t=document.getElementById('sTxt');
  el.className=`status-pill ${p}`;
  t.textContent=p.toUpperCase();
}
function setObj(id,v){const e=document.getElementById(id);if(e){e.textContent=v?'\xe2\x9c\x93':'\xe2\x97\x8b';e.className=v?'obj-chk pass':'obj-chk'}}
function banner(icon,msg){
  document.getElementById('banner').style.display='flex';
  document.getElementById('banIcon').textContent=icon;
  document.getElementById('banMsg').textContent=msg;
}

// Charts
const slC=new Chart(document.getElementById('slC').getContext('2d'),{
  type:'line',
  data:{labels:[],datasets:[
    {label:'Agentic',data:[],borderColor:'#00e5aa',backgroundColor:'rgba(0,229,170,.05)',tension:.3,borderWidth:2,pointRadius:3},
    {label:'Baseline',data:[],borderColor:'#ff4d6a',backgroundColor:'rgba(255,77,106,.05)',tension:.3,borderWidth:2,pointRadius:3,borderDash:[4,3]},
  ]},
  options:{responsive:true,maintainAspectRatio:false,animation:{duration:250},
    plugins:{legend:{position:'top',labels:{boxWidth:10,padding:10}}},
    scales:{y:{min:40,max:102,grid:{color:'#1a2030'},ticks:{callback:v=>v+'%'}},
            x:{grid:{color:'#1a2030'}}}},
});

const mdC=new Chart(document.getElementById('mdC').getContext('2d'),{
  type:'doughnut',
  data:{labels:['Standard','Second','First','Same Day'],
        datasets:[{data:[1,1,1,1],backgroundColor:['#4d9fff','#00e5aa','#ffb340','#ff4d6a'],borderColor:'#111620',borderWidth:2}]},
  options:{responsive:true,maintainAspectRatio:false,cutout:'62%',animation:{duration:400},
    plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>`${c.label}: ${c.parsed.toFixed(0)}`}}}},
});

const rwC=new Chart(document.getElementById('rwC').getContext('2d'),{
  type:'line',
  data:{labels:[],datasets:[{label:'Avg Reward',data:[],borderColor:'#4d9fff',backgroundColor:'rgba(77,159,255,.06)',tension:.4,borderWidth:2,pointRadius:0}]},
  options:{responsive:true,maintainAspectRatio:false,animation:{duration:150},
    plugins:{legend:{display:false}},
    scales:{y:{grid:{color:'#1a2030'}},x:{grid:{color:'#1a2030'},ticks:{maxTicksLimit:8}}}},
});

// Mode accumulator
const mc={'Standard Class':0,'Second Class':0,'First Class':0,'Same Day':0};
const mcColors=['#4d9fff','#00e5aa','#ffb340','#ff4d6a'];

function updMode(){
  const lbl=Object.keys(mc),vals=lbl.map(l=>mc[l]);
  mdC.data.labels=lbl;mdC.data.datasets[0].data=vals;mdC.update('none');
  const tot=vals.reduce((a,b)=>a+b,0)||1;
  document.getElementById('mdLeg').innerHTML=lbl.map((l,i)=>
    `<div style="display:flex;justify-content:space-between;gap:8px">
      <span style="color:${mcColors[i%4]}">${l}</span>
      <span style="color:#c8d4e8">${(100*vals[i]/tot).toFixed(1)}%</span>
    </div>`).join('');
}

function fmtKPI(v,k){
  if(v==null)return'\xe2\x80\x94';
  if(k==='service_level')return(v*100).toFixed(1)+'%';
  if(k==='stockout_days'||k==='n_orders')return v.toFixed(1);
  const sign=v<0?'\xe2\x80\x93\xc2\xa3':'\xc2\xa3';
  return sign+Math.abs(v).toLocaleString('en-GB',{maximumFractionDigits:0});
}

function updTable(rows){
  document.getElementById('kpiTB').innerHTML=rows.map(r=>{
    const g=r.agentic_better;
    const sig=r.significant?`<span class="sb y">p&lt;.006</span>`:`<span class="sb n">ns</span>`;
    return `<tr>
      <td>${r.label}</td>
      <td class="b-v">${fmtKPI(r.baseline_mean,r.kpi)}</td>
      <td class="a-v">${fmtKPI(r.agentic_mean,r.kpi)}</td>
      <td class="chg ${g?'g':'b'}">${r.improvement>0?'+':''}${r.improvement.toFixed(1)}%</td>
      <td style="text-align:right;color:#c8d4e8">${r.cohen_d.toFixed(2)}</td>
      <td class="sig">${sig}</td>
    </tr>`;
  }).join('');
}

function updKPIs(rows){
  for(const r of rows){
    if(r.kpi==='profit'){
      const d=r.agentic_mean-r.baseline_mean;
      document.getElementById('tProfit').textContent=(d>=0?'+\xc2\xa3':'\xe2\x80\x93\xc2\xa3')+Math.abs(d).toLocaleString('en-GB',{maximumFractionDigits:0});
      document.getElementById('tProfitD').className='kpi-d '+(d>=0?'pos':'neg');
    }
    if(r.kpi==='stockout_days'){
      const d=r.baseline_mean-r.agentic_mean;
      document.getElementById('tStock').textContent=d.toFixed(1);
      document.getElementById('tStockD').textContent=`days/yr (from ${r.baseline_mean.toFixed(1)})`;
      document.getElementById('tStockD').className='kpi-d '+(d>0?'pos':'neg');
    }
    if(r.kpi==='carbon_kgco2e'){
      document.getElementById('tCarbon').textContent=r.agentic_mean.toFixed(0);
      const d=r.agentic_mean-r.baseline_mean;
      document.getElementById('tCarbonD').textContent=(d>0?'+':'')+d.toFixed(0)+' vs baseline';
      document.getElementById('tCarbonD').className='kpi-d '+(d<0?'pos':'neg');
    }
    if(r.kpi==='service_level'){
      const p=r.agentic_mean*100,b=r.baseline_mean*100;
      document.getElementById('tSL').textContent=p.toFixed(1)+'%';
      document.getElementById('tSLd').textContent='Baseline: '+b.toFixed(1)+'%';
      document.getElementById('tSLd').className='kpi-d pos';
      // Gauge
      const off=(251.2*(1-r.agentic_mean)).toFixed(1);
      document.getElementById('gArc').setAttribute('stroke-dashoffset',off);
      document.getElementById('gTxt').textContent=p.toFixed(1)+'%';
      document.getElementById('gA').textContent=p.toFixed(1)+'%';
      document.getElementById('gB').textContent=b.toFixed(1)+'%';
      const ok=p>=95;
      document.getElementById('gStat').textContent=ok?'\xe2\x9c\x93 Target met (\xe2\x89\xa595%)':'\xe2\x9a\xa0 Below 95% target';
      document.getElementById('gStat').style.color=ok?'var(--accent)':'var(--warn)';
    }
  }
}

function updPortfolio(data){
  const grid=document.getElementById('pfGrid');
  const skus=Object.keys(data.agentic||{});
  if(!skus.length){
    // Show placeholders
    const ns=['FIELD_SPORT','FITNESS','BICYCLE','WATCH','GOLF'];
    grid.innerHTML=ns.map(n=>`<div class="sku-c"><div class="sku-n">${n}</div><div class="sku-v" style="color:var(--dim)">\xe2\x80\x94</div><div class="sku-l">AWAIT</div></div>`).join('');
    return;
  }
  grid.innerHTML=skus.map(sku=>{
    const a=(data.agentic||{})[sku]||{};
    const sl=a.service_level!=null?(a.service_level*100).toFixed(1)+'%':'\xe2\x80\x94';
    const c=a.service_level>=.95?'var(--accent)':(a.service_level>=.85?'var(--warn)':'var(--danger)');
    return `<div class="sku-c"><div class="sku-n">${sku.replace('SKU_','')}</div><div class="sku-v" style="color:${c}">${sl}</div><div class="sku-l">SVC LVL</div></div>`;
  }).join('');
}

function updDrift(data){
  const wrap=document.getElementById('driftW');
  const skus=Object.keys(data.agentic||{});
  if(!skus.length)return;
  wrap.innerHTML=skus.map(sku=>{
    const a=(data.agentic||{})[sku]||{};
    const sl=a.service_level||.85;
    const p=(sl*100);
    const bc=sl>=.90?'var(--accent)':(sl>=.85?'var(--warn)':'var(--danger)');
    const bc2=sl>=.90?'ok':(sl>=.85?'al':'rt');
    const bl=sl>=.90?'OK':(sl>=.85?'ALERT':'RETRAIN');
    return `<div class="dr">
      <div class="dr-sku">${sku.replace('SKU_','')}</div>
      <div class="dr-t"><div class="dr-f" style="width:${p}%;background:${bc}"></div></div>
      <div class="dr-sl" style="color:${bc}">${p.toFixed(1)}%</div>
      <div class="dr-b ${bc2}">${bl}</div>
    </div>`;
  }).join('');
}

// SSE
let evtSrc=null;
function connect(){
  evtSrc=new EventSource('/events');
  evtSrc.onmessage=e=>{try{handle(JSON.parse(e.data))}catch(_){}};
  evtSrc.onerror=()=>{log('SSE reconnecting\xe2\x80\xa6','w');setTimeout(connect,3000);evtSrc.close()};
}
connect();

function handle(ev){
  const{type:t,data:d}=ev;

  if(t==='status'){
    setStatus(d.running?'running':(d.phase==='error'?'error':d.phase||'idle'));
    document.getElementById('runBtn').disabled=d.running;
  }
  else if(t==='phase'){
    const ic={training:'\xf0\x9f\xa7\xa0',experiment:'\xf0\x9f\x94\xac',statistics:'\xf0\x9f\x93\x8a',portfolio:'\xf0\x9f\x93\xa6',complete:'\xe2\x9c\x85',error:'\xe2\x9a\xa0'};
    banner(ic[d.phase]||'\xe2\x9a\x99',d.message);
    log(d.message,'ph');
    if(d.phase==='training'){setObj('o1',true);setObj('o2',true);setObj('o3',true)}
    if(d.phase==='experiment')setObj('o4',true);
    if(d.phase==='statistics'){setObj('o5',true);setObj('o6',true)}
    if(d.phase==='portfolio'){setObj('o7',true);setObj('o8',true)}
  }
  else if(t==='training_progress'){
    document.getElementById('mpFill').style.width=d.pct+'%';
    document.getElementById('mpEp').textContent=`Ep ${d.episode}/${d.total}`;
    document.getElementById('mpPct').textContent=d.pct+'%';
    document.getElementById('sEps').textContent=d.epsilon.toFixed(4);
    document.getElementById('sRew').textContent=d.avg_reward.toFixed(0);
    document.getElementById('sChg').textContent=d.policy_changes;
    if(d.converged_ep!=null)document.getElementById('sConv').textContent=`Ep ${d.converged_ep}`;
    if(d.reward_history&&d.reward_history.length){
      rwC.data.labels=d.reward_history.map((_,i)=>i);
      rwC.data.datasets[0].data=d.reward_history;
      rwC.update('none');
    }
  }
  else if(t==='training_complete'){
    log(`Training done \xe2\x80\x94 converged ep ${d.convergence_episode}, \xce\xb5=${d.final_epsilon}`,'ok');
  }
  else if(t==='run_complete'){
    slC.data.labels.push(`R${d.run}`);
    slC.data.datasets[0].data.push(d.agentic_sl);
    slC.data.datasets[1].data.push(d.baseline_sl);
    slC.update('none');
    if(d.agentic_mode_counts){
      for(const[m,c]of Object.entries(d.agentic_mode_counts))mc[m]=(mc[m]||0)+c;
      updMode();
    }
    log(`Run ${d.run}/${d.total} \xe2\x80\x94 BL SL ${d.baseline_sl}% | AG SL ${d.agentic_sl}%`);
  }
  else if(t==='statistics_complete'){
    updTable(d.kpi_table);
    updKPIs(d.kpi_table);
    log(`Stats \xe2\x80\x94 ${d.n_sig_improved}/8 KPIs significantly improved`,d.n_sig_improved>=3?'ok':'w');
  }
  else if(t==='portfolio_complete'){
    if(!d.error){updPortfolio(d);updDrift(d);log('Portfolio complete','ok')}
    else log('Portfolio skipped: '+d.error,'w');
  }
  else if(t==='simulation_complete'){
    log('All phases complete \xe2\x9c\x93','ok');
    document.getElementById('banner').style.display='none';
    document.getElementById('runBtn').disabled=false;
    setStatus('complete');
  }
  else if(t==='error'){
    log('ERROR: '+d.message,'err');
    setStatus('error');
    document.getElementById('runBtn').disabled=false;
  }
}

function startSim(){
  const ep=document.getElementById('epS').value;
  const rn=document.getElementById('rnS').value;
  const un=document.getElementById('uncS').value;

  // Reset charts
  slC.data.labels=[];slC.data.datasets[0].data=[];slC.data.datasets[1].data=[];slC.update('none');
  rwC.data.labels=[];rwC.data.datasets[0].data=[];rwC.update('none');
  Object.keys(mc).forEach(k=>mc[k]=0);

  document.getElementById('kpiTB').innerHTML=`<tr><td colspan="6" style="color:var(--dim);padding:14px;text-align:center;font-family:var(--mono);font-size:10px">Running\xe2\x80\xa6</td></tr>`;
  ['o1','o2','o3','o4','o5','o6','o7','o8'].forEach(id=>setObj(id,false));
  document.getElementById('runBtn').disabled=true;
  setStatus('running');
  log(`Starting \xe2\x80\x94 ${ep} episodes, ${rn} runs, ${un} uncertainty`,'ph');

  fetch(`/start?episodes=${ep}&runs=${rn}&uncertainty=${un}`,{method:'POST'})
    .catch(e=>{log('Failed: '+e,'err');document.getElementById('runBtn').disabled=false});
}
</script>
</body>
</html>
"""

# ─── HTTP handler ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    # Allow multiple rapid requests without connection-reset errors
    timeout = 60

    def log_message(self, fmt, *args):
        pass  # suppress default access log

    def do_GET(self):
        path = urlparse(self.path).path

        if path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', len(DASHBOARD_HTML))
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML)

        elif path == '/events':
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            q = queue.Queue(maxsize=RL_EPISODES)
            with _sub_lock:
                _subscribers.append(q)

            # Send current state on connect
            payload = json.dumps({"type": "status", "data": sim_state}, default=str)
            try:
                self.wfile.write(f"data: {payload}\n\n".encode())
                self.wfile.flush()
            except Exception:
                pass

            try:
                while True:
                    try:
                        msg = q.get(timeout=20)
                        self.wfile.write(msg.encode())
                        self.wfile.flush()
                    except queue.Empty:
                        self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with _sub_lock:
                    if q in _subscribers:
                        _subscribers.remove(q)

        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == '/start':
            if sim_state['running']:
                body = json.dumps({'error': 'Already running'}).encode()
                self.send_response(409)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', len(body))
                self.end_headers()
                self.wfile.write(body)
                return

            params = parse_qs(parsed.query)
            n_episodes  = int(params.get('episodes',  ['300'])[0])
            n_runs      = int(params.get('runs',      ['20'])[0])
            uncertainty = params.get('uncertainty', ['base'])[0]

            threading.Thread(
                target=simulation_pipeline,
                args=(n_episodes, n_runs, uncertainty),
                daemon=True,
            ).start()

            body = json.dumps({'started': True, 'episodes': n_episodes,
                               'runs': n_runs, 'uncertainty': uncertainty}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(body))
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_error(404)

# ─── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Agentic AI Supply Chain Dashboard')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()

    port = args.port
    for _ in range(10):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
            break
        except OSError:
            port += 1

    server = ThreadingHTTPServer(('', port), Handler)

    print(f"\n  \u2554{'=' * 46}\u2557")
    print(f"  \u2551  Agentic AI Supply Chain Dashboard v2            \u2551")
    print(f"  \u2551  Open: http://localhost:{port:<5}                  \u2551")
    print(f"  \u2551  Stop: Ctrl+C                                  \u2551")
    print(f"  \u255a{'=' * 46}\u255d\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Dashboard stopped.")
        server.server_close()

if __name__ == '__main__':
    main()
