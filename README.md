# EG7302 — Agentic AI for Autonomous Inventory Management

> **Design, Development and Evaluation of an Agentic AI Prototype for Autonomous Decision-Making in Manufacturing and Logistics under Stochastic Uncertainty**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![EU AI Act Compliant](https://img.shields.io/badge/EU%20AI%20Act-Article%2014%20Compliant-orange.svg)]()
[![ISO 22301](https://img.shields.io/badge/ISO-22301%3A2019-blue.svg)]()
[![ISO/IEC 42001](https://img.shields.io/badge/ISO%2FIEC-42001%3A2023-blue.svg)]()

**University of Leicester · EG7302 Engineering Management Project · April 2026**  
**Author:** Suhas Mulukatte Kenchegowda (279055121)  
**Supervisor:** Timothy C Pearce

---

## Overview

This repository contains the complete implementation of a **five-agent Agentic AI prototype** for autonomous supply chain inventory management. The system uses tabular Q-learning to learn disruption-aware ordering policies, validated against a realistic stochastic simulation calibrated to the DataCo Global Supply Chain Dataset.

The project makes **three original contributions**:

1. **Regime-aware state encoding** (`demand_rate_bin`) — eliminates negative transfer across demand regimes, producing +29.4pp service level improvement at high demand versus a system without regime encoding
2. **Mixed-condition checkpoint validation** — prevents a validated −10.5pp regression from base-only validation seeds, producing training stability of 98.81% ±0.56pp across five independent seeds
3. **Dual-standard governance framework** — first quantified EU AI Act Article 14 + ISO/IEC 42001:2023 compliance mapping for RL inventory management, with measurable HITL activity rates of 4.1–23.9 reviews/year

---

## Key Results

| Condition | Baseline SL | Agentic SL | Δ pp | Cohen's d | Wilcoxon p |
|-----------|------------|-----------|------|-----------|------------|
| C1 Base/Poisson λ=15 | 86.06% ±2.81 | 99.35% ±1.01 | **+13.3** | 5.09 | 1.86×10⁻⁹ |
| C2 High/Poisson λ=25 | 51.92% ±3.89 | 96.05% ±2.48 | **+44.1** | 10.27 | 1.86×10⁻⁹ |
| C3 Base/NegBin λ=15 | 83.59% ±3.08 | 99.26% ±0.84 | **+15.7** | 4.73 | 1.86×10⁻⁹ |
| C4 High/NegBin λ=25 | 51.20% ±3.21 | 94.99% ±3.11 | **+43.8** | 12.92 | 1.86×10⁻⁹ |
| C5 Seasonal | 81.78% | 99.27% | **+17.5** | 6.42 | 1.86×10⁻⁹ |

All results: n=30 paired Monte Carlo runs · Bonferroni-corrected α_BF = 0.0125 · 30/30 wins in all conditions

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AGENTIC AI SYSTEM                           │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │   Adaptive   │  │  RL Inventory│  │   Production       │    │
│  │  Forecaster  │→ │    Agent     │→ │   Scheduler        │    │
│  │ Holt + ARIMA │  │ Q-table 1,344│  │ SPT + Tardiness    │    │
│  └──────────────┘  │    states    │  └────────────────────┘    │
│                    └──────┬───────┘                            │
│                           │                                     │
│                    ┌──────▼───────┐  ┌────────────────────┐    │
│                    │  Governance  │  │    Logistics       │    │
│                    │    Agent     │→ │   Optimiser        │    │
│                    │  HITL + SHA  │  │ Cost+Carbon Min    │    │
│                    └──────────────┘  └────────────────────┘    │
│                                                                 │
│              EU AI Act Article 14(4) Compliance Layer           │
└─────────────────────────────────────────────────────────────────┘
```

### MDP Specification

| Component | Specification |
|-----------|--------------|
| **State \|S\|** | 1,344 states (8×7×4×2×3: inventory × forecast × pending × disruption × regime) |
| **Actions \|A\|** | 8 discrete order quantities: {0, 25, 50, 75, 100, 150, 200, 250} units |
| **Transition P** | Model-free — estimated via Bellman updates |
| **Reward R** | Revenue − holding − stockout − order − shipping − carbon penalty |
| **Discount γ** | 0.95 (12-day disrupted lead-time horizon) |

---

## Repository Structure

```
EG7302_Agentic_AI/
│
├── source_code/
│   ├── config.py                  # All parameters and constants
│   ├── stochastic_env.py          # Simulation environment (Poisson, log-normal LT, Bernoulli disruptions)
│   ├── inventory_agent.py         # Tabular Q-learning agent (1,344-state Q-table)
│   ├── forecasting_agent.py       # Holt DES + ARIMA adaptive forecaster
│   ├── logistics_agent.py         # Logistics optimiser (cost + carbon composite)
│   ├── production_agent.py        # Production scheduler (SPT + weighted tardiness)
│   ├── governance_agent.py        # EU AI Act Article 14 HITL gateway
│   ├── human_approval.py          # HITL trigger logic and SHA-256 audit trail
│   ├── drift_monitor.py           # Real-time policy drift detection
│   ├── simulation_runner.py       # Experiment orchestration + KPI tracking
│   ├── statistical_analysis.py    # Wilcoxon tests, Cohen's d, Bonferroni correction
│   ├── data_processor.py          # DataCo dataset calibration
│   ├── visualisation.py           # All result figures (run directly)
│   ├── pareto_analysis.py         # Carbon-profit Pareto frontier sweep
│   ├── main.py                    # Primary entry point
│   ├── portfolio_runner.py        # Multi-SKU portfolio experiments
│   │
│   ├── llm_inventory_agent.py     # Claude API LLM comparison agent
│   ├── ollama_inventory_agent.py  # Local Ollama (gemma2:2b) agent
│   ├── ollama_fast_agent.py       # Cached + parallel Ollama agent
│   ├── run_fast_comparison.py     # Three-way comparison runner (C1–C4)
│   └── run_ollama_comparison.py   # Standard comparison runner
│
├── results/
│   ├── definitive_results_v68.json  # Ground truth — all experimental results
│   ├── EG7302_Results.xlsx          # Formatted 10-sheet results workbook
│   ├── primary_results.csv
│   ├── ablation.csv
│   ├── llm_comparison.csv
│   ├── sS_comparison.csv
│   ├── profit_sensitivity.csv
│   └── audit_trail_sample.json      # Sample SHA-256 governance log
│
├── figures/
│   ├── fig3_1_sequence_diagram_HD.png   # Multi-agent sequence diagram (1920px)
│   ├── fig3_1_sequence_diagram_HD.html  # Interactive HTML version
│   ├── fig5_2_simulation_trace.html     # Days 41–49 simulation trace
│   ├── fig_state_space_diagram.html     # 1,344-state discretisation
│   └── fig_risk_matrix_appendix_d.html  # ISO 31000 risk matrix
│
├── dissertation/
│   └── EG7302_Final_Report_v11.docx    # Submitted dissertation
│
└── README.md
```

---

## Quick Start

### Prerequisites

```bash
Python 3.10+
pip install numpy pandas scipy matplotlib seaborn openpyxl
```

### Run Primary Experiments (C1–C4)

```bash
cd source_code
python main.py
```

This runs n=30 paired Monte Carlo simulations across all four primary conditions and outputs results to `results/`.

### Reproduce All Figures

```bash
python visualisation.py
```

Generates all 10 result figures in `figures/`. Reads directly from `results/definitive_results_v68.json`.

### Run LLM Comparison (requires Ollama)

```bash
# Install Ollama: https://ollama.ai
ollama pull gemma2:2b

# All four conditions with caching + parallelism
python run_fast_comparison.py
```

### Run Individual Components

```python
from stochastic_env import StochasticSupplyChainEnv
from inventory_agent import QLearningInventoryAgent
from config import *

# Initialise environment
env = StochasticSupplyChainEnv(
    demand_lambda=DEMAND_LAMBDA_BASE,
    disruption_prob=DISRUPTION_PROB_BASE
)

# Train agent (2,000 episodes with domain randomisation)
agent = QLearningInventoryAgent()
agent.train(env, n_episodes=2000, domain_randomise=True, verbose=True)

# Evaluate
results = agent.evaluate(env, n_runs=30, seed_start=0)
```

---

## Experimental Design

### Training

- **Episodes:** 2,000 with ε-greedy exploration (ε: 1.0 → 0.05 over 467 episodes)
- **Domain randomisation:** λ ∈ [6, 26] u/day, LT ∈ [3.5, 7.0] days, p_dis ∈ [0.01, 0.10]/day
- **Checkpoint selection:** Mixed-condition validation (10 base + 5 high-demand seeds) — prevents base-only bias

### Evaluation

| Experiment | Description | n |
|------------|-------------|---|
| Primary (C1–C4) | 2×2 factorial: Base/High demand × Poisson/NegBin | 30 runs/condition |
| Three-way | ROP vs optimised (s,S) vs Agentic | 30 runs/condition |
| Ablation | Each agent removed individually | 30 runs/condition |
| Rate-bin ablation | With vs without demand_rate_bin | 30 runs per λ level |
| C5 Seasonal | Weekly ±30% + Q4 ×1.2 seasonal demand | 30 runs |
| Multi-seed stability | 5 independent training seeds | 30 eval runs/seed |
| OOD generalisation | λ=4.2 (below) and λ=28.6 (above training range) | 20 runs |
| Forecaster comparison | Holt vs ARIMA across demand regimes | 30 runs |

### Statistical Methods

- **Test:** Wilcoxon signed-rank (non-parametric, paired — SL is bounded, non-normal)
- **Family-wise correction:** Bonferroni, α_BF = 0.05/4 = 0.0125 for four primary conditions
- **Effect size:** Cohen's d = mean(diff) / std(diff) for all conditions
- **Power:** n=30 provides 95.5% power to detect Δ=2pp at σ=3pp

---

## Governance Framework

Five compliance mechanisms mapped to EU AI Act Article 14 and ISO/IEC 42001:2023:

| Mechanism | EU AI Act | ISO/IEC 42001 | Activity Rate |
|-----------|-----------|----------------|---------------|
| HITL Approval Gateway | Art. 14(4)(d) | §8.4 Operational control | 4.1–23.9 reviews/year |
| Decision Audit Trail | Art. 14(4)(c) | §8.4 Records | 365 entries/year |
| Drift Monitor | Art. 14(4)(b) | §10.2 Nonconformity | 0 alerts (stable) |
| Governance Order Cap | Art. 14(4)(a) | §6.1.2 Risk treatment | Always active |
| Dynamic Safety Stock Floor | Art. 14(4)(a) | §6.1.2 Risk treatment | As needed |

HITL activity rates scale 5.8× from base (p=0.02) to high (p=0.08) disruption conditions — satisfying Art. 14(4)(b)'s proportionality requirement.

---

## Calibration

All simulation parameters calibrated from the **DataCo Global Supply Chain Dataset** (Constante et al., 2019 — 180,519 order records):

| Parameter | Value | Source |
|-----------|-------|--------|
| Unit cost | £141.23 | DataCo Field & Stream SKU financial records |
| Holding cost rate | 25% per annum | DataCo-derived |
| Base demand λ | 15 u/day | DataCo primary SKU (dispersion index = 1.05, confirming Poisson) |
| Lead time | Log-normal (μ=5d, σ=1.5d) | DataCo shipping records |
| Disruption probability (base) | p = 0.02/day | ISO 22301 Annex A infrequent category |
| Disruption probability (high) | p = 0.08/day | ISO 22301 Annex A frequent category |
| Disruption duration | Exponential (mean = 3d) | ISO 22301 Clause 8.3 Tier 2 |
| Carbon shadow price | £0.50/kgCO₂e | HM Treasury Green Book (2023) |

---

## Business Case

At calibrated stockout cost rate (SCR = 2.0×):

| Condition | Annual Benefit | Payback Period |
|-----------|---------------|----------------|
| C1 (Base demand) | **£205,216/year** | < 4 months |
| C2 (High demand) | **£944,892/year** | < 2 months |

Carbon-profit Pareto threshold: £2.00/kgCO₂e — the penalty rate at which Standard Class shipping dominates with no service level loss.

---

## Known Limitations

1. **Single-echelon scope** — validated for one supplier-to-warehouse link; multi-echelon extension is the primary future work
2. **Perfect disruption signal observability** — the disruption flag is modelled as binary and immediate; real-world signals may be delayed or noisy
3. **Simulation-to-real gap** — all results are simulation-validated; a deployment pilot is required before production use
4. **Discrete state-space** — the 1,344-state Q-table cannot represent continuous demand features without discretisation loss; DQN with attention-based XAI is the recommended next step for scalability

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.24 | Array operations, random seeding |
| `pandas` | ≥2.0 | Data processing, results storage |
| `scipy` | ≥1.11 | Wilcoxon test, KS test, statistical analysis |
| `matplotlib` | ≥3.7 | Result figure generation |
| `seaborn` | ≥0.12 | Heatmaps, visualisation |
| `openpyxl` | ≥3.1 | Results workbook (.xlsx) |
| `requests` | ≥2.31 | Ollama API (LLM comparison only) |

---

## Results Reproducibility

All primary results are stored in `results/definitive_results_v68.json`. To verify:

```bash
python -c "
import json
with open('results/definitive_results_v68.json') as f:
    r = json.load(f)
for cond, data in r['conditions'].items():
    print(f'{cond}: baseline={data[\"baseline_sl_mean\"]:.2f}%, agentic={data[\"agentic_sl_mean\"]:.2f}%, d={data[\"cohens_d\"]}')
"
```

Expected output:
```
C1_base_poisson:  baseline=86.06%, agentic=99.35%, d=5.09
C2_high_poisson:  baseline=51.92%, agentic=96.05%, d=10.27
C3_base_negbin:   baseline=83.59%, agentic=99.26%, d=4.73
C4_high_negbin:   baseline=51.20%, agentic=94.99%, d=12.92
```

---

## Citation

```bibtex
@mastersthesis{kenchegowda2026agentic,
  author    = {Suhas Mulukatte Kenchegowda},
  title     = {Design, Development and Evaluation of an Agentic AI Prototype
               for Autonomous Decision-Making in Manufacturing and Logistics
               under Stochastic Uncertainty},
  school    = {University of Leicester},
  year      = {2026},
  month     = {April},
  type      = {MSc Dissertation},
  note      = {EG7302 Engineering Management Project}
}
```

---

## Key References

- Boute, R.N. et al. (2022) Deep reinforcement learning for inventory control: A roadmap. *European Journal of Operational Research*, 298(2), pp. 401–412.
- Gijsbrechts, J. et al. (2022) Can deep reinforcement learning improve inventory management? *Manufacturing & Service Operations Management*, 24(3), pp. 1349–1368.
- Jannelli, V. et al. (2025) Agentic LLMs in the supply chain. *International Journal of Production Research*.
- Rolf, B. et al. (2023) A review on reinforcement learning algorithms in supply chain management. *International Journal of Production Research*, 61(20), pp. 7151–7179.
- Wiesemann, W., Kuhn, D. and Sim, M. (2014) Distributionally robust convex optimization. *Operations Research*, 62(6), pp. 1358–1376.
- European Commission (2024) Regulation (EU) 2024/1689 — EU Artificial Intelligence Act.
- ISO/IEC 42001:2023 — Artificial Intelligence Management System Standard.
- ISO 22301:2019 — Business Continuity Management Systems.

---

## Licence

This project is released under the MIT Licence. See [LICENSE](LICENSE) for details.

The DataCo Global Supply Chain Dataset is used under its original Mendeley Data licence (Constante et al., 2019). Dataset not included in this repository — download from [Mendeley Data](https://data.mendeley.com/datasets/8gx2fvg2k6/2).

---

*University of Leicester · School of Engineering · EG7302 Engineering Management Project · 2026*
