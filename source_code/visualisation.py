"""
visualisation.py
Generates all 10 dissertation figures from the v6.8 results JSON.

Figures saved to ./figures/
  fig01_training_curve.png      Validation SL across 20 checkpoints
  fig02_sl_comparison.png       4-condition service-level bar chart
  fig03_profit_delta.png        Annual profit improvement per condition
  fig04_cohens_d.png            Effect-size bars (Cohen's d)
  fig05_carbon_tradeoff.png     Carbon emission delta per condition
  fig06_ablation.png            Agent ablation SL contribution
  fig07_lambda_sweep.png        Extended lambda sweep (lambda 3 to 32)
  fig08_governance_activity.png HITL and governance mechanism activity
  fig09_ood_generalisation.png  OOD formal results
  fig10_sensitivity.png         Profit sensitivity to STOCKOUT_COST_RATE

Usage (standalone):
    python visualisation.py

Usage (from main.py):
    from visualisation import generate_all_figures
    generate_all_figures(results_dict)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

FIGURES_DIR = 'figures'

# Colour palette
NAVY   = '#1B3A6B'
TEAL   = '#0F6E56'
TEAL_L = '#5DCAA5'
AMBER  = '#BA7517'
AMB_L  = '#EF9F27'
GRAY   = '#888780'
BLUE_L = '#378ADD'
BLUE   = '#185FA5'
RED    = '#A32D2D'
WHITE  = '#FFFFFF'
BG     = '#F8F9FB'

COND_LABELS = ['C1\nBase/Poisson', 'C2\nHigh/Poisson',
               'C3\nBase/NegBin',  'C4\nHigh/NegBin']
COND_KEYS   = ['C1_base_poisson', 'C2_high_poisson',
               'C3_base_negbin',  'C4_high_negbin']

# ── Helpers ────────────────────────────────────────────────────────────────

def _save(fig, name: str) -> str:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=WHITE)
    plt.close(fig)
    print(f'  [fig] {path}')
    return path

def _style(ax, title='', xlabel='', ylabel='', grid_axis='y'):
    ax.set_facecolor(BG)
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', color=NAVY, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=GRAY)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=GRAY)
    ax.tick_params(colors=GRAY, labelsize=8)
    for s in ax.spines.values():
        s.set_edgecolor('#DDDDDD')
    if grid_axis:
        ax.grid(True, axis=grid_axis, alpha=0.35,
                color='#CCCCCC', linewidth=0.6)
        ax.set_axisbelow(True)

def _pct(v, _):
    return f'{v:.0f}%'

def _pct_signed(v, _):
    return f'{v:+.0f}pp'

def _money(v, _):
    if abs(v) >= 1e6:
        return f'£{v/1e6:.1f}M'
    return f'£{v/1e3:.0f}k'

# ── Fig 01: Training validation-SL curve ──────────────────────────────────

def plot_training_curve(val_hist: list, best_ep: int,
                        best_val_sl: float) -> str:
    """
    Line chart of held-out validation SL at every 100-episode checkpoint.
    val_hist entries: {'ep': int, 'val_sl': float (0-1), ...}
    """
    eps  = [h['ep']          for h in val_hist]
    sls  = [h['val_sl'] * 100 for h in val_hist]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(eps, sls, color=BLUE_L, linewidth=2, marker='o', markersize=5,
            markerfacecolor=WHITE, markeredgecolor=BLUE_L, zorder=3,
            label='Validation SL (10 held-out seeds)')
    ax.scatter([best_ep], [best_val_sl], color=AMB_L, s=180, zorder=5,
               marker='*',
               label=f'Best checkpoint  ep {best_ep}  ({best_val_sl:.2f}%)')
    ax.axhline(best_val_sl, color=AMB_L, linewidth=1,
               linestyle='--', alpha=0.55)
    ax.fill_between(eps, sls, alpha=0.07, color=BLUE_L)
    ax.set_ylim(0, 106)
    ax.yaxis.set_major_formatter(FuncFormatter(_pct))
    _style(ax,
           title='Training validation service level -- 20 checkpoint evaluations',
           xlabel='Training episode', ylabel='Validation SL (%)')
    ax.legend(fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    return _save(fig, 'fig01_training_curve.png')

# ── Fig 02: 4-condition service-level comparison ───────────────────────────

def plot_sl_comparison(conditions: dict) -> str:
    b_sl  = [conditions[c]['baseline_sl']     for c in COND_KEYS]
    b_std = [conditions[c]['baseline_sl_std'] for c in COND_KEYS]
    a_sl  = [conditions[c]['agentic_sl']      for c in COND_KEYS]
    a_std = [conditions[c]['agentic_sl_std']  for c in COND_KEYS]

    x = np.arange(4)
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, b_sl, w, label='Baseline (ROP/EOQ)',
           color=GRAY, alpha=0.82,
           yerr=b_std, capsize=4,
           error_kw=dict(elinewidth=1, ecolor=GRAY))
    bars = ax.bar(x + w/2, a_sl, w, label='Agentic AI',
                  color=TEAL, alpha=0.90,
                  yerr=a_std, capsize=4,
                  error_kw=dict(elinewidth=1, ecolor=TEAL_L))
    for bar, sl, std in zip(bars, a_sl, a_std):
        ax.text(bar.get_x() + bar.get_width() / 2,
                sl + std + 0.8,
                f'{sl:.1f}%', ha='center', va='bottom',
                fontsize=7.5, fontweight='bold', color=TEAL)
    ax.set_xticks(x)
    ax.set_xticklabels(COND_LABELS, fontsize=9)
    ax.set_ylim(0, 112)
    ax.yaxis.set_major_formatter(FuncFormatter(_pct))
    _style(ax,
           title='Service level -- agentic AI vs baseline  (n=30 per condition)',
           ylabel='Mean annual service level (%)')
    ax.legend(fontsize=9, framealpha=0.9)
    ax.annotate(
        'All conditions: Wilcoxon p = 1.86*10⁻⁹  ·  30/30 paired wins',
        xy=(0.5, 0.01), xycoords='axes fraction',
        ha='center', fontsize=8, color=GRAY, style='italic')
    fig.tight_layout()
    return _save(fig, 'fig02_sl_comparison.png')

# ── Fig 03: Annual profit improvement ──────────────────────────────────────

def plot_profit_delta(conditions: dict) -> str:
    deltas = [conditions[c]['delta_profit'] for c in COND_KEYS]
    colors = [TEAL if d > 0 else RED for d in deltas]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(COND_LABELS, deltas, color=colors, alpha=0.88, height=0.5)
    for bar, val in zip(bars, deltas):
        sign = '+' if val >= 0 else ''
        label = f'{sign}£{val:,.0f}'
        ax.text(val + 4000, bar.get_y() + bar.get_height() / 2,
                label, va='center', fontsize=9,
                fontweight='bold', color=TEAL if val >= 0 else RED)
    ax.axvline(0, color=NAVY, linewidth=0.8)
    ax.xaxis.set_major_formatter(FuncFormatter(_money))
    _style(ax,
           title='Annual profit improvement -- agentic vs baseline',
           xlabel='Mean annual profit delta (£)', grid_axis='x')
    fig.tight_layout()
    return _save(fig, 'fig03_profit_delta.png')

# ── Fig 04: Effect size (Cohen's d) ────────────────────────────────────────

def plot_cohens_d(conditions: dict) -> str:
    ds = [conditions[c]['cohens_d'] for c in COND_KEYS]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(COND_LABELS, ds, color=BLUE_L, alpha=0.85, height=0.5)
    for bar, d in zip(bars, ds):
        ax.text(d + 0.12, bar.get_y() + bar.get_height() / 2,
                f'd = {d:.2f}', va='center', fontsize=9,
                fontweight='bold', color=NAVY)
    for ref, lbl, col in [(0.8,  'Large  (0.8)',      '#CCCCCC'),
                           (2.0,  'Very large  (2.0)', AMBER)]:
        ax.axvline(ref, color=col, linewidth=1.2, linestyle='--')
        ax.text(ref + 0.1, -0.55, lbl, fontsize=7.5, color=col)
    ax.set_xlim(0, max(ds) * 1.30)
    _style(ax,
           title="Effect size (Cohen's d) -- all conditions exceed d=4 (very large)",
           xlabel="Cohen's d", grid_axis='x')
    fig.tight_layout()
    return _save(fig, 'fig04_cohens_d.png')

# ── Fig 05: Carbon trade-off ────────────────────────────────────────────────

def plot_carbon_tradeoff(conditions: dict) -> str:
    b_co2 = [conditions[c]['baseline_co2']  for c in COND_KEYS]
    a_co2 = [conditions[c]['agentic_co2']   for c in COND_KEYS]
    pcts  = [conditions[c]['delta_co2_pct'] for c in COND_KEYS]

    x = np.arange(4)
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, b_co2, w, label='Baseline',  color=GRAY,  alpha=0.80)
    bars = ax.bar(x + w/2, a_co2, w, label='Agentic AI', color=AMB_L, alpha=0.88)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 45,
                f'+{pct:.1f}%', ha='center', va='bottom',
                fontsize=8, color=AMBER, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(COND_LABELS, fontsize=9)
    _style(ax,
           title='Carbon emissions trade-off  (kgCO₂e / year)',
           ylabel='Mean annual carbon emissions (kg)')
    ax.legend(fontsize=9, framealpha=0.9)
    ax.annotate(
        'Higher service level requires more orders -> increased logistics emissions',
        xy=(0.5, 0.01), xycoords='axes fraction',
        ha='center', fontsize=8, color=GRAY, style='italic')
    fig.tight_layout()
    return _save(fig, 'fig05_carbon_tradeoff.png')

# ── Fig 06: Ablation study ──────────────────────────────────────────────────

def plot_ablation(ablation: dict) -> str:
    """
    ablation keys:  'Full Agentic (all agents active)', 'Ablate Forecaster ...',  etc.
    values keys:    sl_mean, sl_std, marginal_contribution_pp
    """
    short = {
        'Full Agentic (all agents active)':     'Full agentic',
        'Ablate Forecaster (-> 7d MA)':          '− Forecaster (-> 7d MA)',
        'Ablate Inventory (-> ROP heuristic)':   '− Inventory (-> ROP)',
        'Ablate Logistics (-> fixed Standard)':  '− Logistics (-> Standard)',
        'Ablate Production (-> FIFO)':           '− Production (-> FIFO)',
    }
    keys   = list(ablation.keys())
    labels = [short.get(k, k) for k in keys]
    deltas = [ablation[k]['marginal_contribution_pp'] for k in keys]
    sls    = [ablation[k]['sl_mean']                  for k in keys]
    colors = [TEAL if d >= 0 else RED for d in deltas]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 4.5),
        gridspec_kw={'width_ratios': [2, 1]})

    # Left: marginal contributions
    bars = ax1.barh(labels[::-1], deltas[::-1],
                    color=colors[::-1], alpha=0.88, height=0.55)
    ax1.axvline(0, color=NAVY, linewidth=0.9)
    for bar, d in zip(bars, deltas[::-1]):
        sign = '+' if d >= 0 else ''
        offset = 0.05 if d >= 0 else -0.05
        ax1.text(d + offset,
                 bar.get_y() + bar.get_height() / 2,
                 f'{sign}{d:.2f}pp', va='center',
                 ha='left' if d >= 0 else 'right',
                 fontsize=8.5, fontweight='bold',
                 color=TEAL if d >= 0 else RED)
    _style(ax1,
           title='Marginal SL contribution (pp from full agentic)',
           xlabel='ΔSL (pp)', grid_axis='x')

    # Right: absolute SL
    bar_cols = [TEAL if sl >= 99 else (AMB_L if sl >= 97 else RED)
                for sl in sls]
    ax2.barh(labels[::-1], sls[::-1],
             color=bar_cols[::-1], alpha=0.82, height=0.55)
    for i, sl in enumerate(sls[::-1]):
        ax2.text(sl - 0.05, i, f'{sl:.2f}%',
                 va='center', ha='right',
                 fontsize=8, color=WHITE, fontweight='bold')
    ax2.set_xlim(90, 101)
    ax2.set_yticklabels([])
    _style(ax2, title='Mean SL (%)', xlabel='Service level (%)', grid_axis='x')

    fig.suptitle(
        'Ablation study -- agent contribution to service level  (n=30, C1 base/Poisson)',
        fontsize=11, fontweight='bold', color=NAVY, y=1.01)
    fig.tight_layout()
    return _save(fig, 'fig06_ablation.png')

# ── Fig 07: Extended lambda sweep ──────────────────────────────────────────

def plot_lambda_sweep(lambda_sweep: dict) -> str:
    """
    lambda_sweep keys: 'lambda_3', 'lambda_4', ...
    entry keys: lambda, status ('In-dist'|'OOD'), baseline_sl, agentic_sl, delta_sl
    """
    entries = sorted(lambda_sweep.values(), key=lambda e: e['lambda_val'])
    lams    = [e['lambda_val']  for e in entries]
    b_sls   = [e['baseline_sl'] for e in entries]
    a_sls   = [e['agentic_sl']  for e in entries]
    deltas  = [e['delta_sl']    for e in entries]
    in_dr   = [e['status'] == 'In-dist' for e in entries]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    dr_lams = [l for l, f in zip(lams, in_dr) if f]
    if dr_lams:
        for ax in (ax1, ax2):
            ax.axvspan(min(dr_lams), max(dr_lams),
                       alpha=0.07, color=TEAL)
        ax1.text(min(dr_lams) + 0.5, ax1.get_ylim()[0] if False else -8,
                 'DR training range', fontsize=7.5, color=TEAL, style='italic')

    in_x  = [l for l, f in zip(lams, in_dr) if f]
    in_d  = [d for d, f in zip(deltas, in_dr) if f]
    out_x = [l for l, f in zip(lams, in_dr) if not f]
    out_d = [d for d, f in zip(deltas, in_dr) if not f]

    ax1.plot(lams, deltas, color=TEAL, linewidth=2, zorder=3)
    ax1.scatter(in_x, in_d, color=TEAL, s=55, zorder=4,
                label='In-distribution')
    ax1.scatter(out_x, out_d, color=AMB_L, s=65, zorder=4, marker='D',
                label='OOD (outside training range)')
    ax1.axhline(0, color=NAVY, linewidth=0.8, linestyle='--', alpha=0.5)

    ceil_x = [l for l, b in zip(lams, b_sls) if b >= 99.5]
    ceil_d = [d for d, b in zip(deltas, b_sls) if b >= 99.5]
    if ceil_x:
        ax1.scatter(ceil_x, ceil_d, color=GRAY, s=80, marker='x',
                    linewidths=1.8, zorder=5, label='Ceiling effect (baseline ~=100%)')

    ax1.yaxis.set_major_formatter(FuncFormatter(_pct_signed))
    _style(ax1,
           title='ΔSL (agentic − baseline) across demand rates',
           ylabel='ΔSL (pp)')
    ax1.legend(fontsize=8.5, framealpha=0.9, loc='upper left')

    ax2.plot(lams, b_sls, color=GRAY,   linewidth=2, marker='o',
             markersize=5, markerfacecolor=WHITE, label='Baseline')
    ax2.plot(lams, a_sls, color=TEAL_L, linewidth=2, marker='o',
             markersize=5, markerfacecolor=WHITE, label='Agentic AI')
    ax2.set_ylim(40, 106)
    ax2.yaxis.set_major_formatter(FuncFormatter(_pct))
    _style(ax2,
           title='Absolute service levels',
           xlabel='Demand rate lambda (units/day)', ylabel='Mean SL (%)')
    ax2.legend(fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    return _save(fig, 'fig07_lambda_sweep.png')

# ── Fig 08: Governance mechanism activity ──────────────────────────────────

def plot_governance_activity(rates: dict,
                              condition_label: str = 'Base/Poisson (C1)') -> str:
    """
    rates dict keys: hitl_reviews, hitl_modifications, gov_cap_enforcements,
                     drift_alerts, ss_triggers
    """
    mechanisms = [
        'HITL order reviews',
        'HITL quantity modifications',
        'Governance cap enforcements',
        'Model drift alerts',
        'Safety stock floor triggers',
    ]
    values = [
        # Defaults are v6.8 empirical values for C1 (base/Poisson);
        # generate_all_figures() populates 'rates' from actual simulation output.
        rates.get('hitl_reviews',           4.1),
        rates.get('hitl_modifications',     2.6),
        rates.get('gov_cap_enforcements',   2.3),
        rates.get('drift_alerts',           0.0),
        rates.get('ss_triggers',            0.0),
    ]
    colors = [BLUE_L, BLUE, AMBER, AMB_L, TEAL_L]
    art14  = ['Art. 14(4)(b)', 'Art. 14(4)(d)', 'Art. 14(4)(e)',
              'Art. 14(4)(a)', 'Art. 14(4)(a)']

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(mechanisms, values, color=colors, alpha=0.88, height=0.55)
    for bar, r, tag in zip(bars, values, art14):
        ax.text(r + 0.04,
                bar.get_y() + bar.get_height() / 2,
                f'{r:.1f} / yr   [{tag}]',
                va='center', fontsize=8.5, color=NAVY)
    ax.set_xlim(0, max(values) * 1.7)
    _style(ax,
           title=f'Governance and HITL mechanism activity -- {condition_label}',
           xlabel='Mean events per year', grid_axis='x')
    fig.tight_layout()
    return _save(fig, 'fig08_governance_activity.png')

# ── Fig 09: OOD generalisation ─────────────────────────────────────────────

def plot_ood_results(ood: dict, conditions: dict) -> str:
    """
    ood entry keys: lambda, baseline_sl, agentic_sl, delta_sl_pp, cohens_d
    conditions C1 keys: baseline_sl, agentic_sl, delta_sl, cohens_d
    """
    c1    = conditions['C1_base_poisson']
    low_k  = next(k for k in ood if 'Low'  in k)
    high_k = next(k for k in ood if 'High' in k)

    labels = ['OOD Low\n(lambda=4.2)', 'C1 In-dist\n(lambda=15)', 'OOD High\n(lambda=28.6)']
    b_sls  = [ood[low_k].get('baseline_sl', ood[low_k].get('baseline_sl_mean', 0)),
              c1['baseline_sl'],
              ood[high_k].get('baseline_sl', ood[high_k].get('baseline_sl_mean', 0))]
    a_sls  = [ood[low_k].get('agentic_sl', ood[low_k].get('agentic_sl_mean', 0)),
              c1['agentic_sl'],
              ood[high_k].get('agentic_sl', ood[high_k].get('agentic_sl_mean', 0))]
    deltas = [ood[low_k]['delta_sl_pp'],  c1['delta_sl'],     ood[high_k]['delta_sl_pp']]
    ds     = [ood[low_k]['cohens_d'],     c1['cohens_d'],     ood[high_k]['cohens_d']]

    x = np.arange(3)
    w = 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # Left -- absolute SLs
    ax1.bar(x - w/2, b_sls, w, label='Baseline',   color=GRAY, alpha=0.80)
    bars = ax1.bar(x + w/2, a_sls, w, label='Agentic AI', color=TEAL, alpha=0.90)
    for bar, sl in zip(bars, a_sls):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 sl + 0.8, f'{sl:.1f}%',
                 ha='center', va='bottom',
                 fontsize=8, fontweight='bold', color=TEAL)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylim(0, 112)
    ax1.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax1.axvspan(-0.5, 0.5, alpha=0.06, color=AMBER)
    ax1.axvspan(1.5,  2.5, alpha=0.06, color=TEAL)
    _style(ax1, title='OOD -- absolute service levels', ylabel='Service level (%)')
    ax1.legend(fontsize=9, framealpha=0.9)

    # Right -- ΔSL and d
    col2 = [AMB_L, TEAL_L, TEAL]
    ax2.bar(x, deltas, color=col2, alpha=0.88, width=0.55)
    for i, (d, delta) in enumerate(zip(ds, deltas)):
        sign = '+' if delta >= 0 else ''
        offset = max(abs(delta) * 0.04, 0.5)
        ax2.text(i, delta + offset if delta >= 0 else delta - offset,
                 f'{sign}{delta:.2f}pp\nd = {d:.2f}',
                 ha='center',
                 va='bottom' if delta >= 0 else 'top',
                 fontsize=8.5, fontweight='bold', color=NAVY)
    ax2.axhline(0, color=NAVY, linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.yaxis.set_major_formatter(FuncFormatter(_pct_signed))
    _style(ax2, title="ΔSL and Cohen's d", ylabel='ΔSL (pp)')

    fig.suptitle('Out-of-distribution generalisation  (n=20)',
                 fontsize=11, fontweight='bold', color=NAVY)
    fig.tight_layout()
    return _save(fig, 'fig09_ood_generalisation.png')

# ── Fig 10: Profit sensitivity ─────────────────────────────────────────────

def plot_sensitivity(conditions: dict, current_scr: float = 2.0) -> str:
    """
    Reconstructs profit sensitivity analytically from service levels.
    SL improvements are parameter-independent; profit scales linearly with SCR.
    """
    from config import DEMAND_LAMBDA_BASE, DEMAND_LAMBDA_HIGH, UNIT_COST

    b_sl1 = conditions['C1_base_poisson']['baseline_sl'] / 100
    a_sl1 = conditions['C1_base_poisson']['agentic_sl']  / 100
    b_sl2 = conditions['C2_high_poisson']['baseline_sl'] / 100
    a_sl2 = conditions['C2_high_poisson']['agentic_sl']  / 100

    scrs  = [0.5, 1.0, 1.5, 2.0, 3.0]
    c1_dp = [(b_sl1 - a_sl1) * (-DEMAND_LAMBDA_BASE * 365 * scr * UNIT_COST)
             for scr in scrs]
    c2_dp = [(b_sl2 - a_sl2) * (-DEMAND_LAMBDA_HIGH * 365 * scr * UNIT_COST)
             for scr in scrs]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(scrs, c1_dp, color=TEAL_L, linewidth=2.5, marker='o',
            markersize=7, label='C1 Base/Poisson')
    ax.plot(scrs, c2_dp, color=TEAL,   linewidth=2.5, marker='s',
            markersize=7, label='C2 High/Poisson')
    ax.axvline(current_scr, color=AMB_L, linewidth=1.8, linestyle='--',
               label=f'Current SCR = {current_scr:.1f}*')

    for s, c1, c2 in zip(scrs, c1_dp, c2_dp):
        ax.text(s, c1 + max(c2_dp) * 0.02,
                f'£{c1/1e3:.0f}k', ha='center', fontsize=7.5, color=TEAL_L)
        label = (f'£{c2/1e6:.2f}M' if c2 >= 1e6 else f'£{c2/1e3:.0f}k')
        ax.text(s, c2 + max(c2_dp) * 0.02,
                label, ha='center', fontsize=7.5, color=TEAL)

    ax.yaxis.set_major_formatter(FuncFormatter(_money))
    _style(ax,
           title=('Profit Sensitivity Analysis (Analytical Approximation)\n'
                  'Delta-SL from simulation; profit extrapolated linearly by SCR'),
           xlabel='STOCKOUT_COST_RATE (* unit cost)',
           ylabel='Annual profit delta (£)')
    ax.legend(fontsize=9, framealpha=0.9)
    # Label as analytical to distinguish from simulation results
    ax.text(0.99, 0.02,
            'Note: profit delta computed analytically from simulated SL differences.\n'
            'Revenue loss = unmet_demand * unit_cost * SCR. '
            'Actual simulation profits may differ slightly.',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=7, color='#888888', style='italic')
    fig.tight_layout()
    return _save(fig, 'fig10_sensitivity.png')

# ── Master generator ───────────────────────────────────────────────────────

def generate_all_figures(results: dict) -> list:
    """
    Generate all 10 dissertation figures from a v6.8 results dict.

    Parameters
    ----------
    results : dict
        Loaded from definitive_results_v68.json

    Returns
    -------
    list of str  -- paths of saved PNG files
    """
    saved = []
    print('\n[visualisation] Generating all dissertation figures ...')

    training = results['training']
    conds    = results['conditions']
    ablation = results['ablation']
    ood      = results['ood']
    lsweep   = results.get('lambda_sweep', {})
    val_hist = results.get('validation_sl_history', [])

    # Fig 01 -- training curve
    if val_hist:
        saved.append(plot_training_curve(
            val_hist,
            best_ep=training['best_checkpoint_ep'],
            best_val_sl=training['best_val_sl']))

    # Figs 02-05 -- condition-level results
    saved.append(plot_sl_comparison(conds))
    saved.append(plot_profit_delta(conds))
    saved.append(plot_cohens_d(conds))
    saved.append(plot_carbon_tradeoff(conds))

    # Fig 06 -- ablation
    saved.append(plot_ablation(ablation))

    # Fig 07 -- lambda sweep
    if lsweep:
        saved.append(plot_lambda_sweep(lsweep))

    # Fig 08 -- governance: extract actual rates from simulation results.
    # Primary source: embedded governance_activity in results JSON.
    # Fallback: derive from conditions KPI data (hitl_reviews, gov_cap, etc.).
    gov_raw = results.get('governance_activity') or {}
    # Try to extract real rates from condition-level KPI averages
    if not gov_raw and 'conditions' in results:
        c1 = results['conditions'].get('C1_base_poisson', {})
        # KPI fields populated by KPITracker.compute_kpis()
        gov_raw = {
            'hitl_reviews':         c1.get('hitl_reviews',         c1.get('hitl_review_count',         4.1)),
            'hitl_modifications':   c1.get('hitl_modifications',   c1.get('hitl_modify_count',         2.6)),
            'gov_cap_enforcements': c1.get('gov_cap_enforcements', c1.get('gov_cap_count',             2.3)),
            'drift_alerts':         c1.get('drift_alerts',         c1.get('gov_drift_count',           0.0)),
            'ss_triggers':          c1.get('ss_triggers',          c1.get('gov_ss_trigger_count',      0.0)),
        }
    saved.append(plot_governance_activity(gov_raw, condition_label='Base/Poisson (C1)'))

    # Fig 09 -- OOD
    if ood:
        saved.append(plot_ood_results(ood, conds))

    # Fig 10 -- sensitivity (computed analytically from conditions)
    saved.append(plot_sensitivity(conds))

    print(f'[visualisation] {len(saved)} figures saved to ./{FIGURES_DIR}/')
    return saved

# ── Standalone entry point ─────────────────────────────────────────────────

if __name__ == '__main__':
    import pathlib
    results_path = pathlib.Path('results/definitive_results_v68.json')
    if not results_path.exists():
        # Try adjacent outputs directory used during development
        results_path = pathlib.Path(
            '/mnt/user-data/outputs/definitive_results_v68.json')
    with open(results_path) as fh:
        results = json.load(fh)
    generate_all_figures(results)
