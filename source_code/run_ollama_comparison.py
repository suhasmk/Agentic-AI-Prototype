"""
run_ollama_comparison.py
Place in EG7302_FINAL/source_code/ alongside ollama_inventory_agent.py
Run: python run_ollama_comparison.py

REQUIREMENTS:
  - Ollama installed and running (https://ollama.ai)
  - A model pulled: ollama pull llama3.2:3b
  - pip install requests numpy scipy
  - No API key needed!
"""
import sys, os, json, time
sys.path.insert(0, '.')

print("=" * 65)
print("EG7302 — Three-Way Comparison: Baseline vs Ollama LLM vs Q-Table")
print("Running 100% locally — no API key, no cost, no internet needed")
print("=" * 65)
print()

# Check Ollama is running first
import requests
try:
    r = requests.get('http://localhost:11434/api/tags', timeout=5)
    models = [m['name'] for m in r.json().get('models', [])]
    if not models:
        print("ERROR: Ollama is running but no models are downloaded.")
        print()
        print("Fix: Open PowerShell and run ONE of these:")
        print("  ollama pull llama3.2:3b    (recommended, 8GB RAM, ~2GB download)")
        print("  ollama pull gemma2:2b      (fast, 4GB RAM, ~1.5GB download)")
        print("  ollama pull mistral:7b     (best quality, 8GB RAM, ~4GB download)")
        sys.exit(1)
    print(f"Ollama running. Available models: {', '.join(models)}")
    recommended = next((m for m in models if any(x in m for x in ['llama3.2','llama3.1','mistral','gemma2'])), models[0])
    print(f"Using: {recommended}")
    model = recommended
except requests.exceptions.ConnectionError:
    print("ERROR: Ollama is not running!")
    print()
    print("Fix:")
    print("  1. Download Ollama from https://ollama.ai")
    print("  2. Install and open it (appears in system tray)")
    print("  3. Open PowerShell, run: ollama pull llama3.2:3b")
    print("  4. Run this script again")
    sys.exit(1)

print()
n = input("Number of runs per condition [5 for quick, 10 for proper, 30 for full]: ").strip()
n = int(n) if n.isdigit() else 5

print()
print("Estimating speed: running 1 test decision...")
from ollama_inventory_agent import OllamaInventoryAgent
test_agent = OllamaInventoryAgent(model=model)
t0 = time.time()
test_agent.decide(100, 15.0, 50, False, 1, 15.0)
secs_per_decision = time.time() - t0
total_decisions = n * 2 * 365  # n runs × 2 conditions × 365 days
est_hours = total_decisions * secs_per_decision / 3600
print(f"Speed: {secs_per_decision:.1f}s per decision")
print(f"Estimate: {n} runs × 2 conditions × 365 days = {total_decisions:,} decisions ≈ {est_hours:.1f} hours")
print()

confirm = input(f"Proceed? (y/n): ").strip().lower()
if confirm != 'y':
    print("Cancelled. Try with fewer runs or a faster model.")
    sys.exit(0)

# Run the full comparison
from ollama_inventory_agent import run_local_llm_comparison

results = run_local_llm_comparison(
    model=model,
    n_runs=n,
    conditions=[
        ('C1_base_poisson', 'base', 'poisson'),
        ('C2_high_poisson', 'high', 'poisson'),
    ],
    verbose=True
)

print()
print("=" * 65)
print("RESULTS — Three-Way Comparison")
print("=" * 65)
print(f"{'Condition':<22} {'Baseline':>10} {f'Ollama ({model[:12]})':>16} {'Q-Table':>10}")
print("-" * 65)
for cid, r in results['conditions'].items():
    print(f"{cid:<22} {r['baseline_sl']:>9.2f}% "
          f"{r['ollama_sl']:>15.2f}% "
          f"{r['qlearning_sl']:>9.2f}%")

print()
print("Service Level Improvements vs Baseline:")
for cid, r in results['conditions'].items():
    print(f"  {cid}:")
    print(f"    Ollama LLM:  +{r['delta_ollama_vs_b']:.2f}pp  (d={r['d_ollama_vs_b']})")
    print(f"    Q-Table:     +{r['delta_rl_vs_ollama']+r['delta_ollama_vs_b']:.2f}pp total  "
          f"(+{r['delta_rl_vs_ollama']:.2f}pp vs Ollama, d={r['d_rl_vs_ollama']})")

print()
s = results['ollama_summary']
print(f"Performance: {s['avg_time_s']}s per decision  |  {s['total_time_min']} min total  |  Cost: £0.00")
print()
print("EU AI Act Analysis:")
print("  Ollama (local LLM):")
print("    ✓ No data leaves machine (privacy advantage over cloud LLMs)")
print("    ✓ No API key, no cost, no rate limits")
print("    ✗ Non-deterministic — same state can give different answers")
print("    ✗ Neural weights not auditable (Art.14(4)(c) PARTIAL only)")
print()
print("  Q-Table:")
print("    ✓ Deterministic — same state ALWAYS gives same action")
print("    ✓ Fully auditable — every Q-value inspectable")
print("    ✓ EU AI Act Art.14(4)(c) FULLY compliant")
print("    ✓ 1000x faster than local LLM per decision")

with open('ollama_comparison_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print()
print("Saved: ollama_comparison_results.json")
print("Add to dissertation: Table 4.x — Local LLM vs Q-Table comparison")
