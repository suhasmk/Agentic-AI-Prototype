"""
llm_inventory_agent.py
═══════════════════════════════════════════════════════════════════════════════
EG7302 Extension: LLM-Based Inventory Agent using Claude API

Replaces the Q-table lookup with Claude claude-haiku-4-5-20251001 as the
decision-making core. The agent receives the full inventory state as structured
text and returns an ordering decision with a natural-language rationale.

Key differences from QLearningInventoryAgent:
  - No training phase required (zero-shot or few-shot)
  - Decisions are context-aware and explainable in natural language
  - Policy is NOT inspectable (opaque neural weights vs auditable Q-table)
  - Stochastic decisions even at temperature=0 (API non-determinism)
  - Significantly higher cost (~£5-8 per full experimental run at Haiku pricing)
  - Slower: ~0.5-1s per decision vs ~0.001ms for Q-table lookup

EU AI Act Article 14 Implications:
  - Fails inspectability requirement: cannot audit why Claude made a specific
    decision on day 47 of run 23
  - Satisfies Article 14(4)(d) override: governance cap still applied
  - Does NOT satisfy Article 14(4)(c) interpretation: rationale varies run-to-run
  - This limitation is the PRIMARY motivation for the Q-table approach

Usage:
    agent = LLMInventoryAgent(model='claude-haiku-4-5-20251001')
    order_qty, audit = agent.decide(
        inventory=120, demand_forecast=18.5, pending_qty=50,
        disruption_active=True, day=43, demand_rate=22.0
    )

API Cost Estimate (April 2026):
    claude-haiku-4-5-20251001: ~$0.25/M input tokens, ~$1.25/M output tokens
    Per decision: ~350 input + ~80 output tokens ≈ $0.000188/decision
    30 runs × 365 days × 4 conditions = 43,800 decisions ≈ $8.24 total
"""

import json
import os
import time
from typing import Optional, Tuple
import numpy as np

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("WARNING: anthropic not installed. Run: pip install anthropic")


# ── System prompt: defines the agent's role and constraints ──────────────────
SYSTEM_PROMPT = """You are an inventory replenishment agent for a single-SKU manufacturing
and logistics operation. Your goal is to maximise service level (units fulfilled / units
demanded) while managing holding costs, stockout costs, and carbon emissions.

OPERATIONAL CONTEXT:
- Product: Field & Stream Sportsman 16 Gun Fire Safe
- Unit cost: £141.23 | Unit price: ~£180 | Stockout cost: 2× unit cost per unmet unit
- Holding cost: 25% of unit cost per year (£0.097/unit/day)
- Normal lead time: ~5 days | Disrupted lead time: ~12.5 days (2.5× extension)
- Target service level: 95%+

DECISION RULES:
- Available order quantities: [0, 25, 50, 75, 100, 150, 200, 250] units
- Maximum order: 200 units (governance cap — never exceed this)
- During active disruptions: order proactively before stock depletes
- Under high demand (>18 u/day): maintain larger safety buffer
- Under low demand (<9 u/day): avoid over-ordering (holding cost dominates)

RESPOND ONLY with valid JSON, no preamble or explanation outside the JSON:
{"order_qty": <integer from the list above>, "rationale": "<one concise sentence>"}"""


class LLMInventoryAgent:
    """
    LLM-based inventory agent using Claude as the decision core.

    This agent is used for COMPARISON with the tabular Q-learning agent.
    It demonstrates that while LLMs can make reasonable inventory decisions
    in a zero-shot setting, they lack the inspectability required by
    EU AI Act Article 14 and ISO/IEC 42001 §8.4.
    """

    def __init__(
        self,
        model: str = 'claude-haiku-4-5-20251001',
        temperature: float = 0.0,
        max_retries: int = 3,
        fallback_qty: int = 75,
        seed: Optional[int] = None,
    ):
        self.model        = model
        self.temperature  = temperature
        self.max_retries  = max_retries
        self.fallback_qty = fallback_qty  # ROP heuristic fallback on API failure
        self.name         = f"LLM Agent ({model.split('-')[-3] if '-' in model else model})"

        # Statistics
        self.n_decisions   = 0
        self.n_api_calls   = 0
        self.n_fallbacks   = 0
        self.total_tokens  = 0
        self.decision_log  = []
        self.api_cost_usd  = 0.0

        # Token cost (Haiku rates, April 2026)
        self._cost_per_input_mtok  = 0.25   # USD per million input tokens
        self._cost_per_output_mtok = 1.25   # USD per million output tokens

        # RNG for tie-breaking (reproducible per seed)
        self._rng = np.random.default_rng(seed) if seed is not None else None

        if ANTHROPIC_AVAILABLE:
            self._client = anthropic.Anthropic()
        else:
            self._client = None
            print(f"[{self.name}] WARNING: anthropic not available. Will use fallback.")

        from config import ORDER_ACTIONS, GOVERNANCE_ORDER_CAP
        self._valid_actions = ORDER_ACTIONS
        self._cap           = GOVERNANCE_ORDER_CAP

    # ── Main decision method ──────────────────────────────────────────────────
    def decide(
        self,
        inventory: float,
        demand_forecast: float,
        pending_qty: float,
        disruption_active: bool,
        day: int,
        demand_rate: Optional[float] = None,
    ) -> Tuple[int, dict]:
        """
        Make an ordering decision for the current day.

        Returns: (order_qty: int, audit_record: dict)
        """
        self.n_decisions += 1

        prompt = self._build_prompt(
            inventory, demand_forecast, pending_qty,
            disruption_active, day, demand_rate
        )

        order_qty, raw_response, tokens_used = self._call_api(prompt)
        order_qty = self._snap_to_valid_action(order_qty)
        order_qty = min(order_qty, self._cap)  # Governance cap always enforced

        # Compute cost
        cost_usd = (tokens_used.get('input', 0) / 1_000_000 * self._cost_per_input_mtok +
                    tokens_used.get('output', 0) / 1_000_000 * self._cost_per_output_mtok)
        self.api_cost_usd += cost_usd
        self.total_tokens += tokens_used.get('input', 0) + tokens_used.get('output', 0)

        audit = {
            'day':              day,
            'order_qty':        order_qty,
            'inventory':        round(inventory, 1),
            'demand_forecast':  round(demand_forecast, 1),
            'pending_qty':      round(pending_qty, 1),
            'disruption':       disruption_active,
            'demand_rate':      round(demand_rate, 1) if demand_rate else None,
            'raw_response':     raw_response,
            'tokens':           tokens_used,
            'cost_usd':         round(cost_usd, 6),
            'agent':            self.name,
            'policy_source':    'llm' if self._client else 'fallback',
        }
        self.decision_log.append(audit)
        return order_qty, audit

    # ── Prompt construction ───────────────────────────────────────────────────
    def _build_prompt(
        self,
        inventory: float,
        demand_forecast: float,
        pending_qty: float,
        disruption_active: bool,
        day: int,
        demand_rate: Optional[float],
    ) -> str:
        eff_stock = inventory + pending_qty
        regime = (
            'HIGH (>18 u/day — high-demand regime, maintain larger buffer)'
            if (demand_rate or demand_forecast) > 18 else
            'LOW (<9 u/day — low-demand, avoid over-ordering)'
            if (demand_rate or demand_forecast) < 9 else
            'MID (9-18 u/day — standard regime)'
        )
        disruption_note = (
            'ACTIVE — lead times are 2.5× longer (~12.5 days). ORDER PROACTIVELY '
            'even if inventory seems adequate. Expected pipeline demand over disrupted '
            f'lead time: ~{demand_forecast * 12.5:.0f} units.'
            if disruption_active else
            'None — normal lead times (~5 days).'
        )
        return (
            f"Inventory decision required for Day {day}.\n\n"
            f"Current state:\n"
            f"  On-hand inventory:      {inventory:.0f} units\n"
            f"  Pending orders (in-transit): {pending_qty:.0f} units\n"
            f"  Effective stock position:    {eff_stock:.0f} units\n"
            f"  7-day demand forecast:   {demand_forecast:.1f} u/day\n"
            f"  Rolling 7-day demand rate: {f'{demand_rate:.1f} u/day' if demand_rate else 'N/A'}\n"
            f"  Demand regime:           {regime}\n"
            f"  Supply disruption:       {disruption_note}\n\n"
            f"Available order quantities: {self._valid_actions}\n"
            f"Maximum allowed: {self._cap} units (governance cap — NEVER exceed)\n\n"
            f"Decide the order quantity for today."
        )

    # ── API call with retry ───────────────────────────────────────────────────
    def _call_api(self, prompt: str) -> Tuple[int, str, dict]:
        """Call Claude API with retry logic. Returns (order_qty, raw_text, tokens)."""
        if self._client is None:
            self.n_fallbacks += 1
            return self.fallback_qty, 'API unavailable — using ROP fallback', {}

        last_error = None
        for attempt in range(self.max_retries):
            try:
                self.n_api_calls += 1
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=120,
                    temperature=self.temperature,
                    system=SYSTEM_PROMPT,
                    messages=[{'role': 'user', 'content': prompt}]
                )
                raw = response.content[0].text.strip()
                tokens = {
                    'input':  response.usage.input_tokens,
                    'output': response.usage.output_tokens,
                }

                # Parse JSON response
                parsed = json.loads(raw)
                order_qty = int(parsed.get('order_qty', self.fallback_qty))
                return order_qty, raw, tokens

            except json.JSONDecodeError as e:
                last_error = f"JSON parse error: {e} | raw: {raw[:100]}"
                # Try to extract a number from the response
                import re
                nums = re.findall(r'\b(0|25|50|75|100|150|200|250)\b', raw)
                if nums:
                    return int(nums[0]), raw, tokens if 'tokens' in dir() else {}
                time.sleep(0.5 * (attempt + 1))

            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    time.sleep(1.0 * (attempt + 1))

        # All retries failed — use ROP fallback
        self.n_fallbacks += 1
        print(f"[{self.name}] API failed after {self.max_retries} attempts: {last_error}")
        return self.fallback_qty, f'FALLBACK: {last_error}', {}

    # ── Snap to nearest valid action ─────────────────────────────────────────
    def _snap_to_valid_action(self, qty: int) -> int:
        """Snap proposed quantity to nearest valid action."""
        return min(self._valid_actions, key=lambda a: abs(a - qty))

    # ── Summary statistics ────────────────────────────────────────────────────
    def get_summary(self) -> dict:
        return {
            'model':        self.model,
            'n_decisions':  self.n_decisions,
            'n_api_calls':  self.n_api_calls,
            'n_fallbacks':  self.n_fallbacks,
            'total_tokens': self.total_tokens,
            'api_cost_usd': round(self.api_cost_usd, 4),
            'api_cost_gbp': round(self.api_cost_usd * 0.79, 4),
            'avg_tokens_per_call': (
                round(self.total_tokens / max(1, self.n_api_calls), 0)
            ),
        }

    def reset_log(self):
        """Clear decision log between simulation runs."""
        self.decision_log = []


# ── EU AI Act compliance comparison ──────────────────────────────────────────
EU_AI_ACT_COMPARISON = {
    'Art14_4a_understand_capabilities': {
        'QLearning': 'FULL — every state-action value is auditable; Q-table exported as JSON',
        'LLM':       'PARTIAL — capability bounded by system prompt; internal weights opaque',
    },
    'Art14_4b_detect_anomalies': {
        'QLearning': 'FULL — drift monitor tracks SL vs training-time baseline',
        'LLM':       'PARTIAL — no baseline to compare against; drift detection requires external monitor',
    },
    'Art14_4c_interpret_outputs': {
        'QLearning': 'FULL — Q-values, state encoding, and policy source all logged per decision',
        'LLM':       'PARTIAL — rationale field provides reasoning but varies run-to-run; non-auditable',
    },
    'Art14_4d_override_intervene': {
        'QLearning': 'FULL — governance cap + HITL gateway enforced pre-execution',
        'LLM':       'FULL — same governance cap + HITL gateway applied post-LLM decision',
    },
    'ISO42001_8_4_operational_records': {
        'QLearning': 'FULL — deterministic: same state always produces same action (verifiable)',
        'LLM':       'PARTIAL — non-deterministic even at temperature=0; records exist but not reproducible',
    },
    'overall_compliance_verdict': {
        'QLearning': 'COMPLIANT-BY-DESIGN — fully inspectable, deterministic, auditable',
        'LLM':       'REQUIRES ADDITIONAL SAFEGUARDS — opaque reasoning, non-reproducible decisions',
    }
}


if __name__ == '__main__':
    # Quick smoke test — single decision
    agent = LLMInventoryAgent(seed=42)
    print(f"Agent: {agent.name}")
    print("Making a test decision (Day 43, disruption active)...")
    qty, audit = agent.decide(
        inventory=217, demand_forecast=24.5, pending_qty=0,
        disruption_active=True, day=43, demand_rate=22.0
    )
    print(f"Decision: {qty} units")
    print(f"Rationale: {json.loads(audit['raw_response']).get('rationale', 'N/A') if '{' in audit['raw_response'] else audit['raw_response']}")
    print(f"Summary: {agent.get_summary()}")
    print()
    print("EU AI Act comparison:")
    for k, v in EU_AI_ACT_COMPARISON.items():
        print(f"  {k}:")
        print(f"    Q-table: {v['QLearning']}")
        print(f"    LLM:     {v['LLM']}")
