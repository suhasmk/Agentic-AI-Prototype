"""
Governance Agent

Manages audit logs, decision explanations, and reproducibility checks.
"""

import json
import hashlib
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
from config import GOVERNANCE_EXPORT_LIMIT

class GovernanceAgent:
    """
    Governance and auditability layer for the agentic AI prototype.
    
    
    - Full decision traceability
    - Reproducibility verification (hash-based)
    - Human-readable decision explanations
    - Audit trail export
    """

    def __init__(self, system_name: str = "Agentic AI System"):
        self.system_name = system_name
        self.audit_trail: List[Dict] = []
        self.run_hashes: List[str] = []
        self.run_seeds: List[int] = []
        self.decision_count = 0
        self.anomaly_log: List[Dict] = []

    def log_decision(self, day: int, agent_name: str, 
                     decision_type: str, inputs: Dict,
                     action: Any, rationale: str,
                     q_values: List[float] = None):
        """
        Record a decision with full traceability.
        - Inputs (state)
        - Action taken
        - Rationale / rule applied
        - Q-values (for RL agent)
        - Timestamp equivalent (day)
        """
        record = {
            'decision_id': self.decision_count,
            'day': day,
            'agent': agent_name,
            'type': decision_type,
            'inputs': inputs,
            'action': action,
            'rationale': rationale,
        }
        if q_values:
            record['q_values'] = q_values
            record['confidence'] = float(np.max(q_values) - np.mean(q_values))

        self.audit_trail.append(record)
        self.decision_count += 1
        return record

    def explain_decision(self, decision_id: int) -> str:
        """Generate human-readable explanation for a decision."""
        if decision_id >= len(self.audit_trail):
            return "Decision not found."

        d = self.audit_trail[decision_id]
        lines = [
            f"── Decision #{d['decision_id']} ────────────────────",
            f"  Day:         {d['day']}",
            f"  Agent:       {d['agent']}",
            f"  Type:        {d['type']}",
            f"  Action:      {d['action']}",
            f"  Rationale:   {d['rationale']}",
            f"  Inputs:"
        ]
        for k, v in d.get('inputs', {}).items():
            if isinstance(v, float):
                lines.append(f"    {k}: {v:.2f}")
            else:
                lines.append(f"    {k}: {v}")
        if 'q_values' in d:
            lines.append(f"  Q-values:    {[round(q, 3) for q in d['q_values']]}")
            lines.append(f"  Confidence:  {d.get('confidence', 0):.3f}")
        return "\n".join(lines)

    def compute_run_hash(self, run_results: Dict, seed: int) -> str:
        """
        Compute deterministic hash of run results for reproducibility.
        Same seed + same code = same hash.
        """
        # Round floats for stable hashing
        def round_dict(d, n=8):
            return {k: round(float(v), n) if isinstance(v, (float, np.floating))
                    else v for k, v in d.items()}

        payload = json.dumps({
            'seed': seed,
            'results': round_dict(run_results)
        }, sort_keys=True)
        h = hashlib.sha256(payload.encode()).hexdigest()[:16]
        self.run_hashes.append(h)
        self.run_seeds.append(seed)
        return h

    def verify_reproducibility(self) -> Dict:
        """
        Check that repeated runs with same seeds produce identical hashes.
        
        """
        if len(self.run_hashes) < 2:
            return {'reproducible': True, 'message': 'Insufficient runs'}

        # Check for duplicate seeds with same hashes
        seed_to_hashes: Dict[int, List[str]] = {}
        for seed, h in zip(self.run_seeds, self.run_hashes):
            seed_to_hashes.setdefault(seed, []).append(h)

        inconsistencies = []
        for seed, hashes in seed_to_hashes.items():
            if len(set(hashes)) > 1:
                inconsistencies.append(f"Seed {seed}: hashes differ {hashes}")

        return {
            'reproducible': len(inconsistencies) == 0,
            'total_runs': len(self.run_hashes),
            'unique_seeds': len(seed_to_hashes),
            'inconsistencies': inconsistencies,
            'message': 'All runs reproducible' if not inconsistencies
                       else f'{len(inconsistencies)} reproducibility failures'
        }

    def detect_anomalies(self, inventory: float, demand: float,
                         day: int, threshold_stockout_risk: float = 0.8):
        """Flag potential anomalies for human review."""
        if inventory <= 0 and demand > 0:
            self.anomaly_log.append({
                'day': day,
                'type': 'STOCKOUT',
                'severity': 'HIGH',
                'inventory': inventory,
                'demand': demand,
                'message': f'Stockout on day {day}: demand={demand:.0f}, inventory=0'
            })

    def generate_audit_summary(self) -> Dict:
        """High-level audit summary for reporting."""
        agents = {}
        for d in self.audit_trail:
            agents.setdefault(d['agent'], 0)
            agents[d['agent']] += 1

        return {
            'system': self.system_name,
            'total_decisions': self.decision_count,
            'decisions_by_agent': agents,
            'total_anomalies': len(self.anomaly_log),
            'stockouts_detected': sum(1 for a in self.anomaly_log
                                      if a['type'] == 'STOCKOUT'),
            'reproducibility': self.verify_reproducibility()
        }

    def export_audit_trail(self, max_records: int = GOVERNANCE_EXPORT_LIMIT) -> List[Dict]:
        """Export audit trail (first N records for report)."""
        return self.audit_trail[:max_records]
