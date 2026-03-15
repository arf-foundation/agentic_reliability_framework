"""
Canonical governance loop – combines policy, cost, reliability, and risk signals
to produce a comprehensive HealingIntent for each infrastructure intent.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from .intents import InfrastructureIntent
from .healing_intent import HealingIntent, IntentSource, RecommendedAction, IntentStatus
from .policy_engine import PolicyEngine
from .cost_estimator import CostEstimator
from ..risk.engine import RiskEngine  # assuming exists

logger = logging.getLogger(__name__)

# Default weights for TotalRisk logistic formula
DEFAULT_WEIGHTS = {
    "risk": 0.5,           # infrastructure risk (Bayesian)
    "policy": 0.2,         # normalized policy severity
    "cost": 0.1,           # normalized cost pressure
    "epistemic": 0.1,      # epistemic risk (hallucination / evidence mismatch)
    "uncertainty": 0.05,   # uncertainty penalty
    "human": 0.05,         # human escalation pressure (complexity, etc.)
}
DEFAULT_INTERCEPT = -2.0   # b in logistic formula

class GovernanceLoop:
    """
    Orchestrates the full governance decision process for an infrastructure intent.
    """

    def __init__(
        self,
        policy_engine: PolicyEngine,
        cost_estimator: CostEstimator,
        risk_engine: RiskEngine,
        weights: Optional[Dict[str, float]] = None,
        intercept: float = DEFAULT_INTERCEPT,
        enable_epistemic: bool = False,
    ):
        self.policy_engine = policy_engine
        self.cost_estimator = cost_estimator
        self.risk_engine = risk_engine
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.intercept = intercept
        self.enable_epistemic = enable_epistemic

    def _normalize_policy_violations(self, violations: List[str]) -> float:
        """Convert policy violations to a severity score [0,1]."""
        if not violations:
            return 0.0
        # Simple count capped at MAX_POLICY_VIOLATIONS
        from ..config.constants import MAX_POLICY_VIOLATIONS
        return min(len(violations) / MAX_POLICY_VIOLATIONS, 1.0)

    def _compute_epistemic_risk(self, intent: InfrastructureIntent, context: Dict) -> Optional[float]:
        """
        If research probe is enabled, compute epistemic risk (hallucination, contradiction).
        Returns None if not available.
        """
        if not self.enable_epistemic:
            return None
        try:
            # Deferred import – only when needed and if available
            from ...research.eclipse_probe.hallucination_model import hallucination_risk
            # For now, return a placeholder; replace with actual call when research probe is integrated
            return 0.2  # placeholder
        except ImportError:
            logger.debug("Epistemic probe not available")
            return None

    def _compute_total_risk(self, components: Dict[str, float]) -> float:
        """
        Logistic combination: TotalRisk = sigmoid( sum(w_i * x_i) + b )
        """
        linear = self.intercept
        for key, value in components.items():
            linear += self.weights.get(key, 0.0) * value
        # Clamp to avoid extreme values
        linear = np.clip(linear, -10, 10)
        return 1.0 / (1.0 + np.exp(-linear))

    def _decide_action(
        self,
        total_risk: float,
        hard_constraints_passed: bool,
        epistemic_risk: Optional[float],
        ambiguity: float,
        confidence: float,
    ) -> Tuple[RecommendedAction, str]:
        """
        Determine final recommended action based on total risk and hard constraints.
        """
        if not hard_constraints_passed:
            return RecommendedAction.DENY, "Hard policy constraints violated"

        # High epistemic risk or ambiguity should escalate rather than act
        if epistemic_risk is not None and epistemic_risk > 0.7:
            return RecommendedAction.ESCALATE, "High epistemic uncertainty"
        if ambiguity > 0.8:
            return RecommendedAction.ESCALATE, "Situation too ambiguous"

        # Use DPT thresholds (from constants)
        from ..config.constants import DPT_LOW, DPT_HIGH
        if total_risk < DPT_LOW:
            return RecommendedAction.APPROVE, f"Total risk {total_risk:.2f} below threshold"
        elif total_risk > DPT_HIGH:
            return RecommendedAction.DENY, f"Total risk {total_risk:.2f} above threshold"
        else:
            return RecommendedAction.ESCALATE, f"Total risk {total_risk:.2f} in escalation zone"

    def run(self, intent: InfrastructureIntent, context: Optional[Dict] = None) -> HealingIntent:
        """
        Execute the full governance loop for a single intent.
        """
        context = context or {}

        # 1. Validate intent (if any basic validation)
        # (assume intent is valid by construction)

        # 2. Evaluate hard policy constraints (deterministic)
        # For now, we'll simulate policy violations list.
        policy_violations = context.get("policy_violations", [])  # placeholder
        hard_constraints_passed = len(policy_violations) == 0

        # 3. Compute cost pressure
        cost_estimate = None
        cost_pressure = 0.0
        if hasattr(intent, "resource_type") and hasattr(intent, "size"):
            # only provision intents have direct cost
            cost_estimate = self.cost_estimator.estimate_monthly_cost(intent)
            if cost_estimate is not None:
                # Normalize cost pressure (e.g., relative to some max)
                max_cost = 1000.0  # configurable
                cost_pressure = min(cost_estimate / max_cost, 1.0)

        # 4. Compute epistemic risk (if enabled)
        epistemic_risk = self._compute_epistemic_risk(intent, context)

        # 5. Compute Bayesian infrastructure risk
        # Assume RiskEngine has a method calculate_risk returning risk_score and components
        risk_result = self.risk_engine.calculate_risk(intent, cost_estimate, policy_violations)
        # risk_result might be a dict with risk_score, contributions, etc.
        risk_score = risk_result.get("risk_score", 0.5) if isinstance(risk_result, dict) else 0.5
        risk_contributions = risk_result.get("contributions", {}) if isinstance(risk_result, dict) else {}

        # 6. Assemble component scores
        components = {
            "risk": risk_score,
            "policy": self._normalize_policy_violations(policy_violations),
            "cost": cost_pressure,
            "epistemic": epistemic_risk if epistemic_risk is not None else 0.0,
            "uncertainty": 0.0,  # placeholder
            "human": 0.0,         # placeholder
        }

        total_risk = self._compute_total_risk(components)

        # 7. Decide action
        ambiguity = context.get("ambiguity", 0.0)
        confidence = 1.0 - total_risk  # simple inverse
        action, reason = self._decide_action(
            total_risk, hard_constraints_passed, epistemic_risk, ambiguity, confidence
        )

        # 8. Build HealingIntent
        # Use from_analysis factory (adds OSS advisory mode)
        healing_intent = HealingIntent.from_analysis(
            action=action.value,
            component=intent.service_name if hasattr(intent, "service_name") else "unknown",
            parameters={},  # could derive from intent
            justification=reason,
            confidence=confidence,
            incident_id=context.get("incident_id", ""),
            source=IntentSource.INFRASTRUCTURE_ANALYSIS,
            risk_score=risk_score,
            cost_projection=cost_estimate,
        ).mark_as_oss_advisory()

        # Add new fields (use object.__setattr__ because HealingIntent is frozen)
        object.__setattr__(healing_intent, "risk_contributions", risk_contributions)
        object.__setattr__(healing_intent, "policy_violations", policy_violations)
        object.__setattr__(healing_intent, "epistemic_uncertainty", epistemic_risk)
        object.__setattr__(healing_intent, "ambiguity_score", ambiguity)
        if action == RecommendedAction.ESCALATE:
            object.__setattr__(healing_intent, "decision_margin", total_risk - 0.5)
        else:
            object.__setattr__(healing_intent, "decision_margin", None)

        return healing_intent

    def run_batch(self, intents: List[InfrastructureIntent], context: Optional[Dict] = None) -> List[HealingIntent]:
        """Process multiple intents."""
        return [self.run(intent, context) for intent in intents]
