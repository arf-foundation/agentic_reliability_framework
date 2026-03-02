"""
Azure Infrastructure Simulator – Main orchestration engine.

This module ties together intents, policies, cost estimation, and risk scoring
to produce a HealingIntent. It is the primary entry point for the OSS advisory layer.
"""

from typing import List, Optional, Dict, Any

from agentic_reliability_framework.core.governance.intents import (
    InfrastructureIntent,
    ProvisionResourceIntent,
)
from agentic_reliability_framework.core.governance.policies import (
    Policy,
    PolicyEvaluator,
    CostThresholdPolicy,
)
from agentic_reliability_framework.core.governance.cost_estimator import CostEstimator
from agentic_reliability_framework.core.governance.risk_engine import RiskEngine
from agentic_reliability_framework.core.governance.healing_intent import (
    HealingIntent,
    RecommendedAction,
    IntentSource,
    create_infrastructure_healing_intent,  # <-- Use factory
)
from agentic_reliability_framework.core.config.constants import MAX_POLICY_VIOLATIONS


class AzureInfrastructureSimulator:
    """
    Orchestrates the evaluation of an infrastructure intent.

    The simulator uses:
        - A policy evaluator (with a policy tree)
        - A cost estimator
        - A risk engine (Bayesian version)

    It returns a HealingIntent with a recommendation, already marked as OSS advisory.
    """

    def __init__(
        self,
        policy: Policy,
        pricing_file: Optional[str] = None,
        risk_factors: Optional[List] = None,  # kept for compatibility, but not used
    ):
        """
        Initialize the simulator.

        Args:
            policy: The root policy (a Policy object, possibly composite).
            pricing_file: Optional path to custom pricing YAML.
            risk_factors: Ignored (kept for backward compatibility).
        """
        self._policy_evaluator = PolicyEvaluator(policy)
        self._cost_estimator = CostEstimator(pricing_file)
        self._risk_engine = RiskEngine()

    def evaluate(self, intent: InfrastructureIntent) -> HealingIntent:
        """
        Evaluate the intent and produce a HealingIntent.

        This method is pure and deterministic (same inputs → same output).
        The returned HealingIntent is already marked as OSS advisory.
        """
        # 1. Estimate cost (if applicable)
        cost = None
        if isinstance(intent, ProvisionResourceIntent):
            cost = self._cost_estimator.estimate_monthly_cost(intent)

        # 2. Evaluate policies with context (cost)
        context = {"cost_estimate": cost} if cost is not None else {}
        violations = self._policy_evaluator.evaluate(intent, context)

        # Enforce OSS limit on policy violations
        if len(violations) > MAX_POLICY_VIOLATIONS:
            violations = violations[:MAX_POLICY_VIOLATIONS]

        # 3. Compute risk (Bayesian)
        risk_score, explanation, contributions = self._risk_engine.calculate_risk(
            intent, cost, violations
        )

        # 4. Determine recommended action
        if risk_score > 0.8 or violations:
            recommended_action = RecommendedAction.DENY
        elif risk_score > 0.4:
            recommended_action = RecommendedAction.ESCALATE
        else:
            recommended_action = RecommendedAction.APPROVE

        # 5. Build justification
        justification_parts = [f"Risk score: {risk_score:.2f}."]
        if cost is not None:
            justification_parts.append(f"Estimated monthly cost: ${cost:.2f}.")
        if violations:
            justification_parts.append(f"Policy violations: {'; '.join(violations)}.")
        justification_parts.append(explanation)
        justification = " ".join(justification_parts)

        # 6. Package evaluation details
        details = {
            "cost_estimate": cost,
            "violations": violations,
            "risk_score": risk_score,
            "factor_contributions": contributions,
        }

<<<<<<< HEAD
        # 8. Create the HealingIntent via the factory helper and then mark as OSS advisory
        healing_intent = HealingIntent.from_infrastructure_intent(
            infrastructure_intent=intent,
            action=intent.intent_type,
            component=getattr(intent, "component", "unknown"),
            parameters={},
            justification=justification,
            confidence=0.9,  # could be derived from factor uncertainties
            risk_score=risk_score,
            risk_factors=contributions,
            cost_projection=cost,
            policy_violations=violations,
            recommended_action=recommended_action,
            source=IntentSource.INFRASTRUCTURE_ANALYSIS,
        )
=======
        # 7. Create a mock result object for the factory function
        class MockInfrastructureResult:
            def __init__(self):
                self.recommended_action = recommended_action
                self.intent_id = intent.intent_id
                self.risk_score = risk_score
                self.cost_projection = cost
                self.policy_violations = violations
                self.justification = justification
                self.confidence_score = 0.9
                self.evaluation_details = details
                self.infrastructure_intent = intent
>>>>>>> adf837024fd6d06c8d3dd61a120b662cc49a2c77

        result = MockInfrastructureResult()

        # 8. Create the HealingIntent via factory (handles OSS advisory marking)
        healing_intent = create_infrastructure_healing_intent(
            infrastructure_result=result,
            action_mapping=None
        )
        return healing_intent
