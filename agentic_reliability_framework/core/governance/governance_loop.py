# core/governance/governance_loop.py
"""
Canonical Governance Loop – orchestrates policy, cost, risk, epistemic, and memory analysis.
Integrates the ECLIPSE hallucination probe for epistemic uncertainty quantification.
"""

import logging
import time
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

from agentic_reliability_framework.core.governance.intents import InfrastructureIntent
from agentic_reliability_framework.core.governance.policies import PolicyEvaluator
from agentic_reliability_framework.core.governance.cost_estimator import CostEstimator
from agentic_reliability_framework.core.governance.risk_engine import RiskEngine
from agentic_reliability_framework.runtime.memory import RAGGraphMemory
from agentic_reliability_framework.core.governance.healing_intent import (
    HealingIntent,
    RecommendedAction,
    IntentStatus,
    IntentSource,
)
from agentic_reliability_framework.core.research.eclipse_probe import HallucinationRisk
from agentic_reliability_framework.runtime.analytics.predictive import SimplePredictiveEngine, BusinessImpactCalculator
from agentic_reliability_framework.core.models.event import ReliabilityEvent
from agentic_reliability_framework.core.config.constants import (
    EPISTEMIC_ESCALATION_THRESHOLD, DPT_LOW, DPT_HIGH
)

logger = logging.getLogger(__name__)

class GovernanceLoop:
    """
    Orchestrates the full governance evaluation, integrating policy, cost, risk,
    epistemic uncertainty (including hallucination detection), predictive foresight,
    and semantic memory.
    """

    def __init__(
        self,
        policy_evaluator: PolicyEvaluator,
        cost_estimator: CostEstimator,
        risk_engine: RiskEngine,
        memory: Optional[RAGGraphMemory] = None,
        enable_epistemic: bool = False,
        hallucination_probe: Optional[HallucinationRisk] = None,
        dpt_low: float = 0.2,
        dpt_high: float = 0.8,
        # NEW: Predictive components
        predictive_engine: Optional[SimplePredictiveEngine] = None,
        business_calculator: Optional[BusinessImpactCalculator] = None,
    ):
        """
        Args:
            policy_evaluator: Evaluator for infrastructure policies.
            cost_estimator: Cost estimator for intents.
            risk_engine: Bayesian risk engine.
            memory: Optional semantic memory for retrieving similar incidents.
            enable_epistemic: Whether to compute epistemic uncertainty.
            hallucination_probe: Optional hallucination risk probe.
            dpt_low: Lower DPT threshold (default 0.2).
            dpt_high: Upper DPT threshold (default 0.8).
            predictive_engine: Optional predictive engine for forecasting.
            business_calculator: Optional business impact calculator.
        """
        self.policy_evaluator = policy_evaluator
        self.cost_estimator = cost_estimator
        self.risk_engine = risk_engine
        self.memory = memory
        self.enable_epistemic = enable_epistemic
        self.hallucination_probe = hallucination_probe
        self.dpt_low = dpt_low
        self.dpt_high = dpt_high
        self.predictive_engine = predictive_engine
        self.business_calculator = business_calculator or BusinessImpactCalculator()

    def run(
        self,
        intent: InfrastructureIntent,
        context: Optional[Dict[str, Any]] = None,
    ) -> HealingIntent:
        """
        Run the governance loop for a single intent.

        Args:
            intent: The infrastructure intent to evaluate.
            context: Additional context (e.g., incident metadata, query/evidence for hallucination).

        Returns:
            HealingIntent containing the full assessment.
        """
        context = context or {}
        logger.debug(f"Running governance loop for intent {intent.intent_id if hasattr(intent, 'intent_id') else 'unknown'}")

        # 1. Cost estimation (may be None for non‑provision intents)
        cost_projection = None
        try:
            cost_projection = self.cost_estimator.estimate_monthly_cost(intent)
        except Exception as e:
            logger.warning(f"Cost estimation failed: {e}")

        # 2. Policy evaluation (pass cost in context for cost‑based policies)
        policy_context = {"cost_estimate": cost_projection}
        policy_violations = self.policy_evaluator.evaluate(intent, policy_context)

        # 3. Risk calculation
        risk_score, explanation, contributions = self.risk_engine.calculate_risk(
            intent=intent,
            cost_estimate=cost_projection,
            policy_violations=policy_violations,
        )

        # -------------------------------------------------------------------
        # NEW: Predictive foresight (Branch 1)
        # -------------------------------------------------------------------
        predictive_risk = 0.0
        forecasts = []
        service = getattr(intent, "service_name", None) or context.get("service_name")
        if self.predictive_engine and service:
            forecasts = self.predictive_engine.forecast_service_health(service)
            if forecasts:
                # Map risk levels to numeric values
                risk_map = {"low": 0.1, "medium": 0.4, "high": 0.7, "critical": 0.95}
                weighted_risk = sum(risk_map[f.risk_level] * f.confidence for f in forecasts)
                predictive_risk = 1.0 / (1.0 + np.exp(-weighted_risk))  # sigmoid

        # -------------------------------------------------------------------
        # NEW: Business impact (Branch 3)
        # -------------------------------------------------------------------
        event = None
        if service:
            event = ReliabilityEvent(
                component=service,
                latency_p99=context.get("latency_p99"),
                error_rate=context.get("error_rate"),
                throughput=context.get("throughput", 0),
            )
        business_impact = self.business_calculator.calculate_impact(event) if event else {"severity_level": "LOW", "revenue_loss_estimate": 0, "affected_users_estimate": 0}
        impact_score = 0.0
        if business_impact["severity_level"] == "CRITICAL":
            impact_score = 0.9
        elif business_impact["severity_level"] == "HIGH":
            impact_score = 0.6
        elif business_impact["severity_level"] == "MEDIUM":
            impact_score = 0.3
        else:
            impact_score = 0.0

        # -------------------------------------------------------------------
        # NEW: Epistemic uncertainty (Branch 2)
        # -------------------------------------------------------------------
        epistemic_uncertainty = None
        hallucination_risk_score = None
        if self.enable_epistemic:
            epistemic_uncertainty = self._compute_epistemic_uncertainty(intent, context, risk_score, forecasts, service)
            # The hallucination probe can be part of epistemic uncertainty
            if self.hallucination_probe and "query" in context and "evidence" in context:
                entropy = context.get("entropy")
                evidence_lift = context.get("evidence_lift")
                contradiction = context.get("contradiction")
                if entropy is not None and evidence_lift is not None and contradiction is not None:
                    result = self.hallucination_probe.compute_risk(entropy, evidence_lift, contradiction)
                    hallucination_risk_score = result["risk_score"]
                    # Combine with other epistemic sources
                    epistemic_uncertainty = max(epistemic_uncertainty or 0.0, hallucination_risk_score)

        # 5. Semantic memory retrieval (similar incidents)
        similar_incidents = self._retrieve_similar_incidents(intent, context)

        # -------------------------------------------------------------------
        # NEW: Multi‑factor decision (Branch 3)
        # -------------------------------------------------------------------
        recommended_action = self._determine_action(
            policy_violations=policy_violations,
            risk_score=risk_score,
            predictive_risk=predictive_risk,
            epistemic_uncertainty=epistemic_uncertainty or 0.0,
            impact_score=impact_score
        )

        # -------------------------------------------------------------------
        # NEW: Extended HealingIntent (Branch 4)
        # -------------------------------------------------------------------
        # Build reasoning chain
        reasoning_chain = [
            f"Policy evaluation: {policy_violations if policy_violations else 'no violations'}",
            f"Risk score: {risk_score:.2f}",
            f"Predictive risk: {predictive_risk:.2f}",
            f"Epistemic uncertainty: {epistemic_uncertainty:.2f}",
            f"Business impact: {business_impact['severity_level']}",
            f"Final decision: {recommended_action.value}"
        ]

        # Epistemic breakdown
        epistemic_breakdown = {
            "hallucination": hallucination_risk_score or 0.0,
            "forecast_uncertainty": self._forecast_uncertainty(forecasts) if forecasts else 0.0,
            "data_sparsity": self._data_sparsity(service) if service else 0.5,
            "policy_ambiguity": 0.0  # can be computed from policy evaluator if needed
        }

        # Decision factors
        decision_factors = {
            "risk": risk_score,
            "predictive": predictive_risk,
            "uncertainty": epistemic_uncertainty,
            "impact": impact_score
        }

        # 7. Build HealingIntent using the factory method
        healing_intent = HealingIntent.from_analysis(
            action=recommended_action.value,
            component=getattr(intent, 'service_name', getattr(intent, 'component', 'unknown')),
            parameters={},
            justification=explanation,
            confidence=1.0 - (epistemic_uncertainty or 0.05),
            similar_incidents=similar_incidents,
            reasoning_chain=reasoning_chain,
            incident_id=context.get("incident_id", ""),
            source=IntentSource.INFRASTRUCTURE_ANALYSIS,
            rag_similarity_score=None,
            risk_score=risk_score,
            cost_projection=cost_projection,
            # NEW: Extended fields stored in metadata
            metadata={
                "predictive_risk": predictive_risk,
                "epistemic_breakdown": epistemic_breakdown,
                "decision_factors": decision_factors,
                "forecasts": [f.dict() for f in forecasts],
                "business_impact": business_impact
            }
        )
        # Mark as OSS advisory (no execution)
        healing_intent = healing_intent.mark_as_oss_advisory()
        return healing_intent

    # -------------------------------------------------------------------
    # NEW: Enhanced epistemic uncertainty (Branch 2)
    # -------------------------------------------------------------------
    def _compute_epistemic_uncertainty(
        self,
        intent: InfrastructureIntent,
        context: Dict[str, Any],
        risk_score: float,
        forecasts: List,
        service: Optional[str],
    ) -> float:
        """Combine multiple sources of epistemic uncertainty."""
        uncertainty = 0.05  # baseline

        # 1. Hallucination risk
        if self.hallucination_probe and "query" in context and "evidence" in context:
            entropy = context.get("entropy")
            evidence_lift = context.get("evidence_lift")
            contradiction = context.get("contradiction")
            if entropy is not None and evidence_lift is not None and contradiction is not None:
                result = self.hallucination_probe.compute_risk(entropy, evidence_lift, contradiction)
                uncertainty = max(uncertainty, result["risk_score"])

        # 2. Forecast uncertainty
        forecast_uncertainty = self._forecast_uncertainty(forecasts)
        uncertainty = max(uncertainty, forecast_uncertainty)

        # 3. Data sparsity
        sparsity = self._data_sparsity(service)
        uncertainty = max(uncertainty, sparsity)

        # Cap to [0,1]
        return max(0.0, min(1.0, uncertainty))

    def _forecast_uncertainty(self, forecasts: List) -> float:
        """Average 1 - confidence across forecasts."""
        if not forecasts:
            return 0.0
        return 1.0 - np.mean([f.confidence for f in forecasts])

    def _data_sparsity(self, service: Optional[str]) -> float:
        """Sparsity based on amount of history available."""
        if not self.predictive_engine or not service:
            return 0.5  # default moderate
        history = self.predictive_engine.service_history.get(service, [])
        if not history:
            return 1.0
        # Use a simple logistic function: sparsity = 1 / (1 + len(history))
        return 1.0 / (1.0 + len(history))

    def _retrieve_similar_incidents(
        self,
        intent: InfrastructureIntent,
        context: Dict[str, Any],
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar incidents from semantic memory.
        Requires the memory to have a method `find_similar` that returns IncidentNode objects.
        Converts them to dicts expected by HealingIntent.
        """
        if not self.memory or not self.memory.has_historical_data():
            return []

        try:
            class DummyEvent:
                component = getattr(intent, 'service_name', 'unknown')
            event = DummyEvent()
            analysis = {}
            nodes = self.memory.find_similar(event, analysis, k=k)
            similar = []
            for node in nodes:
                sim_dict = {
                    "incident_id": node.incident_id,
                    "component": node.component,
                    "severity": node.severity,
                    "timestamp": node.timestamp,
                    "metrics": node.metrics,
                    "similarity_score": node.metadata.get("similarity_score", 0.0),
                }
                similar.append(sim_dict)
            return similar
        except Exception as e:
            logger.warning(f"Failed to retrieve similar incidents: {e}")
            return []

    # -------------------------------------------------------------------
    # NEW: Multi‑factor decision (Branch 3)
    # -------------------------------------------------------------------
    def _determine_action(
        self,
        policy_violations: List[str],
        risk_score: float,
        predictive_risk: float,
        epistemic_uncertainty: float,
        impact_score: float,
    ) -> RecommendedAction:
        """
        Determine recommended action based on all factors.
        """
        # Hard constraints
        if policy_violations:
            return RecommendedAction.DENY

        # Epistemic escalation
        if epistemic_uncertainty > EPISTEMIC_ESCALATION_THRESHOLD:
            return RecommendedAction.ESCALATE

        # Predictive risk escalation
        if predictive_risk > 0.85:
            return RecommendedAction.ESCALATE

        # Combined score with weights (alpha, beta, gamma, delta)
        # These weights should be configurable; using sensible defaults
        alpha, beta, gamma, delta = 0.4, 0.3, 0.2, 0.1
        combined = (alpha * risk_score +
                    beta * predictive_risk +
                    gamma * epistemic_uncertainty +
                    delta * impact_score)

        if combined < self.dpt_low:
            return RecommendedAction.APPROVE
        if combined > self.dpt_high:
            return RecommendedAction.DENY
        return RecommendedAction.ESCALATE

    def run_batch(
        self,
        intents: List[InfrastructureIntent],
        contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> List[HealingIntent]:
        """Run multiple intents in batch."""
        if contexts is None:
            contexts = [{}] * len(intents)
        return [self.run(intent, ctx) for intent, ctx in zip(intents, contexts)]
