# core/governance/governance_loop.py
"""
Canonical Governance Loop – orchestrates policy, cost, risk, epistemic, and memory analysis.
Integrates the ECLIPSE hallucination probe for epistemic uncertainty quantification.
Uses Bayesian expected loss minimization for final action.
"""

import logging
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
    EPISTEMIC_ESCALATION_THRESHOLD,
    COST_FP, COST_IMPACT, COST_FN, COST_OPP, COST_REVIEW, COST_UNCERTAINTY,
    COST_PREDICTIVE, COST_VARIANCE, USE_EPISTEMIC_GATE,
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
        predictive_engine: Optional[SimplePredictiveEngine] = None,
        business_calculator: Optional[BusinessImpactCalculator] = None,
    ):
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
        context = context or {}
        logger.debug(f"Running governance loop for intent {intent.intent_id if hasattr(intent, 'intent_id') else 'unknown'}")

        # 1. Cost estimation
        cost_projection = None
        try:
            cost_projection = self.cost_estimator.estimate_monthly_cost(intent)
        except Exception as e:
            logger.warning(f"Cost estimation failed: {e}")

        # 2. Policy evaluation
        policy_context = {"cost_estimate": cost_projection}
        policy_violations = self.policy_evaluator.evaluate(intent, policy_context)

        # 3. Risk calculation (posterior mean and variance)
        risk_score, explanation, contributions = self.risk_engine.calculate_risk(
            intent=intent,
            cost_estimate=cost_projection,
            policy_violations=policy_violations,
        )

        # Extract Beta parameters from contributions (if available)
        alpha = contributions.get("conjugate_alpha", 1.0)
        beta = contributions.get("conjugate_beta", 10.0)
        total = alpha + beta
        if total > 0:
            variance = (alpha * beta) / (total * total * (total + 1))
        else:
            variance = 0.0

        # -------------------------------------------------------------------
        # Predictive foresight
        # -------------------------------------------------------------------
        predictive_risk = 0.0
        forecasts = []
        service = getattr(intent, "service_name", None) or context.get("service_name")
        if self.predictive_engine and service:
            forecasts = self.predictive_engine.forecast_service_health(service)
            if forecasts:
                confidences = np.array([f.confidence for f in forecasts])
                risk_numeric = np.array([{"low":0.1, "medium":0.4, "high":0.7, "critical":0.95}[f.risk_level] for f in forecasts])
                exp_c = np.exp(confidences)
                w = exp_c / np.sum(exp_c)
                weighted_risk = np.sum(w * risk_numeric)
                avg_confidence = np.mean(confidences)
                predictive_risk = weighted_risk * avg_confidence

        # -------------------------------------------------------------------
        # Business impact
        # -------------------------------------------------------------------
        event = None
        if service:
            event = ReliabilityEvent(
                component=service,
                latency_p99=context.get("latency_p99"),
                error_rate=context.get("error_rate"),
                throughput=context.get("throughput", 0),
            )
        business_impact = self.business_calculator.calculate_impact(event) if event else {"revenue_loss_estimate": 0, "affected_users_estimate": 0, "severity_level": "LOW", "throughput_reduction_pct": 0}
        b_mean = business_impact["revenue_loss_estimate"]

        # -------------------------------------------------------------------
        # Epistemic uncertainty (product of complements)
        # -------------------------------------------------------------------
        psi_mean = 0.0
        hallucination_risk = 0.0
        forecast_uncertainty = 0.0
        sparsity = 1.0
        if self.enable_epistemic:
            # Compute components
            if self.hallucination_probe and "query" in context and "evidence" in context:
                entropy = context.get("entropy")
                evidence_lift = context.get("evidence_lift")
                contradiction = context.get("contradiction")
                if entropy is not None and evidence_lift is not None and contradiction is not None:
                    result = self.hallucination_probe.compute_risk(entropy, evidence_lift, contradiction)
                    hallucination_risk = result["risk_score"]

            forecast_uncertainty = 1.0 - np.mean([f.confidence for f in forecasts]) if forecasts else 0.0

            if self.predictive_engine and service:
                history = self.predictive_engine.service_history.get(service, [])
                sparsity = np.exp(-0.05 * len(history))
            else:
                sparsity = 1.0

            uncertainty_components = [hallucination_risk, forecast_uncertainty, sparsity]
            psi_mean = 1.0 - np.prod([1.0 - min(1.0, max(0.0, u)) for u in uncertainty_components])

        # -------------------------------------------------------------------
        # Expected losses (computed unconditionally)
        # -------------------------------------------------------------------
        v_mean = context.get("estimated_value", 0.0)  # optional

        L_approve = (COST_FP * risk_score +
                     COST_IMPACT * b_mean +
                     COST_PREDICTIVE * predictive_risk +
                     COST_VARIANCE * variance)

        L_deny = COST_FN * (1 - risk_score) + COST_OPP * v_mean

        L_escalate = COST_REVIEW + COST_UNCERTAINTY * psi_mean

        expected_losses = {
            RecommendedAction.APPROVE: L_approve,
            RecommendedAction.DENY: L_deny,
            RecommendedAction.ESCALATE: L_escalate,
        }

        # -------------------------------------------------------------------
        # Decision
        # -------------------------------------------------------------------
        if policy_violations:
            recommended_action = RecommendedAction.DENY
        else:
            if USE_EPISTEMIC_GATE and psi_mean > EPISTEMIC_ESCALATION_THRESHOLD:
                recommended_action = RecommendedAction.ESCALATE
            else:
                recommended_action = min(expected_losses, key=expected_losses.get)

        # -------------------------------------------------------------------
        # Semantic memory retrieval
        # -------------------------------------------------------------------
        similar_incidents = self._retrieve_similar_incidents(intent, context)

        # -------------------------------------------------------------------
        # Build reasoning chain and decision trace
        # -------------------------------------------------------------------
        reasoning_chain = [
            f"Policy evaluation: {policy_violations if policy_violations else 'no violations'}",
            f"Risk score (E[θ]): {risk_score:.3f}, Variance: {variance:.4f}",
            f"Predictive risk: {predictive_risk:.3f}",
            f"Epistemic uncertainty: {psi_mean:.3f}",
            f"Business impact (revenue loss): ${b_mean:.2f}",
            f"Expected losses: Approve={L_approve:.2f}, Deny={L_deny:.2f}, Escalate={L_escalate:.2f}",
            f"Decision: {recommended_action.value}"
        ]

        epistemic_breakdown = {
            "hallucination": hallucination_risk,
            "forecast_uncertainty": forecast_uncertainty,
            "data_sparsity": sparsity,
            "policy_ambiguity": 0.0,
        }

        decision_factors = {
            "risk": risk_score,
            "predictive": predictive_risk,
            "uncertainty": psi_mean,
            "impact": b_mean,
            "variance": variance,
        }

        decision_trace = {
            "expected_losses": {a.value: expected_losses[a] for a in expected_losses},
            "selected_action": recommended_action.value,
            "posterior_mean": risk_score,
            "posterior_variance": variance,
        }

        # -------------------------------------------------------------------
        # Build HealingIntent
        # -------------------------------------------------------------------
        # Convert forecasts to JSON‑serializable dicts using model_dump()
        forecasts_serializable = [f.model_dump() for f in forecasts] if forecasts else []

        healing_intent = HealingIntent.from_analysis(
            action=recommended_action.value,
            component=getattr(intent, 'service_name', getattr(intent, 'component', 'unknown')),
            parameters={},
            justification=explanation,
            confidence=1.0 - psi_mean,
            similar_incidents=similar_incidents,
            reasoning_chain=reasoning_chain,
            incident_id=context.get("incident_id", ""),
            source=IntentSource.INFRASTRUCTURE_ANALYSIS,
            rag_similarity_score=None,
            risk_score=risk_score,
            cost_projection=cost_projection,
            policy_violations=policy_violations,
            metadata={
                "predictive_risk": predictive_risk,
                "epistemic_breakdown": epistemic_breakdown,
                "decision_factors": decision_factors,
                "decision_trace": decision_trace,
                "forecasts": forecasts_serializable,
                "business_impact": business_impact
            }
        )
        healing_intent = healing_intent.mark_as_oss_advisory()
        return healing_intent

    def _retrieve_similar_incidents(
        self,
        intent: InfrastructureIntent,
        context: Dict[str, Any],
        k: int = 5,
    ) -> List[Dict[str, Any]]:
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

    def run_batch(
        self,
        intents: List[InfrastructureIntent],
        contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> List[HealingIntent]:
        if contexts is None:
            contexts = [{}] * len(intents)
        return [self.run(intent, ctx) for intent, ctx in zip(intents, contexts)]
