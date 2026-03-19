"""
Canonical Governance Loop – orchestrates policy, cost, risk, epistemic, and memory analysis.
Integrates the ECLIPSE hallucination probe for epistemic uncertainty quantification.
"""

import logging
import time
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

logger = logging.getLogger(__name__)

class GovernanceLoop:
    """
    Orchestrates the full governance evaluation, integrating policy, cost, risk,
    epistemic uncertainty (including hallucination detection), and semantic memory.
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
        """
        self.policy_evaluator = policy_evaluator
        self.cost_estimator = cost_estimator
        self.risk_engine = risk_engine
        self.memory = memory
        self.enable_epistemic = enable_epistemic
        self.hallucination_probe = hallucination_probe
        self.dpt_low = dpt_low
        self.dpt_high = dpt_high

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

        # 4. Epistemic uncertainty (if enabled)
        epistemic_uncertainty = None
        hallucination_risk_score = None
        if self.enable_epistemic:
            epistemic_uncertainty = self._compute_epistemic_uncertainty(intent, context, risk_score)
            # The hallucination probe can be part of epistemic uncertainty
            if self.hallucination_probe and "query" in context and "evidence" in context:
                # Pre‑computed signals may be provided, or we compute them if model/tokenizer are in context
                entropy = context.get("entropy")
                evidence_lift = context.get("evidence_lift")
                contradiction = context.get("contradiction")
                if entropy is not None and evidence_lift is not None and contradiction is not None:
                    result = self.hallucination_probe.compute_risk(entropy, evidence_lift, contradiction)
                    hallucination_risk_score = result["risk_score"]
                    # Combine with other epistemic sources (e.g., data sparsity)
                    epistemic_uncertainty = hallucination_risk_score  # or a weighted blend

        # 5. Semantic memory retrieval (similar incidents)
        similar_incidents = self._retrieve_similar_incidents(intent, context)

        # 6. Determine recommended action based on DPT thresholds
        recommended_action = self._determine_action(risk_score, policy_violations, epistemic_uncertainty)

        # 7. Build HealingIntent using the factory method
        healing_intent = HealingIntent.from_analysis(
            action=recommended_action.value,
            component=getattr(intent, 'service_name', getattr(intent, 'component', 'unknown')),
            parameters={},  # You can fill with intent‑specific parameters
            justification=explanation,
            confidence=1.0 - (epistemic_uncertainty or 0.05),  # base confidence
            similar_incidents=similar_incidents,
            reasoning_chain=[{"step": "governance_loop", "details": contributions}],
            incident_id=context.get("incident_id", ""),
            source=IntentSource.INFRASTRUCTURE_ANALYSIS,
            rag_similarity_score=None,  # Not used yet
            risk_score=risk_score,
            cost_projection=cost_projection,
        )
        # Mark as OSS advisory (no execution)
        healing_intent = healing_intent.mark_as_oss_advisory()
        return healing_intent

    def _compute_epistemic_uncertainty(
        self,
        intent: InfrastructureIntent,
        context: Dict[str, Any],
        risk_score: float,
    ) -> float:
        """
        Combine multiple sources of epistemic uncertainty:
        - Data sparsity (from risk engine's weight calculation)
        - Hallucination risk (if probe available)
        - Ambiguity from context
        Returns a number between 0 and 1.
        """
        uncertainty = 0.05  # baseline

        # Example: if risk engine's weight for HMC is low, that indicates uncertainty
        # (This would require access to risk engine's internals; not implemented here)

        if self.hallucination_probe and "query" in context and "evidence" in context:
            # Use precomputed signals or compute them
            entropy = context.get("entropy")
            evidence_lift = context.get("evidence_lift")
            contradiction = context.get("contradiction")
            if entropy is not None and evidence_lift is not None and contradiction is not None:
                result = self.hallucination_probe.compute_risk(entropy, evidence_lift, contradiction)
                uncertainty = max(uncertainty, result["risk_score"])

        # Cap to [0,1]
        return max(0.0, min(1.0, uncertainty))

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

        # We need an event-like object to pass to find_similar.
        # For now, we'll attempt to create a minimal event from the intent.
        # This is a placeholder – you may need to adapt based on your memory implementation.
        # If not possible, return empty list.
        try:
            # Create a dummy event with fields that memory expects (component, etc.)
            class DummyEvent:
                component = getattr(intent, 'service_name', 'unknown')
                # other fields like latency_p99, error_rate would be needed for embedding;
                # without them, embedding may be meaningless. We'll return empty.
                pass
            event = DummyEvent()
            analysis = {}  # empty analysis
            nodes = self.memory.find_similar(event, analysis, k=k)
            # Convert IncidentNode objects to dicts
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

    def _determine_action(
        self,
        risk_score: float,
        policy_violations: List[str],
        epistemic_uncertainty: Optional[float],
    ) -> RecommendedAction:
        """
        Determine the recommended action based on DPT thresholds and policy violations.
        """
        if policy_violations:
            # Hard policy violations lead to deny (or escalate depending on policy)
            return RecommendedAction.DENY
        if risk_score < self.dpt_low:
            return RecommendedAction.APPROVE
        if risk_score > self.dpt_high:
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
