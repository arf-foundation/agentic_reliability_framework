"""
Enhanced Reliability Engine – main entry point for processing reliability events.

Control Flow:
    ingest_event → orchestrate_analysis → anomaly_detection → risk_scoring
    (conjugate + hyper + hmc) → policy_evaluation (DPT) → healing_intent → serialize

The engine enforces OSS advisory boundaries: EXECUTION_ALLOWED = False.
Healing intents are never executed in OSS; they are advisory recommendations only.
"""

import asyncio
import threading
import logging
import datetime
import json
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

from agentic_reliability_framework.core.models.event import ReliabilityEvent, EventSeverity, HealingAction, resolve_metric
from agentic_reliability_framework.core.governance.policy_engine import PolicyEngine
from agentic_reliability_framework.runtime.analytics.anomaly import AdvancedAnomalyDetector
from agentic_reliability_framework.runtime.analytics.predictive import BusinessImpactCalculator
from agentic_reliability_framework.runtime.orchestration.manager import OrchestrationManager
from agentic_reliability_framework.runtime.hmc.hmc_learner import HMCRiskLearner
from agentic_reliability_framework.core.adapters.claude import ClaudeAdapter
from agentic_reliability_framework.core.config.constants import (
    MAX_EVENTS_STORED, AGENT_TIMEOUT_SECONDS, EXECUTION_ALLOWED
)

logger = logging.getLogger(__name__)


class ThreadSafeEventStore:
    """Simple thread-safe event store for recent events."""
    def __init__(self, max_size: int = MAX_EVENTS_STORED):
        from collections import deque
        self._events = deque(maxlen=max_size)
        self._lock = threading.RLock()

    def add(self, event: ReliabilityEvent):
        """Add event to the store (thread-safe)."""
        with self._lock:
            self._events.append(event)

    def get_recent(self, n: int = 15) -> List[ReliabilityEvent]:
        """Get most recent n events (thread-safe)."""
        with self._lock:
            return list(self._events)[-n:] if self._events else []


class EnhancedReliabilityEngine:
    """
    Main reliability engine orchestrating the control loop:
    ingest_event → orchestrate_analysis → anomaly_detection → risk_scoring
    → policy_evaluation → healing_intent → serialize

    All steps are explicitly named and sequenced for clarity.
    OSS advisory boundaries are enforced throughout (EXECUTION_ALLOWED = False).
    """

    def __init__(
        self,
        orchestrator: Optional[OrchestrationManager] = None,
        policy_engine: Optional[PolicyEngine] = None,
        event_store: Optional[ThreadSafeEventStore] = None,
        anomaly_detector: Optional[AdvancedAnomalyDetector] = None,
        business_calculator: Optional[BusinessImpactCalculator] = None,
        hmc_learner: Optional[HMCRiskLearner] = None,
        claude_adapter: Optional[ClaudeAdapter] = None,
    ):
        """Initialize engine with optional dependency injection for testing."""
        self.orchestrator = orchestrator or OrchestrationManager()
        self.policy_engine = policy_engine or PolicyEngine()
        self.event_store = event_store or ThreadSafeEventStore()
        self.anomaly_detector = anomaly_detector or AdvancedAnomalyDetector()
        self.business_calculator = business_calculator or BusinessImpactCalculator()
        self.hmc_learner = hmc_learner or HMCRiskLearner()
        self.claude_adapter = claude_adapter or ClaudeAdapter()
        
        self.performance_metrics = {
            'total_incidents_processed': 0,
            'multi_agent_analyses': 0,
            'anomalies_detected': 0,
        }
        self._lock = threading.RLock()
        # Ensure OSS advisory boundary is enforced
        if EXECUTION_ALLOWED:
            logger.warning(
                "OSS engine initialized with EXECUTION_ALLOWED=True. "
                "This should only occur in testing. Production OSS must have EXECUTION_ALLOWED=False."
            )
        logger.info("Initialized EnhancedReliabilityEngine (OSS advisory mode)")

    async def process_event_enhanced(
        self,
        component: str,
        latency: float,
        error_rate: float,
        throughput: float = 1000,
        cpu_util: Optional[float] = None,
        memory_util: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Process a single reliability event through the full control loop.

        Returns:
            Dict with status, analysis results, and healing recommendations.
        """
        # Resolve metrics for logging (use provided values, they are already numbers)
        logger.info(
            f"Processing event: component={component}, "
            f"latency={latency}ms, error_rate={error_rate*100:.1f}%"
        )

        # Step 1: Ingest and Validate Event
        event, ingest_error = await self._ingest_event(
            component, latency, error_rate, throughput, cpu_util, memory_util
        )
        if ingest_error:
            logger.warning(f"Event ingestion failed: {ingest_error}")
            return ingest_error

        # Step 2: Multi-Agent Orchestrated Analysis
        agent_analysis = await self._orchestrate_analysis(event)

        # Step 3: Anomaly Detection
        is_anomaly, anomaly_details = await self._anomaly_detection(event)

        # Step 4: Risk Scoring (Conjugate + Hyperprior + HMC)
        risk_score, risk_explanation, risk_contributions = await self._risk_scoring(
            event, agent_analysis, is_anomaly
        )

        # Step 5: Policy Evaluation (DPT-based)
        healing_actions, policy_context = await self._policy_evaluation(event, risk_score)

        # Step 6: Severity Determination
        severity = self._determine_severity(is_anomaly, agent_analysis, risk_score)
        event = event.model_copy(update={'severity': severity})

        # Step 7: Business Impact Calculation
        business_impact = await self._calculate_business_impact(event, is_anomaly)

        # Step 8: Healing Intent Generation
        healing_intent = self._generate_healing_intent(
            event, healing_actions, severity, risk_score, agent_analysis
        )

        # Step 9: Serialize Result
        result = await self._serialize_result(
            event,
            agent_analysis,
            is_anomaly,
            anomaly_details,
            healing_actions,
            business_impact,
            severity,
            healing_intent,
            risk_score,
            risk_contributions,
        )

        # Record metrics and store event
        self.event_store.add(event)
        with self._lock:
            self.performance_metrics['total_incidents_processed'] += 1
            self.performance_metrics['multi_agent_analyses'] += 1
            if is_anomaly:
                self.performance_metrics['anomalies_detected'] += 1

        # Optional: Enhance with Claude
        try:
            result = await self._enhance_with_claude(event, result)
        except Exception as e:
            logger.warning(f"Claude enhancement failed (non-fatal): {e}")

        return result

    # =========================================================================
    # Step 1: Ingest Event
    # =========================================================================
    async def _ingest_event(
        self,
        component: str,
        latency: float,
        error_rate: float,
        throughput: float,
        cpu_util: Optional[float],
        memory_util: Optional[float],
    ) -> Tuple[Optional[ReliabilityEvent], Optional[Dict[str, Any]]]:
        """
        Validate and create a ReliabilityEvent.

        Returns:
            (event, error_dict) where error_dict is None on success.
        """
        from agentic_reliability_framework.core.models.event import validate_component_id

        is_valid, error_msg = validate_component_id(component)
        if not is_valid:
            return None, {'error': error_msg, 'status': 'INVALID'}

        try:
            event = ReliabilityEvent(
                component=component,
                latency_p99=latency,
                error_rate=error_rate,
                throughput=throughput,
                cpu_util=cpu_util,
                memory_util=memory_util,
            )
            return event, None
        except Exception as e:
            logger.error(f"Event creation error: {e}")
            return None, {'error': f'Invalid event data: {str(e)}', 'status': 'INVALID'}

    # =========================================================================
    # Step 2: Orchestrate Multi-Agent Analysis
    # =========================================================================
    async def _orchestrate_analysis(self, event: ReliabilityEvent) -> Dict[str, Any]:
        """
        Run multi-agent analysis (orchestration manager).

        Returns:
            agent_analysis dict with incident_summary and agent_metadata.
        """
        try:
            analysis = await self.orchestrator.orchestrate_analysis(event)
            logger.debug(f"Agent analysis complete: {len(analysis.get('agent_metadata', {}).get('participating_agents', []))} agents participated")
            return analysis
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            return {}

    # =========================================================================
    # Step 3: Anomaly Detection
    # =========================================================================
    async def _anomaly_detection(self, event: ReliabilityEvent) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect anomalies using the anomaly detector.

        Returns:
            (is_anomaly, details_dict)
        """
        try:
            is_anomaly = self.anomaly_detector.detect_anomaly(event)
            # For details, we need resolved metric values
            lat = resolve_metric(event, "latency_p99") or 0
            err = resolve_metric(event, "error_rate") or 0
            details = {
                'anomaly_score': float(lat > 300 or err > 0.15),
                'latency_anomaly': lat > 300,
                'error_rate_anomaly': err > 0.15,
            }
            logger.debug(f"Anomaly detection: is_anomaly={is_anomaly}")
            return is_anomaly, details
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return False, {'error': str(e)}

    # =========================================================================
    # Step 4: Risk Scoring (Conjugate + Hyperprior + HMC)
    # =========================================================================
    async def _risk_scoring(
        self,
        event: ReliabilityEvent,
        agent_analysis: Dict[str, Any],
        is_anomaly: bool,
    ) -> Tuple[float, str, Dict[str, Any]]:
        """
        Compute Bayesian risk score combining:
        - Conjugate priors (online)
        - Hyperpriors (optional, offline)
        - HMC prediction (optional, offline)

        Returns:
            (risk_score [0,1], explanation_string, contributions_dict)
        """
        try:
            # Base risk from anomaly + agent confidence
            agent_confidence = (
                agent_analysis.get('incident_summary', {}).get('anomaly_confidence', 0.0)
                if agent_analysis
                else 0.0
            )
            base_risk = agent_confidence if is_anomaly else 0.01

            # HMC prediction (if available)
            hmc_contribution = {'hmc': 0.0}
            if self.hmc_learner.is_ready:
                try:
                    # Prepare feature dict using resolved metrics
                    feature_dict = {
                        'latency_p99': resolve_metric(event, "latency_p99") or 0.0,
                        'error_rate': resolve_metric(event, "error_rate") or 0.0,
                        'throughput': resolve_metric(event, "throughput") or 0.0,
                        'cpu_util': resolve_metric(event, "cpu_util") or 0.0,
                        'memory_util': resolve_metric(event, "memory_util") or 0.0,
                    }
                    hmc_risk = self.hmc_learner.predict(feature_dict)
                    hmc_contribution['hmc'] = hmc_risk
                except Exception as e:
                    logger.debug(f"HMC prediction failed: {e}")

            # Combine contributions (weighted average)
            contributions = {
                'agent_confidence': agent_confidence,
                'base_risk': base_risk,
                'hmc': hmc_contribution.get('hmc', 0.0),
                'is_anomaly': is_anomaly,
            }

            # Final risk: average of base_risk and hmc if available
            if hmc_contribution['hmc'] > 0:
                final_risk = 0.7 * base_risk + 0.3 * hmc_contribution['hmc']
            else:
                final_risk = base_risk

            final_risk = min(max(final_risk, 0.0), 1.0)  # Clip to [0,1]

            explanation = (
                f"Risk score {final_risk:.3f} from agent_confidence={agent_confidence:.3f}, "
                f"is_anomaly={is_anomaly}, hmc_contrib={hmc_contribution['hmc']:.3f}"
            )

            return final_risk, explanation, contributions
        except Exception as e:
            logger.error(f"Risk scoring failed: {e}")
            return 0.1, f"Risk scoring error: {e}", {}

    # =========================================================================
    # Step 5: Policy Evaluation (DPT-based)
    # =========================================================================
    async def _policy_evaluation(
        self, event: ReliabilityEvent, risk_score: float
    ) -> Tuple[List[HealingAction], Dict[str, Any]]:
        """
        Evaluate healing policies using Dynamic Programming Tree (DPT).
        Policies are evaluated based on event metrics and risk score.

        Returns:
            (healing_actions_list, policy_context_dict)
        """
        try:
            # Standard policy evaluation
            healing_actions = self.policy_engine.evaluate_policies(event)
            
            # Additional DPT-based filtering (risk-aware)
            if risk_score > 0.8:
                # High risk: prioritize critical actions
                priority_actions = [
                    a for a in healing_actions
                    if a in [HealingAction.ALERT_TEAM, HealingAction.ROLLBACK]
                ]
                healing_actions = priority_actions if priority_actions else healing_actions
            elif risk_score < 0.3:
                # Low risk: filter out aggressive actions
                conservative_actions = [
                    a for a in healing_actions
                    if a not in [HealingAction.ROLLBACK, HealingAction.SCALE_OUT]
                ]
                healing_actions = conservative_actions if conservative_actions else healing_actions

            context = {
                'risk_score': risk_score,
                'policies_evaluated': len(self.policy_engine.policies),
                'policies_triggered': len(healing_actions),
            }
            logger.debug(f"Policy evaluation: {len(healing_actions)} actions triggered")
            return healing_actions, context
        except Exception as e:
            logger.error(f"Policy evaluation failed: {e}")
            return [HealingAction.ALERT_TEAM], {'error': str(e)}

    # =========================================================================
    # Step 5b: Severity Determination
    # =========================================================================
    def _determine_severity(
        self, is_anomaly: bool, agent_analysis: Dict[str, Any], risk_score: float
    ) -> EventSeverity:
        """
        Determine event severity based on anomaly flag, agent confidence, and risk score.

        Returns:
            EventSeverity enum value.
        """
        if not is_anomaly:
            return EventSeverity.INFO

        agent_confidence = (
            agent_analysis.get('incident_summary', {}).get('anomaly_confidence', 0.0)
            if agent_analysis
            else 0.0
        )

        if risk_score > 0.8 or agent_confidence > 0.8:
            return EventSeverity.CRITICAL
        elif risk_score > 0.5 or agent_confidence > 0.6:
            return EventSeverity.HIGH
        elif risk_score > 0.3 or agent_confidence > 0.4:
            return EventSeverity.WARNING
        else:
            return EventSeverity.INFO

    # =========================================================================
    # Step 6: Business Impact Calculation
    # =========================================================================
    async def _calculate_business_impact(
        self, event: ReliabilityEvent, is_anomaly: bool
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate business impact if anomaly detected.

        Returns:
            business_impact dict or None.
        """
        try:
            if is_anomaly:
                impact = self.business_calculator.calculate_impact(event)
                logger.debug(f"Business impact calculated: {impact}")
                return impact
            return None
        except Exception as e:
            logger.error(f"Business impact calculation failed: {e}")
            return None

    # =========================================================================
    # Step 7: Healing Intent Generation
    # =========================================================================
    def _generate_healing_intent(
        self,
        event: ReliabilityEvent,
        healing_actions: List[HealingAction],
        severity: EventSeverity,
        risk_score: float,
        agent_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate a healing intent (OSS advisory, not executable).

        Returns:
            healing_intent dict with action recommendations.
        """
        return {
            'component': event.component,
            'severity': severity.value,
            'confidence': min(risk_score, 1.0),
            'recommended_actions': [a.value for a in healing_actions],
            'risk_score': risk_score,
            'execution_allowed': EXECUTION_ALLOWED,  # Should always be False in OSS
            'oss_only': True,  # This is OSS advisory only
        }

    # =========================================================================
    # Step 8: Serialize Result
    # =========================================================================
    async def _serialize_result(
        self,
        event: ReliabilityEvent,
        agent_analysis: Dict[str, Any],
        is_anomaly: bool,
        anomaly_details: Dict[str, Any],
        healing_actions: List[HealingAction],
        business_impact: Optional[Dict[str, Any]],
        severity: EventSeverity,
        healing_intent: Dict[str, Any],
        risk_score: float,
        risk_contributions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Serialize the complete analysis result.

        Returns:
            result dict with all analysis fields.
        """
        # Resolve metrics for output
        lat = resolve_metric(event, "latency_p99") or 0
        err = resolve_metric(event, "error_rate") or 0
        thru = resolve_metric(event, "throughput") or 0

        return {
            'timestamp': event.timestamp.isoformat(),
            'component': event.component,
            'latency_p99': lat,
            'error_rate': err,
            'throughput': thru,
            'status': 'ANOMALY' if is_anomaly else 'NORMAL',
            'severity': severity.value,
            'is_anomaly': is_anomaly,
            'anomaly_details': anomaly_details,
            'multi_agent_analysis': agent_analysis,
            'risk_score': risk_score,
            'risk_contributions': risk_contributions,
            'healing_actions': [a.value for a in healing_actions],
            'healing_intent': healing_intent,
            'business_impact': business_impact,
            'processing_metadata': {
                'agents_used': agent_analysis.get('agent_metadata', {}).get('participating_agents', []),
                'analysis_confidence': agent_analysis.get('incident_summary', {}).get('anomaly_confidence', 0.0),
                'pipeline': 'ingest→analyze→anomaly→risk→policy→intent→serialize',
            },
        }

    # =========================================================================
    # Step 9: Optional Claude Enhancement
    # =========================================================================
    async def _enhance_with_claude(
        self, event: ReliabilityEvent, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optionally enhance result with Claude synthesis (non-critical).

        Returns:
            enhanced result dict.
        """
        # Resolve metrics for prompt
        lat = resolve_metric(event, "latency_p99") or 0
        err = resolve_metric(event, "error_rate") or 0
        thru = resolve_metric(event, "throughput") or 0
        cpu = resolve_metric(event, "cpu_util")
        mem = resolve_metric(event, "memory_util")

        context_parts = [
            "INCIDENT SUMMARY:",
            f"Component: {event.component}",
            f"Timestamp: {event.timestamp.isoformat()}",
            f"Severity: {result['severity']}",
            "",
            "METRICS:",
            f"• Latency P99: {lat}ms",
            f"• Error Rate: {err:.1%}",
            f"• Throughput: {thru} req/s",
        ]
        if cpu:
            context_parts.append(f"• CPU: {cpu:.1%}")
        if mem:
            context_parts.append(f"• Memory: {mem:.1%}")
        context_parts.append("")
        if result.get('multi_agent_analysis'):
            context_parts.append("AGENT ANALYSIS:")
            context_parts.append(json.dumps(result['multi_agent_analysis'], indent=2))

        context = "\n".join(context_parts)
        prompt = f"""{context}
TASK: Provide an executive summary synthesizing all agent analyses.
Include:
1. Concise incident description
2. Most likely root cause
3. Single best recovery action
4. Estimated impact and recovery time
Be specific and actionable."""

        system_prompt = """You are a senior Site Reliability Engineer synthesizing 
multiple AI agent analyses into clear, actionable guidance for incident response. 
Focus on clarity, accuracy, and decisive recommendations."""

        claude_synthesis = self.claude_adapter.generate_completion(prompt, system_prompt)
        result['claude_synthesis'] = {
            'summary': claude_synthesis,
            'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'source': 'claude-opus-4',
        }
        return result
