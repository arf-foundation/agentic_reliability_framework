"""
Anomaly Detection Agent – specializes in detecting anomalies, pattern recognition,
and hallucination detection for AI systems.
"""

import logging
from typing import Dict, Any, List, Optional
from agentic_reliability_framework.core.models.event import ReliabilityEvent
from agentic_reliability_framework.core.config.constants import (
    LATENCY_WARNING, LATENCY_EXTREME, LATENCY_CRITICAL,
    ERROR_RATE_WARNING, ERROR_RATE_HIGH, ERROR_RATE_CRITICAL,
    CPU_WARNING, CPU_CRITICAL, MEMORY_WARNING, MEMORY_CRITICAL
)
from agentic_reliability_framework.core.nlp.nli import NLIDetector
from .base import BaseAgent, AgentSpecialization

logger = logging.getLogger(__name__)


class AnomalyDetectionAgent(BaseAgent):
    """
    Detects anomalies in infrastructure metrics and hallucinations in AI-generated text.
    Combines traditional anomaly detection with AI-specific hallucination detection.
    """

    def __init__(self, nli_detector: Optional[NLIDetector] = None):
        """
        Args:
            nli_detector: Optional NLI detector for hallucination detection.
                          If not provided, hallucination detection is disabled.
        """
        super().__init__(AgentSpecialization.DETECTIVE)
        self.nli = nli_detector
        self._hallucination_thresholds = {
            'confidence': 0.7,
            'entailment': 0.6
        }
        logger.info("Initialized AnomalyDetectionAgent (hallucination detection: %s)", 
                   "enabled" if nli_detector else "disabled")

    async def analyze(self, event: ReliabilityEvent) -> Dict[str, Any]:
        """
        Analyze event for anomalies and hallucinations.
        Detects:
        - Infrastructure anomalies (latency, errors, resource usage)
        - AI hallucinations (if event contains AI-specific fields)
        """
        try:
            # Check if this is an AI event (has prompt/response)
            is_ai_event = hasattr(event, 'prompt') and hasattr(event, 'response')
            
            if is_ai_event:
                # Run both infrastructure and hallucination analysis
                infra_findings = self._analyze_infrastructure(event)
                hallu_findings = self._analyze_hallucination(event)
                
                # Combine findings
                return {
                    'specialization': self.specialization.value,
                    'confidence': max(infra_findings['confidence'], hallu_findings['confidence']),
                    'findings': {
                        'infrastructure': infra_findings['findings'],
                        'hallucination': hallu_findings['findings']
                    },
                    'recommendations': infra_findings['recommendations'] + hallu_findings['recommendations']
                }
            else:
                # Infrastructure-only analysis
                return self._analyze_infrastructure(event)
                
        except Exception as e:
            logger.error(f"AnomalyDetectionAgent error: {e}", exc_info=True)
            return {
                'specialization': self.specialization.value,
                'confidence': 0.0,
                'findings': {},
                'recommendations': [f"Analysis error: {str(e)}"]
            }

    def _analyze_infrastructure(self, event: ReliabilityEvent) -> Dict[str, Any]:
        """Analyze infrastructure metrics for anomalies."""
        anomaly_score = self._calculate_anomaly_score(event)
        return {
            'specialization': self.specialization.value,
            'confidence': anomaly_score,
            'findings': {
                'anomaly_score': anomaly_score,
                'severity_tier': self._classify_severity(anomaly_score),
                'primary_metrics_affected': self._identify_affected_metrics(event),
                'type': 'infrastructure'
            },
            'recommendations': self._generate_detection_recommendations(event, anomaly_score)
        }

    def _analyze_hallucination(self, event: ReliabilityEvent) -> Dict[str, Any]:
        """
        Analyze AI event for potential hallucinations.
        Combines model confidence score and NLI entailment.
        """
        if self.nli is None:
            return {
                'specialization': self.specialization.value,
                'confidence': 0.0,
                'findings': {'type': 'hallucination', 'note': 'NLI detector not available'},
                'recommendations': []
            }

        try:
            confidence = getattr(event, 'confidence', 0.5)
            prompt = getattr(event, 'prompt', '')
            response = getattr(event, 'response', '')
            
            flags = []
            risk_score = 1.0
            entail_prob = None

            if confidence < self._hallucination_thresholds['confidence']:
                flags.append('low_confidence')
                risk_score *= 0.5

            if prompt and response:
                entail_prob = self.nli.check(prompt, response)
                if entail_prob is not None and entail_prob < self._hallucination_thresholds['entailment']:
                    flags.append('low_entailment')
                    risk_score *= 0.6

            is_hallucination = len(flags) > 0

            return {
                'specialization': self.specialization.value,
                'confidence': 1 - risk_score if is_hallucination else 0,
                'findings': {
                    'is_hallucination': is_hallucination,
                    'flags': flags,
                    'risk_score': risk_score,
                    'confidence': confidence,
                    'entailment': entail_prob,
                    'type': 'hallucination'
                },
                'recommendations': [
                    "Regenerate with lower temperature",
                    "Provide more context",
                    "Use a different model"
                ] if is_hallucination else []
            }
        except Exception as e:
            logger.error(f"Hallucination analysis error: {e}", exc_info=True)
            return {
                'specialization': self.specialization.value,
                'confidence': 0.0,
                'findings': {'type': 'hallucination', 'error': str(e)},
                'recommendations': []
            }

    def _calculate_anomaly_score(self, event: ReliabilityEvent) -> float:
        """Calculate infrastructure anomaly score."""
        scores = []
        if event.latency_p99 > LATENCY_WARNING:
            latency_score = min(1.0, (event.latency_p99 - LATENCY_WARNING) / 500)
            scores.append(0.4 * latency_score)
        if event.error_rate > ERROR_RATE_WARNING:
            error_score = min(1.0, event.error_rate / 0.3)
            scores.append(0.3 * error_score)
        resource_score = 0
        if event.cpu_util and event.cpu_util > CPU_WARNING:
            resource_score += 0.15 * min(1.0, (event.cpu_util - CPU_WARNING) / 0.2)
        if event.memory_util and event.memory_util > MEMORY_WARNING:
            resource_score += 0.15 * min(1.0, (event.memory_util - MEMORY_WARNING) / 0.2)
        scores.append(resource_score)
        return min(1.0, sum(scores))

    def _classify_severity(self, score: float) -> str:
        """Classify anomaly severity."""
        if score > 0.8: return "CRITICAL"
        if score > 0.6: return "HIGH"
        if score > 0.4: return "MEDIUM"
        return "LOW"

    def _identify_affected_metrics(self, event: ReliabilityEvent) -> List[Dict[str, Any]]:
        """Identify which infrastructure metrics are affected."""
        affected = []
        if event.latency_p99 > LATENCY_EXTREME:
            affected.append({"metric": "latency", "value": event.latency_p99, "severity": "CRITICAL", "threshold": LATENCY_WARNING})
        elif event.latency_p99 > LATENCY_CRITICAL:
            affected.append({"metric": "latency", "value": event.latency_p99, "severity": "HIGH", "threshold": LATENCY_WARNING})
        elif event.latency_p99 > LATENCY_WARNING:
            affected.append({"metric": "latency", "value": event.latency_p99, "severity": "MEDIUM", "threshold": LATENCY_WARNING})
        if event.error_rate > ERROR_RATE_CRITICAL:
            affected.append({"metric": "error_rate", "value": event.error_rate, "severity": "CRITICAL", "threshold": ERROR_RATE_WARNING})
        elif event.error_rate > ERROR_RATE_HIGH:
            affected.append({"metric": "error_rate", "value": event.error_rate, "severity": "HIGH", "threshold": ERROR_RATE_WARNING})
        elif event.error_rate > ERROR_RATE_WARNING:
            affected.append({"metric": "error_rate", "value": event.error_rate, "severity": "MEDIUM", "threshold": ERROR_RATE_WARNING})
        if event.cpu_util and event.cpu_util > CPU_CRITICAL:
            affected.append({"metric": "cpu", "value": event.cpu_util, "severity": "CRITICAL", "threshold": CPU_WARNING})
        elif event.cpu_util and event.cpu_util > CPU_WARNING:
            affected.append({"metric": "cpu", "value": event.cpu_util, "severity": "HIGH", "threshold": CPU_WARNING})
        if event.memory_util and event.memory_util > MEMORY_CRITICAL:
            affected.append({"metric": "memory", "value": event.memory_util, "severity": "CRITICAL", "threshold": MEMORY_WARNING})
        elif event.memory_util and event.memory_util > MEMORY_WARNING:
            affected.append({"metric": "memory", "value": event.memory_util, "severity": "HIGH", "threshold": MEMORY_WARNING})
        return affected

    def _generate_detection_recommendations(self, event: ReliabilityEvent, anomaly_score: float) -> List[str]:
        """Generate recommendations based on infrastructure anomalies."""
        recommendations = []
        for metric in self._identify_affected_metrics(event):
            m = metric["metric"]
            sev = metric["severity"]
            val = metric["value"]
            thr = metric["threshold"]
            if m == "latency":
                if sev == "CRITICAL":
                    recommendations.append(f"🚨 CRITICAL: Latency {val:.0f}ms (>{thr}ms) - Check database & external dependencies")
                elif sev == "HIGH":
                    recommendations.append(f"⚠️ HIGH: Latency {val:.0f}ms (>{thr}ms) - Investigate service performance")
                else:
                    recommendations.append(f"📈 Latency elevated: {val:.0f}ms (>{thr}ms) - Monitor trend")
            elif m == "error_rate":
                if sev == "CRITICAL":
                    recommendations.append(f"🚨 CRITICAL: Error rate {val*100:.1f}% (>{thr*100:.1f}%) - Check recent deployments")
                elif sev == "HIGH":
                    recommendations.append(f"⚠️ HIGH: Error rate {val*100:.1f}% (>{thr*100:.1f}%) - Review application logs")
                else:
                    recommendations.append(f"📈 Errors increasing: {val*100:.1f}% (>{thr*100:.1f}%)")
            elif m == "cpu":
                recommendations.append(f"🔥 CPU {sev}: {val*100:.1f}% utilization - Consider scaling")
            elif m == "memory":
                recommendations.append(f"💾 Memory {sev}: {val*100:.1f}% utilization - Check for memory leaks")
        if anomaly_score > 0.8:
            recommendations.append("🎯 IMMEDIATE ACTION REQUIRED: Multiple critical metrics affected")
        elif anomaly_score > 0.6:
            recommendations.append("🎯 INVESTIGATE: Significant performance degradation detected")
        elif anomaly_score > 0.4:
            recommendations.append("📊 MONITOR: Early warning signs detected")
        return recommendations[:4]
