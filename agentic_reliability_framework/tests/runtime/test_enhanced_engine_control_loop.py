"""
Tests for EnhancedReliabilityEngine control loop.

Validates the refactored pipeline:
ingest_event → orchestrate_analysis → anomaly_detection → risk_scoring
→ policy_evaluation → healing_intent → serialize
"""

import asyncio
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentic_reliability_framework.runtime.engine import EnhancedReliabilityEngine
from agentic_reliability_framework.core.models.event import EventSeverity, HealingAction
from agentic_reliability_framework.runtime.hmc.hmc_learner import HMCRiskLearner


class MockOrchestrator:
    """Mock orchestrator for testing."""
    async def orchestrate_analysis(self, event):
        return {
            'incident_summary': {'anomaly_confidence': 0.85},
            'agent_metadata': {'participating_agents': ['detective', 'diagnostician']}
        }


class MockAnomalyDetector:
    """Mock anomaly detector."""
    def __init__(self, should_detect=True):
        self.should_detect = should_detect

    def detect_anomaly(self, event):
        return self.should_detect


class MockHMCLearner:
    """Mock HMC learner."""
    def __init__(self, is_ready=True):
        self.is_ready = is_ready

    def predict(self, metrics):
        """Return a scalar risk prediction."""
        if self.is_ready:
            return 0.65  # Moderate risk
        return 0.5


class MockBusinessCalculator:
    """Mock business impact calculator."""
    def calculate_impact(self, event):
        return {'estimated_revenue_loss': 15000, 'affected_users': 5000}


class MockClaude:
    """Mock Claude adapter."""
    def generate_completion(self, prompt, system_prompt):
        return "Synthesized incident analysis from Claude"


@pytest.mark.asyncio
async def test_control_loop_with_anomaly_and_hmc():
    """Test full control loop with anomaly detected and HMC ready."""
    engine = EnhancedReliabilityEngine(
        orchestrator=MockOrchestrator(),
        anomaly_detector=MockAnomalyDetector(should_detect=True),
        hmc_learner=MockHMCLearner(is_ready=True),
        business_calculator=MockBusinessCalculator(),
        claude_adapter=MockClaude(),
    )

    result = await engine.process_event_enhanced(
        component="api-gateway",
        latency=450,
        error_rate=0.12,
        throughput=800,
        cpu_util=0.85,
        memory_util=0.70,
    )

    # Verify result structure
    assert result['status'] == 'ANOMALY'
    assert result['is_anomaly'] is True
    assert result['component'] == 'api-gateway'
    assert 'severity' in result
    assert 'risk_score' in result
    assert 'healing_actions' in result
    assert 'healing_intent' in result
    assert 'risk_contributions' in result

    # Verify risk contributions
    assert 'agent_confidence' in result['risk_contributions']
    assert 'hmc' in result['risk_contributions']
    assert 'is_anomaly' in result['risk_contributions']

    # Verify healing intent
    assert result['healing_intent']['component'] == 'api-gateway'
    assert result['healing_intent']['execution_allowed'] is False  # OSS advisory
    assert result['healing_intent']['oss_only'] is True

    # Verify processing metadata
    assert 'processing_metadata' in result
    assert 'pipeline' in result['processing_metadata']
    assert 'ingest→analyze→anomaly→risk→policy→intent' in result['processing_metadata']['pipeline']


@pytest.mark.asyncio
async def test_control_loop_no_anomaly():
    """Test control loop when no anomaly detected."""
    engine = EnhancedReliabilityEngine(
        orchestrator=MockOrchestrator(),
        anomaly_detector=MockAnomalyDetector(should_detect=False),
        hmc_learner=MockHMCLearner(is_ready=False),
        business_calculator=MockBusinessCalculator(),
        claude_adapter=MockClaude(),
    )

    result = await engine.process_event_enhanced(
        component="data-pipeline",
        latency=150,
        error_rate=0.01,
    )

    assert result['status'] == 'NORMAL'
    assert result['is_anomaly'] is False
    assert result['severity'] == EventSeverity.INFO.value
    assert result['business_impact'] is None  # No impact if no anomaly


@pytest.mark.asyncio
async def test_control_loop_invalid_component():
    """Test control loop with invalid component."""
    engine = EnhancedReliabilityEngine(
        orchestrator=MockOrchestrator(),
        anomaly_detector=MockAnomalyDetector(should_detect=False),
    )

    result = await engine.process_event_enhanced(
        component="",  # Invalid: empty
        latency=100,
        error_rate=0.01,
    )

    assert result['status'] == 'INVALID'
    assert 'error' in result


@pytest.mark.asyncio
async def test_severity_determination_critical():
    """Test severity determination for CRITICAL."""
    engine = EnhancedReliabilityEngine(
        orchestrator=MockOrchestrator(),
        anomaly_detector=MockAnomalyDetector(should_detect=True),
        hmc_learner=MockHMCLearner(is_ready=True),
        claude_adapter=MockClaude(),
    )

    # Setup high-risk scenario
    result = await engine.process_event_enhanced(
        component="payment-service",
        latency=600,  # Very high latency
        error_rate=0.35,  # Very high error rate
        cpu_util=0.95,
    )

    assert result['severity'] in [EventSeverity.CRITICAL.value, EventSeverity.HIGH.value]
    assert result['healing_intent']['severity'] in [EventSeverity.CRITICAL.value, EventSeverity.HIGH.value]


@pytest.mark.asyncio
async def test_risk_contributions_with_hmc():
    """Test that risk contributions include HMC contribution."""
    engine = EnhancedReliabilityEngine(
        orchestrator=MockOrchestrator(),
        anomaly_detector=MockAnomalyDetector(should_detect=True),
        hmc_learner=MockHMCLearner(is_ready=True),
        claude_adapter=MockClaude(),
    )

    result = await engine.process_event_enhanced(
        component="cache-server",
        latency=350,
        error_rate=0.18,
    )

    # Verify HMC contribution
    contributions = result['risk_contributions']
    assert 'hmc' in contributions
    assert isinstance(contributions['hmc'], float)
    assert 0.0 <= contributions['hmc'] <= 1.0


@pytest.mark.asyncio
async def test_healing_actions_generated():
    """Test that healing actions are generated via policy evaluation."""
    engine = EnhancedReliabilityEngine(
        orchestrator=MockOrchestrator(),
        anomaly_detector=MockAnomalyDetector(should_detect=True),
        hmc_learner=MockHMCLearner(is_ready=True),
        claude_adapter=MockClaude(),
    )

    result = await engine.process_event_enhanced(
        component="worker-pool",
        latency=520,  # Triggers high_latency_restart policy
        error_rate=0.05,
        cpu_util=0.95,
    )

    healing_actions = result['healing_actions']
    assert isinstance(healing_actions, list)
    assert len(healing_actions) > 0
    # Should have at least ALERT_TEAM
    assert any('ALERT_TEAM' in str(action).upper() for action in healing_actions)


@pytest.mark.asyncio
async def test_oss_advisory_boundary_enforced():
    """Test that OSS advisory boundary (EXECUTION_ALLOWED=False) is enforced."""
    engine = EnhancedReliabilityEngine(
        orchestrator=MockOrchestrator(),
        anomaly_detector=MockAnomalyDetector(should_detect=True),
        hmc_learner=MockHMCLearner(is_ready=True),
        claude_adapter=MockClaude(),
    )

    result = await engine.process_event_enhanced(
        component="test-service",
        latency=400,
        error_rate=0.15,
    )

    # Ensure execution is NOT allowed in OSS
    healing_intent = result['healing_intent']
    assert healing_intent['execution_allowed'] is False
    assert healing_intent['oss_only'] is True


@pytest.mark.asyncio
async def test_claude_enhancement_optional():
    """Test that Claude enhancement is optional and non-blocking."""
    # Mock Claude to raise exception
    mock_claude = MagicMock()
    mock_claude.generate_completion.side_effect = RuntimeError("Claude API unavailable")

    engine = EnhancedReliabilityEngine(
        orchestrator=MockOrchestrator(),
        anomaly_detector=MockAnomalyDetector(should_detect=True),
        hmc_learner=MockHMCLearner(is_ready=True),
        claude_adapter=mock_claude,
    )

    # Should not raise, should handle gracefully
    result = await engine.process_event_enhanced(
        component="service",
        latency=300,
        error_rate=0.10,
    )

    # Result should still be complete
    assert 'component' in result
    assert result['status'] in ['ANOMALY', 'NORMAL']
    # claude_synthesis might not be present due to the error
