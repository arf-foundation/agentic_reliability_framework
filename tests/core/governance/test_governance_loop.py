"""Tests for the canonical governance loop."""

import pytest
from unittest.mock import Mock, MagicMock, patch

from agentic_reliability_framework.core.governance.governance_loop import GovernanceLoop
from agentic_reliability_framework.core.governance.intents import (
    ProvisionResourceIntent,
    ResourceType,
    Environment,
)
from agentic_reliability_framework.core.governance.healing_intent import (
    HealingIntent,
    RecommendedAction,
    IntentStatus,
)
from agentic_reliability_framework.core.governance.policy_engine import PolicyEngine
from agentic_reliability_framework.core.governance.cost_estimator import CostEstimator
from agentic_reliability_framework.core.governance.risk_engine import RiskEngine


@pytest.fixture
def mock_policy_engine():
    """Create a mock policy engine that returns empty violations."""
    engine = Mock(spec=PolicyEngine)
    # We'll override specific methods as needed
    return engine


@pytest.fixture
def mock_cost_estimator():
    """Create a mock cost estimator."""
    engine = Mock(spec=CostEstimator)
    engine.estimate_monthly_cost.return_value = 100.0
    return engine


@pytest.fixture
def mock_risk_engine():
    """Create a mock risk engine."""
    engine = Mock(spec=RiskEngine)
    engine.calculate_risk.return_value = {
        "risk_score": 0.15,
        "contributions": {"conjugate": 0.15, "hmc": 0.0, "hyper": 0.0},
    }
    return engine


@pytest.fixture
def sample_intent():
    """Create a simple provisioning intent."""
    return ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D2s_v3",
        requester="test-user",
        environment=Environment.DEV,
    )


def test_governance_loop_basic_run(
    mock_policy_engine, mock_cost_estimator, mock_risk_engine, sample_intent
):
    """Test that the governance loop runs and returns a HealingIntent."""
    # Setup mocks
    mock_policy_engine.evaluate_policies.return_value = []
    
    loop = GovernanceLoop(
        policy_engine=mock_policy_engine,
        cost_estimator=mock_cost_estimator,
        risk_engine=mock_risk_engine,
        enable_epistemic=False,
    )

    intent = loop.run(sample_intent, context={"incident_id": "test-123"})

    assert isinstance(intent, HealingIntent)
    assert intent.status == IntentStatus.OSS_ADVISORY_ONLY
    assert intent.source.value == "infrastructure_analysis"
    assert intent.risk_score == 0.15
    assert intent.cost_projection == 100.0
    assert intent.policy_violations == []
    assert intent.epistemic_uncertainty is None  # because enable_epistemic=False
    assert intent.decision_margin is None  # not escalated


def test_governance_loop_policy_violation(
    mock_policy_engine, mock_cost_estimator, mock_risk_engine, sample_intent
):
    """Test that policy violations lead to DENY."""
    mock_policy_engine.evaluate_policies.return_value = []
    # Simulate hard constraint violation via context
    context = {"policy_violations": ["Region not allowed"]}

    loop = GovernanceLoop(
        policy_engine=mock_policy_engine,
        cost_estimator=mock_cost_estimator,
        risk_engine=mock_risk_engine,
    )

    intent = loop.run(sample_intent, context=context)

    assert intent.action == RecommendedAction.DENY.value
    assert "Hard policy constraints violated" in intent.justification
    assert intent.policy_violations == ["Region not allowed"]


def test_governance_loop_high_risk(
    mock_policy_engine, mock_cost_estimator, mock_risk_engine, sample_intent
):
    """Test that high risk leads to DENY."""
    mock_policy_engine.evaluate_policies.return_value = []
    mock_risk_engine.calculate_risk.return_value = {
        "risk_score": 0.95,
        "contributions": {"conjugate": 0.95, "hmc": 0.0, "hyper": 0.0},
    }

    loop = GovernanceLoop(
        policy_engine=mock_policy_engine,
        cost_estimator=mock_cost_estimator,
        risk_engine=mock_risk_engine,
    )

    intent = loop.run(sample_intent)

    assert intent.action == RecommendedAction.DENY.value
    assert "above threshold" in intent.justification


def test_governance_loop_epistemic_risk(
    mock_policy_engine, mock_cost_estimator, mock_risk_engine, sample_intent
):
    """Test that high epistemic risk causes ESCALATE."""
    mock_policy_engine.evaluate_policies.return_value = []
    
    # We need to mock the research probe import inside _compute_epistemic_risk
    # This requires patching at the module level.
    with patch(
        "agentic_reliability_framework.core.governance.governance_loop.hallucination_risk",
        return_value=0.85,
    ):
        loop = GovernanceLoop(
            policy_engine=mock_policy_engine,
            cost_estimator=mock_cost_estimator,
            risk_engine=mock_risk_engine,
            enable_epistemic=True,
        )
        # Also mock the internal _compute_epistemic_risk to return a high value
        # Since the real function is patched above, it will return 0.85
        intent = loop.run(sample_intent)

    assert intent.action == RecommendedAction.ESCALATE.value
    assert "epistemic uncertainty" in intent.justification.lower()
    assert intent.epistemic_uncertainty == 0.85


def test_governance_loop_ambiguous(
    mock_policy_engine, mock_cost_estimator, mock_risk_engine, sample_intent
):
    """Test that high ambiguity causes ESCALATE."""
    mock_policy_engine.evaluate_policies.return_value = []
    context = {"ambiguity": 0.9}

    loop = GovernanceLoop(
        policy_engine=mock_policy_engine,
        cost_estimator=mock_cost_estimator,
        risk_engine=mock_risk_engine,
    )

    intent = loop.run(sample_intent, context=context)

    assert intent.action == RecommendedAction.ESCALATE.value
    assert "too ambiguous" in intent.justification
    assert intent.ambiguity_score == 0.9


def test_governance_loop_batch(sample_intent, mock_policy_engine, mock_cost_estimator, mock_risk_engine):
    """Test batch processing of multiple intents."""
    mock_policy_engine.evaluate_policies.return_value = []
    loop = GovernanceLoop(
        policy_engine=mock_policy_engine,
        cost_estimator=mock_cost_estimator,
        risk_engine=mock_risk_engine,
    )

    intents = [sample_intent, sample_intent]
    results = loop.run_batch(intents)

    assert len(results) == 2
    assert all(isinstance(r, HealingIntent) for r in results)
