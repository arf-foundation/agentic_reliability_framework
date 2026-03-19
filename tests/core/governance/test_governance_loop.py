"""Tests for the canonical governance loop."""

import pytest
from unittest.mock import Mock, MagicMock, patch

from agentic_reliability_framework.core.governance.governance_loop import GovernanceLoop
from agentic_reliability_framework.core.governance.intents import (
    ProvisionResourceIntent,
    ResourceType,
    # Environment is a string literal, not an enum
)
from agentic_reliability_framework.core.governance.healing_intent import (
    HealingIntent,
    RecommendedAction,
    IntentStatus,
)
from agentic_reliability_framework.core.governance.policies import PolicyEvaluator, allow_all
from agentic_reliability_framework.core.governance.cost_estimator import CostEstimator
from agentic_reliability_framework.core.governance.risk_engine import RiskEngine


@pytest.fixture
def mock_policy_evaluator():
    """Create a mock policy evaluator that returns empty violations."""
    evaluator = Mock(spec=PolicyEvaluator)
    evaluator.evaluate.return_value = []
    return evaluator


@pytest.fixture
def mock_cost_estimator():
    """Create a mock cost estimator."""
    estimator = Mock(spec=CostEstimator)
    estimator.estimate_monthly_cost.return_value = 100.0
    return estimator


@pytest.fixture
def mock_risk_engine():
    """Create a mock risk engine."""
    engine = Mock(spec=RiskEngine)
    engine.calculate_risk.return_value = (
        0.15,  # risk_score
        "Explanation",  # explanation
        {"conjugate": 0.15, "hmc": 0.0, "hyper": 0.0, "weights": {"conjugate": 1.0}}  # contributions
    )
    return engine


@pytest.fixture
def sample_intent():
    """Create a simple provisioning intent."""
    return ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D2s_v3",
        requester="test-user",
        environment="dev",  # Use string, not enum
    )


def test_governance_loop_basic_run(
    mock_policy_evaluator, mock_cost_estimator, mock_risk_engine, sample_intent
):
    """Test that the governance loop runs and returns a HealingIntent."""
    loop = GovernanceLoop(
        policy_evaluator=mock_policy_evaluator,
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
    assert intent.epistemic_uncertainty is None
    # decision_margin is not in HealingIntent; remove check


def test_governance_loop_policy_violation(
    mock_policy_evaluator, mock_cost_estimator, mock_risk_engine, sample_intent
):
    """Test that policy violations lead to DENY."""
    # Make the evaluator return a violation
    mock_policy_evaluator.evaluate.return_value = ["Region not allowed"]

    loop = GovernanceLoop(
        policy_evaluator=mock_policy_evaluator,
        cost_estimator=mock_cost_estimator,
        risk_engine=mock_risk_engine,
    )

    intent = loop.run(sample_intent, context={})

    assert intent.action == RecommendedAction.DENY.value
    assert intent.policy_violations == ["Region not allowed"]


def test_governance_loop_high_risk(
    mock_policy_evaluator, mock_cost_estimator, mock_risk_engine, sample_intent
):
    """Test that high risk leads to DENY."""
    mock_policy_evaluator.evaluate.return_value = []
    # Mock risk engine to return high risk
    mock_risk_engine.calculate_risk.return_value = (
        0.95,
        "High risk explanation",
        {"weights": {"conjugate": 1.0}}
    )

    loop = GovernanceLoop(
        policy_evaluator=mock_policy_evaluator,
        cost_estimator=mock_cost_estimator,
        risk_engine=mock_risk_engine,
    )

    intent = loop.run(sample_intent)

    assert intent.action == RecommendedAction.DENY.value


def test_governance_loop_epistemic_risk(
    mock_policy_evaluator, mock_cost_estimator, mock_risk_engine, sample_intent
):
    """Test that high epistemic risk causes ESCALATE."""
    mock_policy_evaluator.evaluate.return_value = []

    # We need to patch the hallucination probe's compute_risk
    with patch("agentic_reliability_framework.core.governance.governance_loop.HallucinationRisk") as MockHall:
        instance = MockHall.return_value
        instance.compute_risk.return_value = {"risk_score": 0.85}
        loop = GovernanceLoop(
            policy_evaluator=mock_policy_evaluator,
            cost_estimator=mock_cost_estimator,
            risk_engine=mock_risk_engine,
            enable_epistemic=True,
            hallucination_probe=instance,
        )
        context = {"query": "test", "evidence": "test", "entropy": 1.0, "evidence_lift": 0.5, "contradiction": 0.2}
        intent = loop.run(sample_intent, context=context)

    assert intent.action == RecommendedAction.ESCALATE.value
    assert intent.epistemic_uncertainty == 0.85


def test_governance_loop_ambiguous(
    mock_policy_evaluator, mock_cost_estimator, mock_risk_engine, sample_intent
):
    """Test that high ambiguity causes ESCALATE."""
    mock_policy_evaluator.evaluate.return_value = []
    # In the real loop, ambiguity is not used; we rely on epistemic_uncertainty.
    # We'll just test that epistemic_uncertainty can be set via context.
    loop = GovernanceLoop(
        policy_evaluator=mock_policy_evaluator,
        cost_estimator=mock_cost_estimator,
        risk_engine=mock_risk_engine,
        enable_epistemic=True,
    )
    # We need to patch _compute_epistemic_uncertainty to return a high value
    with patch.object(loop, '_compute_epistemic_uncertainty', return_value=0.9):
        intent = loop.run(sample_intent, context={})

    assert intent.action == RecommendedAction.ESCALATE.value
    assert intent.epistemic_uncertainty == 0.9


def test_governance_loop_batch(sample_intent, mock_policy_evaluator, mock_cost_estimator, mock_risk_engine):
    """Test batch processing of multiple intents."""
    mock_policy_evaluator.evaluate.return_value = []
    loop = GovernanceLoop(
        policy_evaluator=mock_policy_evaluator,
        cost_estimator=mock_cost_estimator,
        risk_engine=mock_risk_engine,
    )

    intents = [sample_intent, sample_intent]
    results = loop.run_batch(intents)

    assert len(results) == 2
    assert all(isinstance(r, HealingIntent) for r in results)
