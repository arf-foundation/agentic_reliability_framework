"""Tests for the canonical governance loop."""
import pytest
from unittest.mock import Mock, patch

from agentic_reliability_framework.core.governance.governance_loop import GovernanceLoop
from agentic_reliability_framework.core.governance.intents import (
    ProvisionResourceIntent,
    ResourceType,
)
from agentic_reliability_framework.core.governance.healing_intent import (
    HealingIntent,
    RecommendedAction,
    IntentStatus,
)
from agentic_reliability_framework.core.governance.policies import PolicyEvaluator
from agentic_reliability_framework.core.governance.cost_estimator import CostEstimator
from agentic_reliability_framework.core.governance.risk_engine import RiskEngine
from agentic_reliability_framework.runtime.analytics.predictive import SimplePredictiveEngine, ForecastResult
from agentic_reliability_framework.core.research.eclipse_probe import HallucinationRisk


@pytest.fixture
def mock_policy_evaluator():
    evaluator = Mock(spec=PolicyEvaluator)
    evaluator.evaluate.return_value = []
    return evaluator


@pytest.fixture
def mock_cost_estimator():
    estimator = Mock(spec=CostEstimator)
    estimator.estimate_monthly_cost.return_value = 100.0
    return estimator


@pytest.fixture
def mock_risk_engine():
    engine = Mock(spec=RiskEngine)
    engine.calculate_risk.return_value = (
        0.15,
        "Explanation",
        {"conjugate_alpha": 1.5, "conjugate_beta": 8.0, "weights": {"conjugate": 1.0}}
    )
    return engine


@pytest.fixture
def sample_intent():
    return ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D2s_v3",
        requester="test-user",
        environment="dev",
    )


def test_governance_loop_basic_run(
    mock_policy_evaluator, mock_cost_estimator, mock_risk_engine, sample_intent
):
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
    assert intent.policy_violations == ()
    assert "predictive_risk" in intent.metadata
    assert "epistemic_breakdown" in intent.metadata
    assert "decision_trace" in intent.metadata  # NEW


def test_governance_loop_policy_violation(
    mock_policy_evaluator, mock_cost_estimator, mock_risk_engine, sample_intent
):
    mock_policy_evaluator.evaluate.return_value = ["Region not allowed"]

    loop = GovernanceLoop(
        policy_evaluator=mock_policy_evaluator,
        cost_estimator=mock_cost_estimator,
        risk_engine=mock_risk_engine,
    )
    intent = loop.run(sample_intent, context={})

    assert intent.policy_violations == ("Region not allowed",)
    assert intent.action == RecommendedAction.DENY.value


def test_governance_loop_high_risk(
    mock_policy_evaluator, mock_cost_estimator, mock_risk_engine, sample_intent
):
    mock_risk_engine.calculate_risk.return_value = (0.95, "High risk", {"weights": {}})

    loop = GovernanceLoop(
        policy_evaluator=mock_policy_evaluator,
        cost_estimator=mock_cost_estimator,
        risk_engine=mock_risk_engine,
    )
    intent = loop.run(sample_intent)

    assert intent.action == RecommendedAction.DENY.value


def test_governance_loop_epistemic_gate(
    mock_policy_evaluator, mock_cost_estimator, mock_risk_engine, sample_intent
):
    """Test that high epistemic uncertainty triggers ESCALATE via gate."""
    # Create a mock hallucination probe that returns high risk
    mock_hall = Mock(spec=HallucinationRisk)
    mock_hall.compute_risk.return_value = {"risk_score": 0.9}

    # Create a mock predictive engine that returns low confidence forecasts
    mock_predictive = Mock(spec=SimplePredictiveEngine)
    mock_predictive.forecast_service_health.return_value = [
        ForecastResult(
            metric="latency",
            predicted_value=100,
            confidence=0.1,
            trend="stable",
            risk_level="low",
        )
    ]
    mock_predictive.service_history = {}  # empty history -> sparsity high

    loop = GovernanceLoop(
        policy_evaluator=mock_policy_evaluator,
        cost_estimator=mock_cost_estimator,
        risk_engine=mock_risk_engine,
        enable_epistemic=True,
        hallucination_probe=mock_hall,
        predictive_engine=mock_predictive,
    )
    context = {
        "query": "test",
        "evidence": "test",
        "entropy": 1.0,
        "evidence_lift": 1.0,
        "contradiction": 1.0,
        "service_name": "test-service",
    }
    intent = loop.run(sample_intent, context=context)
    # Epistemic uncertainty should be high, triggering ESCALATE
    assert intent.action == RecommendedAction.ESCALATE.value


def test_governance_loop_bayesian_decision(
    mock_policy_evaluator, mock_cost_estimator, mock_risk_engine, sample_intent
):
    """Test that Bayesian expected loss chooses action based on costs."""
    # Set risk low, epistemic low, business impact high
    mock_risk_engine.calculate_risk.return_value = (
        0.1, "Low risk",
        {"conjugate_alpha": 1.5, "conjugate_beta": 12.0}
    )
    loop = GovernanceLoop(
        policy_evaluator=mock_policy_evaluator,
        cost_estimator=mock_cost_estimator,
        risk_engine=mock_risk_engine,
        enable_epistemic=False,
    )
    context = {
        "estimated_value": 1000.0,  # high opportunity
        "throughput": 10000,
        "error_rate": 0.1,
        "latency_p99": 200,
    }
    intent = loop.run(sample_intent, context=context)
    # With low risk, high opportunity, approve expected loss should be lowest
    assert intent.action == RecommendedAction.APPROVE.value


def test_governance_loop_variance_escalation(
    mock_policy_evaluator, mock_cost_estimator, mock_risk_engine, sample_intent
):
    """Test that high variance (uncertainty) leads to ESCALATE even without epistemic gate."""
    # Set risk moderate but variance high (Beta(1,1) has variance 1/12 ≈ 0.0833)
    mock_risk_engine.calculate_risk.return_value = (
        0.5, "High uncertainty risk",
        {"conjugate_alpha": 1.0, "conjugate_beta": 1.0}
    )
    loop = GovernanceLoop(
        policy_evaluator=mock_policy_evaluator,
        cost_estimator=mock_cost_estimator,
        risk_engine=mock_risk_engine,
        enable_epistemic=False,
    )
    context = {
        "estimated_value": 0.0,  # no opportunity
        "throughput": 10000,
        "error_rate": 0.1,
        "latency_p99": 200,
    }
    intent = loop.run(sample_intent, context=context)
    # With high variance, approve loss should be high due to COST_VARIANCE,
    # and with no opportunity, escalate may be chosen
    # Check that the decision is either ESCALATE or DENY (not APPROVE)
    assert intent.action in [RecommendedAction.ESCALATE.value, RecommendedAction.DENY.value]


def test_governance_loop_batch(
    sample_intent, mock_policy_evaluator, mock_cost_estimator, mock_risk_engine
):
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
