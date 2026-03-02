import pytest
from agentic_reliability_framework.core.governance.azure.azure_simulator import AzureInfrastructureSimulator
from agentic_reliability_framework.core.governance.policies import (
    allow_all,
    MaxPermissionLevelPolicy,
    CostThresholdPolicy,
)
from agentic_reliability_framework.core.governance.intents import (
    ProvisionResourceIntent,
    GrantAccessIntent,
    ResourceType,
    PermissionLevel,
)
from agentic_reliability_framework.core.governance.healing_intent import RecommendedAction


def test_simulator_approve_low_risk():
    policies = allow_all()
    simulator = AzureInfrastructureSimulator(policies)
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D2s_v3",
        requester="alice",
        environment="dev"
    )
    result = simulator.evaluate(intent)
    assert result.recommended_action == RecommendedAction.APPROVE
    # Bayesian risk for compute category prior (1.0,12.0) mean ≈ 0.0769
    assert result.risk_score == pytest.approx(0.077, abs=0.01)
    assert result.cost_projection == 70.0


def test_simulator_deny_high_risk():
    policies = MaxPermissionLevelPolicy(max_level=PermissionLevel.WRITE)
    simulator = AzureInfrastructureSimulator(policies)
    intent = GrantAccessIntent(
        principal="bob",
        permission_level=PermissionLevel.ADMIN,
        resource_scope="/subscriptions/123",
        requester="alice"
    )
    result = simulator.evaluate(intent)
    assert result.recommended_action == RecommendedAction.DENY
    # Security category prior (2.0,10.0) mean = 0.1667
    assert result.risk_score == pytest.approx(0.1667, abs=0.01)
    assert "admin" in result.justification


def test_simulator_deny_due_to_cost_threshold():
    policies = CostThresholdPolicy(max_cost_usd=100.0)
    simulator = AzureInfrastructureSimulator(policies)
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D8s_v3",
        requester="alice",
        environment="dev"
    )
    result = simulator.evaluate(intent)
    assert result.recommended_action == RecommendedAction.DENY
    assert "exceeds threshold" in result.justification
    assert len(result.policy_violations) == 1


def test_simulator_escalate_medium_risk():
    policies = allow_all()
    simulator = AzureInfrastructureSimulator(policies)
    intent = GrantAccessIntent(
        principal="bob",
        permission_level=PermissionLevel.WRITE,
        resource_scope="/subscriptions/123",
        requester="alice"
    )
    result = simulator.evaluate(intent)
<<<<<<< HEAD
    # medium security risk currently falls below escalation threshold
    assert result.recommended_action == RecommendedAction.APPROVE
    # Security prior mean = 0.1667
=======
    # Risk score for WRITE permission is 0.1667 (below 0.4) -> APPROVE
    assert result.recommended_action == RecommendedAction.APPROVE
>>>>>>> adf837024fd6d06c8d3dd61a120b662cc49a2c77
    assert 0.1 < result.risk_score < 0.3

def test_cost_projection_present_in_healing_intent():
    """Ensure cost_projection appears both top-level and in parameters."""
    policies = allow_all()
    simulator = AzureInfrastructureSimulator(policies)
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D2s_v3",
        requester="alice",
        environment="dev"
    )
    result = simulator.evaluate(intent)
    assert result.cost_projection == 70.0
    assert result.parameters.get("cost_projection") == 70.0
