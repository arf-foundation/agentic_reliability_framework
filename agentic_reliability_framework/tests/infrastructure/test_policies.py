import pytest
from agentic_reliability_framework.core.governance.policies import (
    RegionAllowedPolicy,
    ResourceTypeRestrictedPolicy,
    MaxPermissionLevelPolicy,
    CostThresholdPolicy,
    PolicyEvaluator,
    allow_all,
    deny_all,
)
from agentic_reliability_framework.core.governance.intents import (
    ProvisionResourceIntent,
    GrantAccessIntent,
    ResourceType,
    PermissionLevel,
)


def test_policy_evaluator_region_restriction():
    policy = RegionAllowedPolicy(allowed_regions={"eastus"})
    evaluator = PolicyEvaluator(policy)
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="westus",
        size="Standard_D2s_v3",
        requester="alice",
        environment="prod"
    )
    violations = evaluator.evaluate(intent)
    assert len(violations) == 1
    assert "westus" in violations[0]


def test_policy_evaluator_restricted_resource_type():
    policy = ResourceTypeRestrictedPolicy(forbidden_types={ResourceType.KUBERNETES_CLUSTER})
    evaluator = PolicyEvaluator(policy)
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.KUBERNETES_CLUSTER,
        region="eastus",
        size="Medium",
        requester="alice",
        environment="prod"
    )
    violations = evaluator.evaluate(intent)
    assert len(violations) == 1
    assert "kubernetes_cluster" in violations[0]


def test_policy_evaluator_permission_limit():
    policy = MaxPermissionLevelPolicy(max_level=PermissionLevel.WRITE)
    evaluator = PolicyEvaluator(policy)
    intent = GrantAccessIntent(
        principal="bob",
        permission_level=PermissionLevel.ADMIN,
        resource_scope="/subscriptions/123",
        requester="alice"
    )
    violations = evaluator.evaluate(intent)
    assert len(violations) == 1
    assert "admin" in violations[0]

    intent_ok = GrantAccessIntent(
        principal="bob",
        permission_level=PermissionLevel.WRITE,
        resource_scope="/subscriptions/123",
        requester="alice"
    )
    violations_ok = evaluator.evaluate(intent_ok)
    assert len(violations_ok) == 0


def test_cost_threshold_policy():
    policy = CostThresholdPolicy(max_cost_usd=100.0)
    evaluator = PolicyEvaluator(policy)
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D8s_v3",
        requester="alice",
        environment="prod"
    )
    violations = evaluator.evaluate(intent, context={"cost_estimate": 150.0})
    assert len(violations) == 1
    assert "exceeds threshold" in violations[0]

    violations_ok = evaluator.evaluate(intent, context={"cost_estimate": 50.0})
    assert len(violations_ok) == 0


def test_allow_all_policy():
    policy = allow_all()
    evaluator = PolicyEvaluator(policy)
    # use a valid region/size so intent validation passes
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
<<<<<<< HEAD
        region="eastus",
        size="Standard_D2s_v3",
=======
        region="eastus",                 # valid region
        size="Standard_D2s_v3",           # valid size
>>>>>>> adf837024fd6d06c8d3dd61a120b662cc49a2c77
        requester="alice",
        environment="prod"
    )
    violations = evaluator.evaluate(intent)
    assert len(violations) == 0


def test_deny_all_policy():
    policy = deny_all()
    evaluator = PolicyEvaluator(policy)
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D2s_v3",
        requester="alice",
        environment="prod"
    )
    violations = evaluator.evaluate(intent)
    assert len(violations) == 1
    assert "denied" in violations[0]
