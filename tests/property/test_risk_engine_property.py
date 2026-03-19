"""Property‑based tests for the risk engine invariants."""
import pytest
from hypothesis import given, strategies as st

from agentic_reliability_framework.core.governance.risk_engine import RiskEngine
from agentic_reliability_framework.core.governance.intents import (
    ProvisionResourceIntent,
    ResourceType,
    Environment,
)

# Strategy to generate a valid provisioning intent
@st.composite
def provision_intent(draw):
    resource_type = draw(st.sampled_from(list(ResourceType)))
    region = draw(st.sampled_from(["eastus", "westus", "westeurope"]))
    size = draw(st.sampled_from(["Standard_D2s_v3", "Standard_D4s_v3"]))
    environment = draw(st.sampled_from(["dev", "prod"]))
    return ProvisionResourceIntent(
        resource_type=resource_type,
        region=region,
        size=size,
        requester="test",
        environment=environment,
    )

@given(intent=provision_intent())
def test_risk_engine_risk_in_0_1(intent):
    """Risk score must always be between 0 and 1."""
    engine = RiskEngine()
    risk, _, _ = engine.calculate_risk(intent, cost_estimate=100.0, policy_violations=[])
    assert 0.0 <= risk <= 1.0

@given(intent=provision_intent(), success=st.booleans())
def test_risk_engine_update_invariants(intent, success):
    """After update, risk remains in [0,1] and total_incidents increments."""
    engine = RiskEngine()
    before = engine.total_incidents
    engine.update_outcome(intent, success)
    after = engine.total_incidents
    assert after == before + 1
    risk, _, _ = engine.calculate_risk(intent, None, [])
    assert 0.0 <= risk <= 1.0

@given(intent=provision_intent())
def test_risk_engine_weights_sum_to_1(intent):
    """Weights returned in contributions should sum to 1 (approx)."""
    engine = RiskEngine()
    _, _, contrib = engine.calculate_risk(intent, None, [])
    weights = contrib["weights"]
    total = weights.get("conjugate", 0) + weights.get("hyper", 0) + weights.get("hmc", 0)
    assert abs(total - 1.0) < 1e-6
