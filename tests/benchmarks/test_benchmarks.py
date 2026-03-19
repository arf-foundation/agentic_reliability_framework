"""Performance benchmarks for key components."""
import pytest
from agentic_reliability_framework.core.governance.risk_engine import RiskEngine
from agentic_reliability_framework.core.governance.governance_loop import GovernanceLoop
from agentic_reliability_framework.core.governance.policies import PolicyEvaluator, allow_all
from agentic_reliability_framework.core.governance.cost_estimator import CostEstimator
from agentic_reliability_framework.core.governance.intents import (
    ProvisionResourceIntent,
    ResourceType,
)
from agentic_reliability_framework.runtime.memory import RAGGraphMemory
from agentic_reliability_framework.runtime.memory.faiss_index import create_faiss_index
from agentic_reliability_framework.core.models.event import ReliabilityEvent, EventSeverity

@pytest.fixture
def sample_intent():
    return ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D2s_v3",
        requester="benchmark",
        environment="dev",  # string literal
    )

def test_risk_engine_benchmark(benchmark, sample_intent):
    engine = RiskEngine()
    # Run calculate_risk 1000 times (benchmark will handle repetitions)
    benchmark(engine.calculate_risk, sample_intent, None, [])

def test_memory_find_similar_benchmark(benchmark):
    index = create_faiss_index(dim=384)
    memory = RAGGraphMemory(index)
    # Pre‑populate with 100 incidents
    for i in range(100):
        event = ReliabilityEvent(
            component=f"comp{i}",
            latency_p99=100.0,
            error_rate=0.01,
            severity=EventSeverity.INFO
        )
        memory.store_incident(event, {})
    event = ReliabilityEvent(
        component="test",
        latency_p99=100.0,
        error_rate=0.01,
        severity=EventSeverity.INFO
    )
    benchmark(memory.find_similar, event, {}, k=5)

@pytest.fixture
def governance_loop():
    # Use a trivial policy that allows everything
    policy = PolicyEvaluator(root_policy=allow_all())
    cost = CostEstimator()
    risk = RiskEngine()
    index = create_faiss_index(dim=384)
    memory = RAGGraphMemory(index)
    return GovernanceLoop(policy, cost, risk, memory)

def test_governance_loop_benchmark(benchmark, governance_loop, sample_intent):
    benchmark(governance_loop.run, sample_intent, {})
