"""Stress tests for memory eviction and concurrency."""
import pytest
import threading
import concurrent.futures
from agentic_reliability_framework.runtime.memory import RAGGraphMemory
from agentic_reliability_framework.runtime.memory.faiss_index import create_faiss_index
from agentic_reliability_framework.core.models.event import ReliabilityEvent, EventSeverity
from agentic_reliability_framework.core.config.constants import MAX_INCIDENT_NODES

@pytest.mark.stress
def test_memory_exceeds_limit():
    """Fill memory beyond MAX_INCIDENT_NODES and verify eviction works."""
    index = create_faiss_index(dim=384)
    memory = RAGGraphMemory(index)
    for i in range(MAX_INCIDENT_NODES + 100):
        event = ReliabilityEvent(
            component=f"comp{i}",
            latency_p99=100.0,
            error_rate=0.01,
            severity=EventSeverity.INFO
        )
        memory.store_incident(event, {})
    assert len(memory.incident_nodes) <= MAX_INCIDENT_NODES

@pytest.mark.stress
def test_concurrent_memory_operations():
    """Run many concurrent store/find operations and check for errors."""
    index = create_faiss_index(dim=384)
    memory = RAGGraphMemory(index)

    def worker(i):
        event = ReliabilityEvent(
            component=f"comp{i}",
            latency_p99=100.0,
            error_rate=0.01,
            severity=EventSeverity.INFO
        )
        analysis = {}
        inc_id = memory.store_incident(event, analysis)
        similar = memory.find_similar(event, analysis, k=3)
        return inc_id, len(similar)

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(worker, i) for i in range(200)]
        for future in concurrent.futures.as_completed(futures):
            inc_id, count = future.result()
            assert inc_id.startswith("inc_")
            assert count >= 0
