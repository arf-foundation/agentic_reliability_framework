"""Basic smoke tests for ARF."""

import agentic_reliability_framework as arf

def test_version():
    """Check that the package version is set correctly."""
    assert arf.__version__ == "4.2.0+oss"

def test_imports():
    """Verify that key modules import without errors."""
    from agentic_reliability_framework.core.models.healing_intent import HealingIntent
    from agentic_reliability_framework.runtime.engine import EnhancedReliabilityEngine
    from agentic_reliability_framework.runtime.memory.rag_graph import RAGGraphMemory
    assert True
