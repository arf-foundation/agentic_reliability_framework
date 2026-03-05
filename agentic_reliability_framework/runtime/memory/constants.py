"""
Memory-specific constants (aligned with OSS limits).
"""
from agentic_reliability_framework.core.config.constants import (
    MAX_INCIDENT_NODES, MAX_OUTCOME_NODES, EMBEDDING_DIM,
    SIMILARITY_THRESHOLD, GRAPH_CACHE_SIZE
)


class MemoryConstants:
    """Memory constants – values are enforced by OSS limits."""
    FAISS_BATCH_SIZE = 10
    FAISS_SAVE_INTERVAL_SECONDS = 30
    VECTOR_DIM = EMBEDDING_DIM
    MAX_INCIDENT_NODES = MAX_INCIDENT_NODES
    MAX_OUTCOME_NODES = MAX_OUTCOME_NODES
    GRAPH_CACHE_SIZE = GRAPH_CACHE_SIZE
    SIMILARITY_THRESHOLD = SIMILARITY_THRESHOLD
