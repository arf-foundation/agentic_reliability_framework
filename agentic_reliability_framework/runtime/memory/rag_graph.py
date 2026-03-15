"""
RAG Graph Memory – stores incidents, outcomes, and provides similarity search.
Adapted for v4 with OSS limits.
"""
import hashlib
import logging
import threading
import time
import numpy as np
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager

from .models import (
    IncidentNode, OutcomeNode, GraphEdge,
    SimilarityResult, EdgeType, EventSeverity
)
from .constants import MemoryConstants
from .enhanced_faiss import EnhancedFAISSIndex
from agentic_reliability_framework.core.config.constants import (
    MAX_INCIDENT_NODES, MAX_OUTCOME_NODES, GRAPH_CACHE_SIZE, SIMILARITY_THRESHOLD
)

logger = logging.getLogger(__name__)


class RAGGraphMemory:
    """Enhanced RAG graph with thread safety and OSS limits."""

    def __init__(self, faiss_index):
        self.enhanced_faiss = EnhancedFAISSIndex(faiss_index)
        self.faiss = faiss_index
        self.incident_nodes: Dict[str, IncidentNode] = {}
        self.outcome_nodes: Dict[str, OutcomeNode] = {}
        self.edges: List[GraphEdge] = []
        self._lock = threading.RLock()
        self._stats = {
            "total_incidents_stored": 0,
            "total_outcomes_stored": 0,
            "total_edges_created": 0,
            "similarity_searches": 0,
            "cache_hits": 0,
            "failed_searches": 0,
            "last_search_time": None,
            "last_store_time": None,
        }
        self._rag_failures = 0
        self._rag_disabled_until = 0.0
        self._rag_last_failure_time = 0.0
        self._similarity_cache: OrderedDict[str, List[SimilarityResult]] = OrderedDict()
        self._max_cache_size = GRAPH_CACHE_SIZE
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._max_embedding_cache_size = 100
        self._faiss_to_incident: Dict[int, str] = {}
        logger.info(f"Initialized RAGGraphMemory (OSS) with max incidents {MAX_INCIDENT_NODES}")

    @contextmanager
    def _transaction(self):
        with self._lock:
            yield

    def is_enabled(self) -> bool:
        # RAG is considered available even if empty (cold start allowed)
        return True

    def has_historical_data(self) -> bool:
        return len(self.incident_nodes) > 0

    def _generate_incident_id(self, event) -> str:
        """Deterministic ID based on event fingerprint."""
        from agentic_reliability_framework.core.models.event import ReliabilityEvent
        if not isinstance(event, ReliabilityEvent):
            return f"inc_{hashlib.sha256(str(event).encode()).hexdigest()[:16]}"
        # Use defaults to avoid None issues
        lat = event.latency_p99 or 0.0
        err = event.error_rate or 0.0
        data = f"{event.component}:{lat:.2f}:{err:.4f}"
        return f"inc_{hashlib.sha256(data.encode()).hexdigest()[:16]}"

    def _embed_incident(self, event, analysis: Dict[str, Any]) -> np.ndarray:
        """Create embedding vector (simplified for OSS)."""
        # In a real implementation, use sentence-transformers.
        # For OSS, we use a deterministic random based on event data.
        import hashlib
        # Use defaults to avoid None issues
        lat = event.latency_p99 or 0
        err = event.error_rate or 0
        seed_str = f"{event.component}:{lat}:{err}"
        seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
        np.random.seed(seed)
        embedding = np.random.randn(MemoryConstants.VECTOR_DIM).astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def store_incident(self, event, analysis: Dict[str, Any]) -> str:
        if not self.is_enabled():
            return ""
        incident_id = self._generate_incident_id(event)
        with self._transaction():
            if incident_id in self.incident_nodes:
                node = self.incident_nodes[incident_id]
                node.agent_analysis = analysis
                node.metadata["last_updated"] = datetime.now().isoformat()
                return incident_id

            embedding = self._embed_incident(event, analysis)
            faiss_id = self.faiss.add(embedding)
            self._faiss_to_incident[faiss_id] = incident_id

            node = IncidentNode(
                incident_id=incident_id,
                component=event.component,
                severity=event.severity.value if hasattr(event.severity, 'value') else "low",
                timestamp=event.timestamp.isoformat(),
                metrics={
                    "latency_ms": event.latency_p99,
                    "error_rate": event.error_rate,
                    "throughput": event.throughput,
                    "cpu_util": event.cpu_util or 0.0,
                    "memory_util": event.memory_util or 0.0,
                },
                agent_analysis=analysis,
                embedding_id=faiss_id,
                faiss_index=faiss_id,
                metadata={"created_at": datetime.now().isoformat()}
            )
            self.incident_nodes[incident_id] = node
            self._stats["total_incidents_stored"] += 1
            self._stats["last_store_time"] = datetime.now().isoformat()

            # OSS limit eviction
            if len(self.incident_nodes) > MAX_INCIDENT_NODES:
                oldest = min(self.incident_nodes.keys(), key=lambda x: self.incident_nodes[x].metadata.get("created_at", ""))
                self._faiss_to_incident.pop(self.incident_nodes[oldest].faiss_index, None)
                del self.incident_nodes[oldest]
        return incident_id

    def find_similar(self, event, analysis: Dict[str, Any], k: int = 5) -> List[IncidentNode]:
        if not self.has_historical_data():
            return []
        embedding = self._embed_incident(event, analysis)
        cache_key = hashlib.sha256(embedding.tobytes()).hexdigest()
        with self._transaction():
            if cache_key in self._similarity_cache:
                self._stats["cache_hits"] += 1
                return [res.incident_node for res in self._similarity_cache[cache_key]]

        distances, indices = self.enhanced_faiss.search(embedding, k)
        results = []
        for dist, idx in zip(distances, indices):
            if idx == -1:
                continue
            inc_id = self._faiss_to_incident.get(int(idx))
            if inc_id and inc_id in self.incident_nodes:
                node = self.incident_nodes[inc_id]
                node.metadata["similarity_score"] = 1.0 / (1.0 + dist)
                results.append(node)

        with self._transaction():
            self._stats["similarity_searches"] += 1
            self._similarity_cache[cache_key] = [SimilarityResult(node, 0.0, 0.0, 0) for node in results]  # simplified
            if len(self._similarity_cache) > self._max_cache_size:
                self._similarity_cache.popitem(last=False)
        return results

    def store_outcome(self, incident_id: str, actions_taken: List[str],
                      success: bool, resolution_time_minutes: float,
                      lessons_learned: Optional[List[str]] = None) -> str:
        if incident_id not in self.incident_nodes:
            return ""
        outcome_id = f"out_{hashlib.sha256(f'{incident_id}{actions_taken}'.encode()).hexdigest()[:16]}"
        with self._transaction():
            outcome = OutcomeNode(
                outcome_id=outcome_id,
                incident_id=incident_id,
                actions_taken=actions_taken,
                resolution_time_minutes=resolution_time_minutes,
                success=success,
                lessons_learned=lessons_learned or [],
                metadata={"created_at": datetime.now().isoformat()}
            )
            self.outcome_nodes[outcome_id] = outcome
            self._stats["total_outcomes_stored"] += 1
            # Edge
            edge = GraphEdge(
                edge_id=f"edge_{hashlib.md5(f'{incident_id}{outcome_id}'.encode(), usedforsecurity=False).hexdigest()[:16]}",
                source_id=incident_id,
                target_id=outcome_id,
                edge_type=EdgeType.RESOLVED_BY,
                metadata={}
            )
            self.edges.append(edge)
            self._stats["total_edges_created"] += 1

            if len(self.outcome_nodes) > MAX_OUTCOME_NODES:
                oldest = min(self.outcome_nodes.keys(), key=lambda x: self.outcome_nodes[x].metadata.get("created_at", ""))
                del self.outcome_nodes[oldest]
        return outcome_id

    def get_historical_effectiveness(self, action: str, component_filter: Optional[str] = None) -> Dict[str, Any]:
        relevant = []
        for outcome in self.outcome_nodes.values():
            if action in outcome.actions_taken:
                incident = self.incident_nodes.get(outcome.incident_id)
                if incident and (component_filter is None or incident.component == component_filter):
                    relevant.append(outcome)
        total = len(relevant)
        successful = sum(1 for o in relevant if o.success)
        resolution_times = [o.resolution_time_minutes for o in relevant if o.success]
        avg = np.mean(resolution_times) if resolution_times else 0.0
        std = np.std(resolution_times) if resolution_times else 0.0
        return {
            "action": action,
            "total_uses": total,
            "successful_uses": successful,
            "success_rate": successful / total if total else 0.0,
            "avg_resolution_time_minutes": float(avg),
            "resolution_time_std": float(std),
            "component_filter": component_filter,
            "data_points": total,
        }

    def get_most_effective_actions(self, component: str, k: int = 3) -> List[Dict[str, Any]]:
        action_stats = {}
        for outcome in self.outcome_nodes.values():
            incident = self.incident_nodes.get(outcome.incident_id)
            if incident and incident.component == component:
                for action in outcome.actions_taken:
                    s = action_stats.setdefault(action, {"total": 0, "successful": 0})
                    s["total"] += 1
                    if outcome.success:
                        s["successful"] += 1
        results = []
        for action, stats in action_stats.items():
            if stats["total"] >= 3:  # require minimum data
                results.append({
                    "action": action,
                    "success_rate": stats["successful"] / stats["total"],
                    "total_uses": stats["total"],
                    "successful_uses": stats["successful"],
                })
        results.sort(key=lambda x: x["success_rate"], reverse=True)
        return results[:k]

    def get_graph_stats(self) -> Dict[str, Any]:
        component_dist = {}
        for node in self.incident_nodes.values():
            component_dist[node.component] = component_dist.get(node.component, 0) + 1
        return {
            "incident_nodes": len(self.incident_nodes),
            "outcome_nodes": len(self.outcome_nodes),
            "edges": len(self.edges),
            "similarity_cache_size": len(self._similarity_cache),
            "embedding_cache_size": len(self._embedding_cache),
            "cache_hit_rate": self._stats["cache_hits"] / (self._stats["similarity_searches"] or 1),
            "incidents_with_outcomes": len(set(o.incident_id for o in self.outcome_nodes.values())),
            "avg_outcomes_per_incident": len(self.outcome_nodes) / (len(self.incident_nodes) or 1),
            "component_distribution": component_dist,
            "stats": self._stats,
            "memory_limits": {
                "max_incident_nodes": MAX_INCIDENT_NODES,
                "max_outcome_nodes": MAX_OUTCOME_NODES,
                "graph_cache_size": self._max_cache_size,
            },
            "is_operational": self.has_historical_data(),
        }
