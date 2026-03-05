"""
Data models for memory graph.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


class EdgeType(str, Enum):
    RESOLVED_BY = "RESOLVED_BY"
    SIMILAR_TO = "SIMILAR_TO"


class EventSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class IncidentNode:
    incident_id: str
    component: str
    severity: str
    timestamp: str
    metrics: Dict[str, float]
    agent_analysis: Dict[str, Any]
    embedding_id: Optional[int] = None
    faiss_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutcomeNode:
    outcome_id: str
    incident_id: str
    actions_taken: List[str]
    resolution_time_minutes: float
    success: bool
    lessons_learned: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    edge_id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimilarityResult:
    incident_node: IncidentNode
    similarity_score: float
    raw_score: float
    faiss_index: int
