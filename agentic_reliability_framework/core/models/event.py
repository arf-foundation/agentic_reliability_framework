"""Base reliability event model for ARF."""
import uuid
from datetime import datetime
from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field


class EventSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HealingAction(str, Enum):
    NO_ACTION = "no_action"
    RESTART_CONTAINER = "restart_container"
    SCALE_OUT = "scale_out"
    ROLLBACK = "rollback"
    CIRCUIT_BREAKER = "circuit_breaker"
    TRAFFIC_SHIFT = "traffic_shift"
    ALERT_TEAM = "alert_team"


class ReliabilityEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    component: str
    service_mesh: str = "default"
    latency_p99: float
    error_rate: float
    throughput: int = 0
    cpu_util: Optional[float] = None
    memory_util: Optional[float] = None
    source: str = "system"
    severity: EventSeverity = EventSeverity.INFO

    class Config:
        use_enum_values = True


def validate_component_id(component: str) -> tuple[bool, str]:
    if not component or not component.strip():
        return False, "Component ID cannot be empty"
    if len(component) > 64:
        return False, "Component ID too long (max 64 chars)"
    return True, ""
