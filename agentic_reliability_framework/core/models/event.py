from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum

# New canonical Severity (for ReliabilityEvent)
class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Backward compatibility: old EventSeverity (used in tests and other modules)
class EventSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    ERROR = "error"
    CRITICAL = "critical"

# Backward compatibility: old HealingAction
class HealingAction(str, Enum):
    NO_ACTION = "no_action"
    RESTART_CONTAINER = "restart_container"
    SCALE_OUT = "scale_out"
    ROLLBACK = "rollback"
    CIRCUIT_BREAKER = "circuit_breaker"
    TRAFFIC_SHIFT = "traffic_shift"
    ALERT_TEAM = "alert_team"

@dataclass
class ReliabilityEvent:
    id: str
    timestamp: datetime
    service_name: str
    event_type: str
    severity: Severity
    metrics: Dict[str, float]          # e.g., latency, error_rate, cpu
    metadata: Dict[str, Any] = field(default_factory=dict)
