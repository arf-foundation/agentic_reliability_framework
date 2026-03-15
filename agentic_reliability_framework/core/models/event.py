import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from enum import Enum
from pydantic import BaseModel

# ==================== ENUMS ====================
# New canonical Severity
class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Backward compatibility: old EventSeverity
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

# ==================== FORECAST RESULT ====================
class ForecastResult(BaseModel):
    """Result of a forecasting analysis."""
    metric: str
    predicted_value: float
    confidence: float
    trend: str
    time_to_threshold: Optional[float] = None
    risk_level: str

# ==================== VALIDATION FUNCTION ====================
def validate_component_id(component: str) -> Tuple[bool, str]:
    """Validate a component identifier (backward compatibility)."""
    if not component or not component.strip():
        return False, "Component ID cannot be empty"
    if len(component) > 64:
        return False, "Component ID too long (max 64 chars)"
    return True, ""

# ==================== RELIABILITY EVENT ====================
@dataclass
class ReliabilityEvent:
    """
    Canonical reliability event with full backward compatibility.
    
    All canonical fields are optional and will be auto‑filled if missing.
    Old fields (component, latency_p99, etc.) are accepted and mapped
    into the new structure.
    """
    # Canonical fields (all optional with defaults)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    service_name: str = "unknown"
    event_type: str = "unknown"
    severity: Severity = Severity.LOW
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Old fields (accepted for compatibility)
    component: Optional[str] = None
    latency_p99: Optional[float] = None
    error_rate: Optional[float] = None
    throughput: Optional[int] = None
    cpu_util: Optional[float] = None
    memory_util: Optional[float] = None
    source: Optional[str] = None

    def __post_init__(self):
        # Convert severity if it's an EventSeverity
        if isinstance(self.severity, EventSeverity):
            # Map EventSeverity to Severity
            mapping = {
                EventSeverity.INFO: Severity.LOW,
                EventSeverity.WARNING: Severity.MEDIUM,
                EventSeverity.HIGH: Severity.HIGH,
                EventSeverity.ERROR: Severity.HIGH,
                EventSeverity.CRITICAL: Severity.CRITICAL,
            }
            self.severity = mapping.get(self.severity, Severity.LOW)

        # If component is provided but service_name is not, set service_name
        if self.component and self.service_name == "unknown":
            self.service_name = self.component
        # If service_name is provided but component is not, set component
        if self.service_name != "unknown" and self.component is None:
            self.component = self.service_name

        # Ensure both are set (if neither, raise error)
        if self.service_name == "unknown" and self.component is None:
            raise ValueError("Either component or service_name must be provided")

        # Map old metric fields into metrics dict (only if they are not None)
        if self.latency_p99 is not None:
            self.metrics['latency_p99'] = self.latency_p99
        if self.error_rate is not None:
            self.metrics['error_rate'] = self.error_rate
        if self.throughput is not None:
            self.metrics['throughput'] = float(self.throughput)
        if self.cpu_util is not None:
            self.metrics['cpu_util'] = self.cpu_util
        if self.memory_util is not None:
            self.metrics['memory_util'] = self.memory_util

        # Map source into metadata
        if self.source is not None:
            self.metadata['source'] = self.source

    # Property accessors for old fields (so code that uses them directly still works)
    @property
    def latency_p99_prop(self) -> Optional[float]:
        return self.metrics.get('latency_p99')

    @property
    def error_rate_prop(self) -> Optional[float]:
        return self.metrics.get('error_rate')

    @property
    def throughput_prop(self) -> Optional[int]:
        val = self.metrics.get('throughput')
        return int(val) if val is not None else None

    @property
    def cpu_util_prop(self) -> Optional[float]:
        return self.metrics.get('cpu_util')

    @property
    def memory_util_prop(self) -> Optional[float]:
        return self.metrics.get('memory_util')

    @property
    def source_prop(self) -> Optional[str]:
        return self.metadata.get('source')
