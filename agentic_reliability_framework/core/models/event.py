from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from enum import Enum
from pydantic import BaseModel

# ==================== ENUMS ====================
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
    Canonical reliability event.
    
    For backward compatibility, this class also accepts the old fields
    (component, latency_p99, error_rate, throughput, cpu_util, memory_util, source)
    and stores them appropriately.
    
    - `component` is an alias for `service_name` (they will be kept in sync).
    - Other numeric fields are placed into the `metrics` dictionary.
    - `source` is stored in `metadata`.
    """
    # Canonical fields
    id: str
    timestamp: datetime
    service_name: str                # primary component identifier
    event_type: str
    severity: Severity
    metrics: Dict[str, float]         # e.g., latency, error_rate, cpu
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Backward compatibility fields (optional, set automatically)
    component: Optional[str] = None   # alias for service_name
    latency_p99: Optional[float] = None
    error_rate: Optional[float] = None
    throughput: Optional[int] = None
    cpu_util: Optional[float] = None
    memory_util: Optional[float] = None
    source: Optional[str] = None

    def __post_init__(self):
        # Ensure component and service_name are consistent
        if self.component is None and self.service_name:
            self.component = self.service_name
        if self.service_name is None and self.component:
            self.service_name = self.component
        if self.component is None and self.service_name is None:
            raise ValueError("Either component or service_name must be provided")

        # Store legacy numeric fields into metrics (if provided)
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

        # Store source in metadata if provided
        if self.source is not None:
            self.metadata['source'] = self.source

    # For easier access to the legacy fields as properties
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
