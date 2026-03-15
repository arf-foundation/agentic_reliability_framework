import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator

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

# ==================== RELIABILITY EVENT (Pydantic Model) ====================
class ReliabilityEvent(BaseModel):
    """
    Canonical reliability event with full backward compatibility.
    
    All canonical fields are optional and will be auto‑filled if missing.
    Old fields (component, latency_p99, etc.) are accepted and automatically
    mapped into the new structure. The model supports `.model_copy()` for
    immutability, as expected by the engine.
    """
    # Canonical fields (all optional with defaults)
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    service_name: str = "unknown"
    event_type: str = "unknown"
    severity: Severity = Severity.LOW
    metrics: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Legacy fields (accepted for compatibility, will be mapped)
    component: Optional[str] = None
    latency_p99: Optional[float] = None
    error_rate: Optional[float] = None
    throughput: Optional[int] = None
    cpu_util: Optional[float] = None
    memory_util: Optional[float] = None
    source: Optional[str] = None

    @field_validator('severity', mode='before')
    @classmethod
    def convert_severity(cls, v):
        """Allow EventSeverity values and convert them to canonical Severity."""
        if isinstance(v, EventSeverity):
            mapping = {
                EventSeverity.INFO: Severity.LOW,
                EventSeverity.WARNING: Severity.MEDIUM,
                EventSeverity.HIGH: Severity.HIGH,
                EventSeverity.ERROR: Severity.HIGH,
                EventSeverity.CRITICAL: Severity.CRITICAL,
            }
            return mapping.get(v, Severity.LOW)
        return v

    @model_validator(mode='after')
    def sync_fields(self):
        # Ensure component and service_name are consistent
        if self.component and self.service_name == "unknown":
            self.service_name = self.component
        if self.service_name != "unknown" and self.component is None:
            self.component = self.service_name
        if self.service_name == "unknown" and self.component is None:
            raise ValueError("Either component or service_name must be provided")

        # Populate metrics from legacy fields
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
        if self.source is not None:
            self.metadata['source'] = self.source

        return self

    # Property accessors for legacy fields (optional, for convenience)
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

    class Config:
        # Allow arbitrary types (like enums) in the model
        arbitrary_types_allowed = True
        # Keep the model frozen to match dataclass immutability expectations
        frozen = True
