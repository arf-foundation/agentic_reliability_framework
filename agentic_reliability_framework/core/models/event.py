import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

# ==================== ENUMS ====================
class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EventSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
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

# ==================== FORECAST RESULT ====================
class ForecastResult(BaseModel):
    metric: str
    predicted_value: float
    confidence: float
    trend: str
    time_to_threshold: Optional[float] = None
    risk_level: str

# ==================== VALIDATION FUNCTION ====================
def validate_component_id(component: str) -> Tuple[bool, str]:
    if not component or not component.strip():
        return False, "Component ID cannot be empty"
    if len(component) > 64:
        return False, "Component ID too long (max 64 chars)"
    return True, ""

# ==================== RELIABILITY EVENT ====================
class ReliabilityEvent(BaseModel):
    """
    Canonical reliability event with full backward compatibility.
    Frozen to ensure immutability, as expected by the engine.
    """
    # Canonical fields (all optional with defaults)
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    service_name: str = "unknown"
    event_type: str = "unknown"
    severity: Severity = Severity.LOW
    metrics: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Legacy fields (accepted for compatibility, but not stored directly)
    # They are handled by the pre‑validator and converted into canonical fields.
    component: Optional[str] = Field(None, exclude=True)
    latency_p99: Optional[float] = Field(None, exclude=True)
    error_rate: Optional[float] = Field(None, exclude=True)
    throughput: Optional[int] = Field(None, exclude=True)
    cpu_util: Optional[float] = Field(None, exclude=True)
    memory_util: Optional[float] = Field(None, exclude=True)
    source: Optional[str] = Field(None, exclude=True)

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True
    )

    @field_validator('severity', mode='before')
    @classmethod
    def convert_severity(cls, v):
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

    @model_validator(mode='before')
    @classmethod
    def transform_legacy_fields(cls, data: Any) -> Any:
        """Pre‑validator: transform legacy input into canonical fields."""
        if not isinstance(data, dict):
            return data

        # Handle component → service_name
        if data.get('component') and not data.get('service_name'):
            data['service_name'] = data['component']

        # Populate metrics from legacy numeric fields, skipping None
        metrics = data.get('metrics', {}).copy()
        if 'latency_p99' in data:
            val = data.pop('latency_p99')
            if val is not None:
                metrics['latency_p99'] = val
        if 'error_rate' in data:
            val = data.pop('error_rate')
            if val is not None:
                metrics['error_rate'] = val
        if 'throughput' in data:
            val = data.pop('throughput')
            if val is not None:
                metrics['throughput'] = float(val)
        if 'cpu_util' in data:
            val = data.pop('cpu_util')
            if val is not None:
                metrics['cpu_util'] = val
        if 'memory_util' in data:
            val = data.pop('memory_util')
            if val is not None:
                metrics['memory_util'] = val
        if metrics:
            data['metrics'] = metrics

        # Populate metadata from legacy source
        if 'source' in data:
            source = data.pop('source')
            if source is not None:
                existing_metadata = data.get('metadata', {})
                data['metadata'] = {**existing_metadata, 'source': source}

        return data

    # ===== BACKWARD COMPATIBILITY ALIASES =====
    # These expose legacy attributes expected by the runtime system.

    @property
    def component(self) -> str:
        return self.service_name

    @property
    def latency_p99(self) -> Optional[float]:
        return self.metrics.get("latency_p99")

    @property
    def error_rate(self) -> Optional[float]:
        return self.metrics.get("error_rate")

    @property
    def throughput(self) -> Optional[int]:
        val = self.metrics.get("throughput")
        return int(val) if val is not None else None

    @property
    def cpu_util(self) -> Optional[float]:
        return self.metrics.get("cpu_util")

    @property
    def memory_util(self) -> Optional[float]:
        return self.metrics.get("memory_util")

    # (Optional) Keep the property accessors for clarity
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
