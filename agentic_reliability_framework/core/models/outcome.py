from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class IncidentOutcome:
    event_id: str
    resolved: bool                     # True if mitigated successfully
    resolution_strategy: str           # e.g., "auto_scale", "rollback", "manual"
    resolution_time_seconds: float     # time to resolve
    failure_classification: str         # e.g., "transient", "permanent", "unknown"
    outcome_score: float                # 0.0 (worst) to 1.0 (best)
    timestamp: datetime
