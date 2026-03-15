from typing import List, Tuple
from ..models.event import ReliabilityEvent
from ..models.outcome import IncidentOutcome

class MemoryInterface:
    def store_event(self, event: ReliabilityEvent) -> None:
        """Persist event for future similarity search."""
        raise NotImplementedError

    def query_similar(self, event: ReliabilityEvent, k: int = 5) -> List[Tuple[ReliabilityEvent, float]]:
        """
        Return k most similar events along with similarity scores (0..1).
        """
        raise NotImplementedError

    def update_outcome(self, outcome: IncidentOutcome) -> None:
        """Record outcome for an event (used to compute memory‑derived risk)."""
        raise NotImplementedError
