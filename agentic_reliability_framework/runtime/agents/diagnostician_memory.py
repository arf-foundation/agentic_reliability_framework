"""
Memory Drift Diagnostician Agent – detects drift in semantic memory using z‑score analysis.
"""
import logging
import numpy as np
from typing import Dict, Any, List, Optional

from agentic_reliability_framework.runtime.agents.base import BaseAgent, AgentSpecialization
from agentic_reliability_framework.core.models.event import ReliabilityEvent

logger = logging.getLogger(__name__)


class MemoryDriftDiagnosticianAgent(BaseAgent):
    """
    Detects drift in semantic memory by comparing current retrieval scores
    with their historical distribution using a z‑score test.
    """

    def __init__(self, history_window: int = 100, zscore_threshold: float = 2.0):
        """
        Args:
            history_window: Number of past retrieval scores to keep for baseline.
            zscore_threshold: Absolute z‑score above which drift is flagged.
        """
        super().__init__(AgentSpecialization.DIAGNOSTICIAN)
        self.history_window = history_window
        self.zscore_threshold = zscore_threshold
        self._retrieval_scores_history: List[float] = []

    async def analyze(self, event: ReliabilityEvent, context_window: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze drift using retrieval scores from the event.

        Args:
            event: ReliabilityEvent (expects 'retrieval_scores' attribute if available).
            context_window: Optional override of history window size.

        Returns:
            Dictionary with drift detection findings.
        """
        try:
            # Extract retrieval scores (if present in event; use an empty list as fallback)
            retrieval_scores = getattr(event, 'retrieval_scores', [])
            if not retrieval_scores:
                return {
                    'specialization': 'memory_drift',
                    'confidence': 0.0,
                    'findings': {},
                    'recommendations': []
                }

            current_avg = float(np.mean(retrieval_scores))
            self._retrieval_scores_history.append(current_avg)

            # Use provided context_window if given, else default
            window = context_window if context_window is not None else self.history_window
            if len(self._retrieval_scores_history) > window:
                self._retrieval_scores_history.pop(0)

            if len(self._retrieval_scores_history) < 10:
                return {
                    'specialization': 'memory_drift',
                    'confidence': 0.0,
                    'findings': {
                        'drift_detected': False,
                        'current_avg': current_avg,
                        'historical_avg': None,
                        'z_score': None
                    },
                    'recommendations': []
                }

            historical_avg = float(np.mean(self._retrieval_scores_history[:-1]))
            historical_std = float(np.std(self._retrieval_scores_history[:-1])) + 1e-6
            z_score = (current_avg - historical_avg) / historical_std
            drift_detected = abs(z_score) > self.zscore_threshold
            confidence = min(1.0, abs(z_score) / 5.0)

            return {
                'specialization': 'memory_drift',
                'confidence': confidence,
                'findings': {
                    'drift_detected': drift_detected,
                    'current_avg': current_avg,
                    'historical_avg': historical_avg,
                    'z_score': float(z_score)
                },
                'recommendations': [
                    "Reindex knowledge base",
                    "Adjust embedding model",
                    "Update context window"
                ] if drift_detected else []
            }
        except Exception as e:
            logger.error(f"MemoryDriftDiagnostician error: {e}", exc_info=True)
            return {
                'specialization': 'memory_drift',
                'confidence': 0.0,
                'findings': {},
                'recommendations': []
            }
