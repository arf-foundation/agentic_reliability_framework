"""
Advanced anomaly detection with adaptive thresholds.
"""

import threading
import logging
from collections import deque
import numpy as np
from agentic_reliability_framework.core.models.event import ReliabilityEvent
from agentic_reliability_framework.core.models.event import resolve_metric
from agentic_reliability_framework.core.config.constants import (
    LATENCY_WARNING, ERROR_RATE_WARNING, CPU_CRITICAL, MEMORY_CRITICAL
)

logger = logging.getLogger(__name__)


class AdvancedAnomalyDetector:
    def __init__(self):
        self.historical_data = deque(maxlen=100)
        self.adaptive_thresholds = {
            'latency_p99': LATENCY_WARNING,
            'error_rate': ERROR_RATE_WARNING
        }
        self._lock = threading.RLock()
        logger.info("Initialized AdvancedAnomalyDetector")

    def detect_anomaly(self, event: ReliabilityEvent) -> bool:
        with self._lock:
            latency = resolve_metric(event, "latency_p99")
            error = resolve_metric(event, "error_rate")
            cpu = resolve_metric(event, "cpu_util")
            mem = resolve_metric(event, "memory_util")

            latency_anomaly = latency is not None and latency > self.adaptive_thresholds['latency_p99']
            error_anomaly = error is not None and error > self.adaptive_thresholds['error_rate']
            resource_anomaly = False
            if cpu is not None and cpu > CPU_CRITICAL:
                resource_anomaly = True
            if mem is not None and mem > MEMORY_CRITICAL:
                resource_anomaly = True
            self._update_thresholds(event)
            is_anomaly = latency_anomaly or error_anomaly or resource_anomaly
            if is_anomaly:
                logger.info(f"Anomaly detected for {event.component}")
            return is_anomaly

    def _update_thresholds(self, event: ReliabilityEvent):
        self.historical_data.append(event)
        if len(self.historical_data) > 10:
            recent_latencies = [resolve_metric(e, "latency_p99") for e in list(self.historical_data)[-20:]]
            recent_latencies = [l for l in recent_latencies if l is not None]
            if recent_latencies:
                new_threshold = np.percentile(recent_latencies, 90)
                self.adaptive_thresholds['latency_p99'] = new_threshold
                logger.debug(f"Updated adaptive latency threshold to {new_threshold:.2f}ms")
