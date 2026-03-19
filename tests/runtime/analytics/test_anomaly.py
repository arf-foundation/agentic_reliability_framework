"""Tests for anomaly detection module."""
import pytest
from agentic_reliability_framework.runtime.analytics.anomaly import AdvancedAnomalyDetector
from agentic_reliability_framework.core.models.event import ReliabilityEvent, EventSeverity
from agentic_reliability_framework.core.config.constants import (
    LATENCY_WARNING,
    ERROR_RATE_WARNING,
    CPU_CRITICAL,
    MEMORY_CRITICAL,
)


@pytest.fixture
def detector():
    """Create a fresh anomaly detector for each test."""
    return AdvancedAnomalyDetector()


def test_detect_anomaly_normal(detector):
    """Test that normal metrics are not flagged as anomaly."""
    event = ReliabilityEvent(
        component="test",
        latency_p99=50.0,  # below warning
        error_rate=0.01,    # below warning
        throughput=1000,
        cpu_util=0.3,       # below critical
        memory_util=0.4,    # below critical
        severity=EventSeverity.INFO,
    )
    result = detector.detect_anomaly(event)
    assert result is False


def test_detect_anomaly_high_latency(detector):
    """Test high latency triggers anomaly."""
    event = ReliabilityEvent(
        component="test",
        latency_p99=500.0,  # above warning
        error_rate=0.01,
        throughput=1000,
        severity=EventSeverity.INFO,
    )
    result = detector.detect_anomaly(event)
    assert result is True


def test_detect_anomaly_high_error_rate(detector):
    """Test high error rate triggers anomaly."""
    event = ReliabilityEvent(
        component="test",
        latency_p99=50.0,
        error_rate=0.5,  # above warning
        throughput=1000,
        severity=EventSeverity.INFO,
    )
    result = detector.detect_anomaly(event)
    assert result is True


def test_detect_anomaly_high_cpu(detector):
    """Test high CPU triggers anomaly."""
    event = ReliabilityEvent(
        component="test",
        latency_p99=50.0,
        error_rate=0.01,
        throughput=1000,
        cpu_util=0.95,  # above critical
        severity=EventSeverity.INFO,
    )
    result = detector.detect_anomaly(event)
    assert result is True


def test_detect_anomaly_high_memory(detector):
    """Test high memory triggers anomaly."""
    event = ReliabilityEvent(
        component="test",
        latency_p99=50.0,
        error_rate=0.01,
        throughput=1000,
        memory_util=0.95,  # above critical
        severity=EventSeverity.INFO,
    )
    result = detector.detect_anomaly(event)
    assert result is True


def test_adaptive_threshold_update(detector):
    """Test that thresholds adapt over time."""
    # Create a series of events with increasing latency
    for i in range(20):
        event = ReliabilityEvent(
            component="test",
            latency_p99=100.0 + i * 5,
            error_rate=0.01,
            throughput=1000,
            severity=EventSeverity.INFO,
        )
        detector.detect_anomaly(event)  # triggers update

    # After many updates, the adaptive threshold should have increased
    assert detector.adaptive_thresholds['latency_p99'] > LATENCY_WARNING


def test_detect_anomaly_missing_metrics(detector):
    """Test that missing metrics are handled gracefully (no crash)."""
    # cpu_util and memory_util are optional; event without them should be fine.
    event = ReliabilityEvent(
        component="test",
        latency_p99=50.0,
        error_rate=0.01,
        throughput=1000,
        severity=EventSeverity.INFO,
    )
    result = detector.detect_anomaly(event)
    assert result is False
