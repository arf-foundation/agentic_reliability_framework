"""Tests for the AdvancedAnomalyDetector."""

import pytest
from unittest.mock import Mock, patch

from agentic_reliability_framework.runtime.analytics.anomaly import AdvancedAnomalyDetector
from agentic_reliability_framework.core.models.event import ReliabilityEvent
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


@pytest.fixture
def normal_event():
    """Create an event with normal metrics."""
    return ReliabilityEvent(
        component="test-service",
        latency_p99=100.0,
        error_rate=0.01,
        cpu_util=0.5,
        memory_util=0.5,
    )


@pytest.fixture
def high_latency_event():
    """Create an event with high latency."""
    return ReliabilityEvent(
        component="test-service",
        latency_p99=500.0,  # > LATENCY_WARNING (150.0)
        error_rate=0.01,
        cpu_util=0.5,
        memory_util=0.5,
    )


@pytest.fixture
def high_error_event():
    """Create an event with high error rate."""
    return ReliabilityEvent(
        component="test-service",
        latency_p99=100.0,
        error_rate=0.2,  # > ERROR_RATE_WARNING (0.05)
        cpu_util=0.5,
        memory_util=0.5,
    )


@pytest.fixture
def high_cpu_event():
    """Create an event with high CPU usage."""
    return ReliabilityEvent(
        component="test-service",
        latency_p99=100.0,
        error_rate=0.01,
        cpu_util=0.95,  # > CPU_CRITICAL (0.9)
        memory_util=0.5,
    )


@pytest.fixture
def high_memory_event():
    """Create an event with high memory usage."""
    return ReliabilityEvent(
        component="test-service",
        latency_p99=100.0,
        error_rate=0.01,
        cpu_util=0.5,
        memory_util=0.95,  # > MEMORY_CRITICAL (0.9)
    )


def test_detect_anomaly_normal(detector, normal_event):
    """Test that normal events are not flagged as anomalies."""
    is_anomaly = detector.detect_anomaly(normal_event)
    assert is_anomaly is False


def test_detect_anomaly_high_latency(detector, high_latency_event):
    """Test that high latency is detected as anomaly."""
    is_anomaly = detector.detect_anomaly(high_latency_event)
    assert is_anomaly is True


def test_detect_anomaly_high_error(detector, high_error_event):
    """Test that high error rate is detected as anomaly."""
    is_anomaly = detector.detect_anomaly(high_error_event)
    assert is_anomaly is True


def test_detect_anomaly_high_cpu(detector, high_cpu_event):
    """Test that high CPU is detected as anomaly."""
    is_anomaly = detector.detect_anomaly(high_cpu_event)
    assert is_anomaly is True


def test_detect_anomaly_high_memory(detector, high_memory_event):
    """Test that high memory is detected as anomaly."""
    is_anomaly = detector.detect_anomaly(high_memory_event)
    assert is_anomaly is True


def test_detect_anomaly_missing_metrics(detector):
    """Test that missing metrics are handled gracefully (should not crash)."""
    # Create event with None for some metrics
    event = ReliabilityEvent(
        component="test-service",
        latency_p99=None,
        error_rate=None,
        cpu_util=None,
        memory_util=None,
    )
    # Should not raise exception
    is_anomaly = detector.detect_anomaly(event)
    # Should be false because all checks default to None -> not > threshold
    assert is_anomaly is False


def test_threshold_updates(detector, normal_event, high_latency_event):
    """Test that thresholds update after enough events."""
    # Initial thresholds should be defaults
    assert detector.adaptive_thresholds['latency_p99'] == LATENCY_WARNING

    # Add 11 normal events to trigger update
    for _ in range(11):
        detector.detect_anomaly(normal_event)

    # After 11 events, thresholds should have been updated
    # The update uses percentiles, so it should be something around 100
    assert detector.adaptive_thresholds['latency_p99'] != LATENCY_WARNING
    assert 80 < detector.adaptive_thresholds['latency_p99'] < 120

    # Add many high latency events to push threshold up
    for _ in range(10):
        detector.detect_anomaly(high_latency_event)

    # Threshold should increase
    assert detector.adaptive_thresholds['latency_p99'] > 120


def test_thread_safety(detector, normal_event):
    """Simple thread-safety test (not exhaustive but checks that it doesn't crash)."""
    import threading

    results = []
    errors = []

    def run_detection():
        try:
            results.append(detector.detect_anomaly(normal_event))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=run_detection) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert len(results) == 10
