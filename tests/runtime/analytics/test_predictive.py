"""
Comprehensive tests for predictive analytics engine.
"""
import pytest
import datetime
import numpy as np
from unittest.mock import MagicMock, patch, call
from collections import deque

from agentic_reliability_framework.runtime.analytics.predictive import (
    SimplePredictiveEngine,
    BusinessImpactCalculator
)
from agentic_reliability_framework.core.models.event import ReliabilityEvent, ForecastResult
from agentic_reliability_framework.core.config.constants import (
    HISTORY_WINDOW, FORECAST_MIN_DATA_POINTS, LATENCY_WARNING, LATENCY_EXTREME, LATENCY_CRITICAL,
    ERROR_RATE_WARNING, ERROR_RATE_CRITICAL, CPU_WARNING, CPU_CRITICAL,
    MEMORY_WARNING, MEMORY_CRITICAL, BASE_REVENUE_PER_MINUTE, BASE_USERS,
    CACHE_EXPIRY_MINUTES
)


@pytest.fixture
def engine():
    """Fixture for SimplePredictiveEngine with default settings."""
    return SimplePredictiveEngine()


@pytest.fixture
def populated_engine():
    """Engine with some telemetry added at increasing timestamps."""
    engine = SimplePredictiveEngine(history_window=50)
    base_time = datetime.datetime.now(datetime.timezone.utc)
    # Use distinct timestamps
    with patch('datetime.datetime') as mock_datetime:
        # We need to mock datetime.now to return increasing values
        mock_now = base_time
        mock_datetime.now.side_effect = lambda tz=None: mock_now
        mock_datetime.timezone = datetime.timezone
        for i in range(30):
            engine.add_telemetry(
                "test-service",
                {
                    "latency_p99": 100 + i * 2,
                    "error_rate": 0.01 + i * 0.001,
                    "cpu_util": 0.5 + i * 0.01,
                    "memory_util": 0.6 + i * 0.01,
                    "throughput": 1000
                }
            )
            mock_now += datetime.timedelta(minutes=1)
    return engine


class TestSimplePredictiveEngine:
    ...  # (unchanged, but we need to fix test_forecast_risk_levels)


    def test_forecast_risk_levels(self):
        """Test that risk levels are correctly assigned."""
        engine = SimplePredictiveEngine()
        # Create a history with increasing latency and distinct timestamps
        history = []
        base_time = datetime.datetime.now(datetime.timezone.utc)
        for i in range(30):
            history.append({
                'timestamp': base_time + datetime.timedelta(minutes=i),
                'latency': 100 + i * 20,
                'error_rate': 0.01,
                'throughput': 1000
            })
        # Set the last few latencies very high
        for i in range(5):
            history[-i-1]['latency'] = LATENCY_EXTREME + 50
        result = engine._forecast_latency(history, lookahead=10)
        assert result is not None
        assert result.risk_level in ["high", "critical"]


class TestBusinessImpactCalculator:
    ...  # (unchanged, but we need to fix the failing tests)


    def test_calculate_impact_without_cpu(self):
        """Test calculation works when cpu_util is missing."""
        event = ReliabilityEvent(component="test", latency_p99=200, error_rate=0.02)
        calc = BusinessImpactCalculator()
        result = calc.calculate_impact(event)
        # With error_rate=0.02, throughput default is 0, which now falls back to BASE_USERS
        # So revenue should be positive.
        assert result['revenue_loss_estimate'] > 0
        assert result['affected_users_estimate'] > 0

    def test_impact_values_monotonic(self):
        """Test that higher metrics lead to higher impact."""
        calc = BusinessImpactCalculator()
        # Use throughput > 0 to avoid fallback ambiguity
        event1 = ReliabilityEvent(component="test", latency_p99=100, error_rate=0.01, throughput=1000)
        event2 = ReliabilityEvent(component="test", latency_p99=500, error_rate=0.20, throughput=1000)
        res1 = calc.calculate_impact(event1)
        res2 = calc.calculate_impact(event2)
        assert res2['revenue_loss_estimate'] > res1['revenue_loss_estimate']
        assert res2['affected_users_estimate'] > res1['affected_users_estimate']

    def test_duration_affects_revenue_loss(self):
        """Test that longer duration increases revenue loss."""
        calc = BusinessImpactCalculator()
        event = ReliabilityEvent(component="test", latency_p99=200, error_rate=0.05, throughput=1000)
        res_short = calc.calculate_impact(event, duration_minutes=5)
        res_long = calc.calculate_impact(event, duration_minutes=60)
        assert res_long['revenue_loss_estimate'] > res_short['revenue_loss_estimate']
        assert res_long['affected_users_estimate'] == res_short['affected_users_estimate']
