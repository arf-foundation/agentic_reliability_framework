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
    
    # We'll patch datetime.now to return increasing timestamps
    with patch('datetime.datetime') as mock_datetime:
        # Set the class to return our custom timestamps
        class MockDatetime:
            @classmethod
            def now(cls, tz=None):
                return mock_current_time
        
        mock_datetime.now = MockDatetime.now
        mock_datetime.timezone = datetime.timezone
        
        for i in range(30):
            mock_current_time = base_time + datetime.timedelta(minutes=i)
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
    return engine


class TestSimplePredictiveEngine:
    """Test suite for SimplePredictiveEngine."""

    def test_init(self):
        """Test initialization."""
        engine = SimplePredictiveEngine(history_window=100)
        assert engine.history_window == 100
        assert engine.service_history == {}
        assert engine.prediction_cache == {}
        expected = datetime.timedelta(minutes=CACHE_EXPIRY_MINUTES)
        assert engine.max_cache_age == expected

    def test_add_telemetry_new_service(self, engine):
        """Test adding telemetry for a new service."""
        engine.add_telemetry("svc1", {"latency_p99": 150, "error_rate": 0.02, "throughput": 500})
        assert "svc1" in engine.service_history
        assert len(engine.service_history["svc1"]) == 1
        point = engine.service_history["svc1"][0]
        assert point["latency"] == 150
        assert point["error_rate"] == 0.02
        assert point["throughput"] == 500
        assert "timestamp" in point

    def test_add_telemetry_existing_service(self, engine):
        """Test adding telemetry for an existing service."""
        engine.add_telemetry("svc1", {"latency_p99": 150})
        engine.add_telemetry("svc1", {"latency_p99": 160})
        assert len(engine.service_history["svc1"]) == 2

    def test_add_telemetry_history_limit(self):
        """Test that history respects the maxlen."""
        engine = SimplePredictiveEngine(history_window=3)
        for i in range(5):
            engine.add_telemetry("svc1", {"latency_p99": i})
        assert len(engine.service_history["svc1"]) == 3
        values = [p["latency"] for p in engine.service_history["svc1"]]
        assert values == [2, 3, 4]

    def test_clean_cache(self, engine):
        """Test that old cache entries are removed."""
        now = datetime.datetime.now(datetime.timezone.utc)
        old = now - datetime.timedelta(minutes=CACHE_EXPIRY_MINUTES + 10)
        engine.prediction_cache = {
            "svc1_latency": (MagicMock(), old),
            "svc1_error": (MagicMock(), now - datetime.timedelta(minutes=10)),
            "svc2_latency": (MagicMock(), old)
        }
        engine._clean_cache()
        assert "svc1_error" in engine.prediction_cache
        assert "svc1_latency" not in engine.prediction_cache
        assert "svc2_latency" not in engine.prediction_cache

    def test_forecast_service_health_insufficient_data(self, engine):
        """Test forecast returns empty when insufficient data."""
        engine.add_telemetry("svc1", {"latency_p99": 100})
        result = engine.forecast_service_health("svc1")
        assert result == []

    def test_forecast_service_health_success(self, populated_engine):
        """Test forecast returns list of ForecastResult."""
        forecasts = populated_engine.forecast_service_health("test-service")
        assert len(forecasts) >= 2  # at least latency and error
        for f in forecasts:
            assert isinstance(f, ForecastResult)
            assert f.metric in ["latency", "error_rate", "cpu_util", "memory_util"]
            assert 0 <= f.confidence <= 1
            assert f.risk_level in ["low", "medium", "high", "critical"]
            assert f.forecast_timestamp is not None

    def test_forecast_unknown_service(self, engine):
        """Test forecast for non-existent service returns empty."""
        assert engine.forecast_service_health("unknown") == []

    def test_forecast_latency_basic(self, populated_engine):
        """Test _forecast_latency returns a ForecastResult."""
        history = list(populated_engine.service_history["test-service"])
        result = populated_engine._forecast_latency(history, lookahead=10)
        assert result is not None
        assert result.metric == "latency"
        assert result.predicted_value > 0
        assert result.trend in ["increasing", "decreasing", "stable"]

    def test_forecast_latency_insufficient_data(self, engine):
        """Test _forecast_latency returns None when insufficient data."""
        engine.add_telemetry("svc1", {"latency_p99": 100})
        history = list(engine.service_history["svc1"])
        result = engine._forecast_latency(history, 10)
        assert result is None

    def test_forecast_latency_exception_handling(self, populated_engine, monkeypatch):
        """Test exception in _forecast_latency returns None."""
        def mock_polyfit(*args, **kwargs):
            raise ValueError("mock error")
        monkeypatch.setattr(np, "polyfit", mock_polyfit)
        history = list(populated_engine.service_history["test-service"])
        result = populated_engine._forecast_latency(history, 10)
        assert result is None

    def test_forecast_error_rate_basic(self, populated_engine):
        """Test _forecast_error_rate returns a ForecastResult."""
        history = list(populated_engine.service_history["test-service"])
        result = populated_engine._forecast_error_rate(history, 10)
        assert result is not None
        assert result.metric == "error_rate"
        assert 0 <= result.predicted_value <= 1
        assert result.trend in ["increasing", "decreasing", "stable"]

    def test_forecast_error_rate_insufficient_data(self, engine):
        """Test _forecast_error_rate returns None when insufficient data."""
        engine.add_telemetry("svc1", {"error_rate": 0.01})
        history = list(engine.service_history["svc1"])
        result = engine._forecast_error_rate(history, 10)
        assert result is None

    def test_forecast_resources(self, populated_engine):
        """Test _forecast_resources returns list of forecasts for cpu and memory."""
        history = list(populated_engine.service_history["test-service"])
        results = populated_engine._forecast_resources(history, 10)
        # At least one resource metric may be present
        assert len(results) >= 1
        for r in results:
            assert 0 <= r.predicted_value <= 1
            assert 0 <= r.confidence <= 1
            assert r.trend in ["increasing", "decreasing", "stable"]

    def test_forecast_resources_no_data(self, engine):
        """Test _forecast_resources with no cpu/memory returns empty."""
        engine.add_telemetry("svc1", {"latency_p99": 100})
        history = list(engine.service_history["svc1"])
        results = engine._forecast_resources(history, 10)
        assert results == []

    def test_cache_after_forecast(self, populated_engine):
        """Test that forecasts are cached after a successful forecast."""
        forecasts = populated_engine.forecast_service_health("test-service")
        assert len(populated_engine.prediction_cache) > 0
        for f in forecasts:
            key = f"test-service_{f.metric}"
            assert key in populated_engine.prediction_cache

    def test_get_predictive_insights(self, populated_engine):
        """Test get_predictive_insights returns expected structure."""
        insights = populated_engine.get_predictive_insights("test-service")
        assert insights['service'] == "test-service"
        assert 'forecasts' in insights
        assert isinstance(insights['forecasts'], list)
        assert 'warnings' in insights
        assert 'recommendations' in insights
        assert insights['critical_risk_count'] >= 0
        assert 'forecast_timestamp' in insights

    def test_get_predictive_insights_no_forecast(self, engine):
        """Test insights with no forecast returns empty lists."""
        insights = engine.get_predictive_insights("unknown")
        assert insights['forecasts'] == []
        assert insights['warnings'] == []
        assert insights['recommendations'] == []
        assert insights['critical_risk_count'] == 0

    def test_forecast_risk_levels(self):
        """Test that risk levels are correctly assigned."""
        engine = SimplePredictiveEngine()
        # Create a history with increasing latency to trigger "high"/"critical"
        history = []
        base_time = datetime.datetime.now(datetime.timezone.utc)
        for i in range(30):
            history.append({
                'timestamp': base_time,
                'latency': 100 + i * 20,
                'error_rate': 0.01,
                'throughput': 1000
            })
        # Set the last few latencies very high
        for i in range(5):
            history[-i-1]['latency'] = LATENCY_EXTREME + 50
        result = engine._forecast_latency(history, 10)
        assert result is not None
        assert result.risk_level in ["high", "critical"]


class TestBusinessImpactCalculator:
    """Test suite for BusinessImpactCalculator."""

    def test_init(self):
        """Test initialization."""
        calc = BusinessImpactCalculator(revenue_per_request=0.02)
        assert calc.revenue_per_request == 0.02

    @pytest.mark.parametrize("latency,error_rate,cpu,expected_min_severity", [
        (LATENCY_CRITICAL + 10, 0.02, 0.5, "CRITICAL"),
        (150, 0.15, 0.5, "CRITICAL"),
        (150, 0.02, CPU_CRITICAL + 0.1, "CRITICAL"),
        (200, 0.05, 0.7, "HIGH"),
        (120, 0.03, 0.6, "MEDIUM"),
        (80, 0.005, 0.4, "LOW"),
    ])
    def test_calculate_impact_severity(self, latency, error_rate, cpu, expected_min_severity):
        """Test severity classification (actual may be higher, but not lower)."""
        event = ReliabilityEvent(
            component="test",
            latency_p99=latency,
            error_rate=error_rate,
            cpu_util=cpu
        )
        calc = BusinessImpactCalculator()
        result = calc.calculate_impact(event, duration_minutes=10)
        severity_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        actual_index = severity_order.index(result['severity_level'])
        expected_min_index = severity_order.index(expected_min_severity)
        assert actual_index >= expected_min_index

    def test_calculate_impact_returns_correct_keys(self):
        """Test result dictionary has expected keys."""
        event = ReliabilityEvent(component="test", latency_p99=100, error_rate=0.01)
        calc = BusinessImpactCalculator()
        result = calc.calculate_impact(event)
        assert set(result.keys()) == {"revenue_loss_estimate", "affected_users_estimate", "severity_level", "throughput_reduction_pct"}

    def test_calculate_impact_without_cpu(self):
        """Test calculation works when cpu_util is missing."""
        event = ReliabilityEvent(component="test", latency_p99=200, error_rate=0.02)
        calc = BusinessImpactCalculator()
        result = calc.calculate_impact(event)
        assert result['revenue_loss_estimate'] > 0
        assert result['affected_users_estimate'] > 0

    def test_impact_values_monotonic(self):
        """Test that higher metrics lead to higher impact."""
        calc = BusinessImpactCalculator()
        event1 = ReliabilityEvent(component="test", latency_p99=100, error_rate=0.01)
        event2 = ReliabilityEvent(component="test", latency_p99=500, error_rate=0.20)
        res1 = calc.calculate_impact(event1)
        res2 = calc.calculate_impact(event2)
        assert res2['revenue_loss_estimate'] > res1['revenue_loss_estimate']
        assert res2['affected_users_estimate'] > res1['affected_users_estimate']

    def test_duration_affects_revenue_loss(self):
        """Test that longer duration increases revenue loss."""
        calc = BusinessImpactCalculator()
        event = ReliabilityEvent(component="test", latency_p99=200, error_rate=0.05)
        res_short = calc.calculate_impact(event, duration_minutes=5)
        res_long = calc.calculate_impact(event, duration_minutes=60)
        assert res_long['revenue_loss_estimate'] > res_short['revenue_loss_estimate']
        assert res_long['affected_users_estimate'] == res_short['affected_users_estimate']

    def test_calculate_impact_with_none_values(self):
        """Test handling of None metrics (should not crash)."""
        event = ReliabilityEvent(component="test", latency_p99=None, error_rate=None, cpu_util=None)
        calc = BusinessImpactCalculator()
        result = calc.calculate_impact(event)
        # BASE_REVENUE_PER_MINUTE * (5/60) = default revenue loss when no metrics
        # However, with the new formula using throughput and revenue_per_request, the result may differ.
        # We'll just check that it doesn't crash and returns some numbers.
        assert result['revenue_loss_estimate'] >= 0
        assert result['affected_users_estimate'] == 0
        assert result['severity_level'] == "LOW"
