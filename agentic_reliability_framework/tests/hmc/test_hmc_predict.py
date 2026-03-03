"""
Tests for HMCRiskLearner.predict() method.

Validates that the predict method:
- Returns a float in [0, 1]
- Returns 0.5 if model not trained
- Correctly integrates with risk_engine.py
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from agentic_reliability_framework.runtime.hmc.hmc_learner import HMCRiskLearner


def test_predict_returns_float():
    """Test that predict() returns a float."""
    learner = HMCRiskLearner()
    # Mock is_ready to True and posterior_predictive to return samples
    learner.is_ready = True
    learner.posterior_predictive = MagicMock(return_value=np.array([0.45, 0.55, 0.50]))

    risk = learner.predict({
        'latency_p99': 350,
        'error_rate': 0.12,
        'throughput': 900,
    })

    assert isinstance(risk, float)
    assert 0.0 <= risk <= 1.0


def test_predict_not_ready_returns_default():
    """Test that predict() returns 0.5 if model not trained."""
    learner = HMCRiskLearner()
    assert learner.is_ready is False

    risk = learner.predict({
        'latency_p99': 350,
        'error_rate': 0.12,
    })

    assert risk == 0.5


def test_predict_bounds_check():
    """Test that predict() clamps result to [0, 1]."""
    learner = HMCRiskLearner()
    learner.is_ready = True

    # Test clamping high value
    learner.posterior_predictive = MagicMock(return_value=np.array([1.5, 1.2, 1.1]))
    risk_high = learner.predict({'latency_p99': 500})
    assert risk_high <= 1.0

    # Test clamping low value
    learner.posterior_predictive = MagicMock(return_value=np.array([-0.5, -0.3, -0.1]))
    risk_low = learner.predict({'latency_p99': 100})
    assert risk_low >= 0.0


def test_predict_exception_handling():
    """Test that predict() returns 0.5 on exception."""
    learner = HMCRiskLearner()
    learner.is_ready = True
    learner.posterior_predictive = MagicMock(side_effect=RuntimeError("Test error"))

    risk = learner.predict({'latency_p99': 350})

    assert risk == 0.5


def test_predict_with_missing_metrics():
    """Test that predict() handles missing metrics gracefully."""
    learner = HMCRiskLearner()
    learner.is_ready = True
    # posterior_predictive should handle missing keys by defaulting to 0
    learner.posterior_predictive = MagicMock(return_value=np.array([0.4]))

    # Call with partial metrics
    risk = learner.predict({
        'latency_p99': 350,
        # Missing: error_rate, throughput
    })

    assert 0.0 <= risk <= 1.0


def test_predict_with_all_metrics():
    """Test predict() with all metrics provided."""
    learner = HMCRiskLearner()
    learner.is_ready = True
    learner.feature_names = ['latency_p99', 'error_rate', 'throughput', 'cpu_util', 'memory_util']
    learner._feature_scales = {
        'latency_p99': (250, 100),
        'error_rate': (0.05, 0.1),
        'throughput': (1000, 200),
        'cpu_util': (0.6, 0.2),
        'memory_util': (0.5, 0.25),
    }
    learner.posterior_predictive = MagicMock(return_value=np.array([0.62, 0.65, 0.60]))

    risk = learner.predict({
        'latency_p99': 350,
        'error_rate': 0.15,
        'throughput': 800,
        'cpu_util': 0.85,
        'memory_util': 0.75,
    })

    assert isinstance(risk, float)
    assert 0.0 <= risk <= 1.0
    assert 0.60 <= risk <= 0.65  # Should be mean of [0.62, 0.65, 0.60]


def test_predict_called_from_risk_engine():
    """Test that predict() can be called by risk_engine.py."""
    learner = HMCRiskLearner()
    learner.is_ready = True
    learner.posterior_predictive = MagicMock(return_value=np.array([0.55]))

    # Simulate risk_engine calling predict
    metrics = {
        'latency_p99': 320,
        'error_rate': 0.12,
        'throughput': 850,
        'cpu_util': 0.80,
        'memory_util': 0.60,
    }

    hmc_risk = learner.predict(metrics)

    # Verify it's a valid risk score
    assert isinstance(hmc_risk, (int, float))
    assert 0.0 <= hmc_risk <= 1.0


def test_predict_comparison_with_posterior_predictive():
    """Test that predict() uses posterior_predictive correctly."""
    learner = HMCRiskLearner()
    learner.is_ready = True

    # Mock posterior_predictive with known samples
    samples = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    learner.posterior_predictive = MagicMock(return_value=samples)

    risk = learner.predict({'latency_p99': 350})

    # Should return the mean of samples
    expected_mean = float(np.mean(samples))
    # Allow small floating-point error
    assert abs(risk - expected_mean) < 1e-6


def test_predict_deterministic_for_trained_model():
    """Test that predict() is deterministic for a given input."""
    learner = HMCRiskLearner()
    learner.is_ready = True

    # Setup fixed samples
    fixed_samples = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    learner.posterior_predictive = MagicMock(return_value=fixed_samples)

    metrics = {
        'latency_p99': 300,
        'error_rate': 0.10,
    }

    risk1 = learner.predict(metrics)
    risk2 = learner.predict(metrics)

    # Should get exact same result
    assert risk1 == risk2 == 0.5


def test_predict_zero_and_one_boundaries():
    """Test predict() behavior at risk score boundaries."""
    learner = HMCRiskLearner()
    learner.is_ready = True

    # Test very low risk
    learner.posterior_predictive = MagicMock(return_value=np.array([0.01]))
    risk_low = learner.predict({'latency_p99': 50})
    assert risk_low == 0.01

    # Test very high risk
    learner.posterior_predictive = MagicMock(return_value=np.array([0.99]))
    risk_high = learner.predict({'latency_p99': 500, 'error_rate': 0.5})
    assert risk_high == 0.99
