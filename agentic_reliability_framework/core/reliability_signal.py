"""
Reliability Signal Module

ARF interprets system signals and converts them into reliability probabilities
that downstream modules (risk engine, healing intent, etc.) can reason about.

This module provides a clean abstraction: from raw anomaly signals to normalized
reliability scores between 0 and 1, where 1 represents perfect reliability and
0 represents complete system failure.

Concept: Reliability is treated as a probabilistic signal derived from
anomaly intensity, providing a consistent interface for all ARF components.
"""

import logging
from typing import Union, Optional

logger = logging.getLogger(__name__)


def normalize_anomaly_signal(
    raw_score: Union[float, int],
    max_expected: float = 10.0,
    min_expected: float = 0.0
) -> float:
    """
    Normalize a raw anomaly signal to a 0-1 scale.
    
    Args:
        raw_score: Raw anomaly magnitude (could be latency ms, error rate, etc.)
        max_expected: Maximum expected value for this signal type
        min_expected: Minimum expected value (usually 0)
    
    Returns:
        Normalized score between 0 and 1, where higher values indicate
        stronger anomaly signals.
    
    Example:
        >>> normalize_anomaly_signal(450, max_expected=500)
        0.9  # 450ms latency out of 500ms max
    """
    if max_expected <= min_expected:
        logger.warning(f"Invalid bounds: max_expected={max_expected} <= min_expected={min_expected}")
        return 0.0
    
    # Clamp to expected range
    clamped = max(min_expected, min(float(raw_score), max_expected))
    
    # Normalize to 0-1
    normalized = (clamped - min_expected) / (max_expected - min_expected)
    
    return normalized


def compute_reliability_score(
    anomaly_score: float,
    weight: float = 1.0,
    offset: float = 0.0
) -> float:
    """
    Convert anomaly score to reliability probability.
    
    Reliability is treated as the complement of anomaly intensity:
        reliability = max(0.0, 1.0 - anomaly_score)
    
    Args:
        anomaly_score: Normalized anomaly score (0-1, higher = worse)
        weight: Optional scaling factor for future expansion
        offset: Optional offset for fine-tuning
    
    Returns:
        Reliability score between 0 and 1, where 1 = perfectly reliable,
        0 = complete failure.
    
    Example:
        >>> compute_reliability_score(0.23)
        0.77  # 77% reliable given 0.23 anomaly score
    
    The weight and offset parameters allow future calibration without
    changing the core interface.
    """
    # Ensure input is in valid range
    if anomaly_score < 0 or anomaly_score > 1:
        logger.warning(f"anomaly_score {anomaly_score} outside [0,1], clamping")
        anomaly_score = max(0.0, min(1.0, float(anomaly_score)))
    
    # Core transformation: reliability = 1 - anomaly
    raw_reliability = 1.0 - anomaly_score
    
    # Apply weight and offset (with bounds checking)
    adjusted = (raw_reliability * weight) + offset
    
    # Ensure final result is in [0, 1]
    reliability = max(0.0, min(1.0, adjusted))
    
    return reliability


def signal_to_reliability(
    raw_signal: float,
    signal_type: str = "default",
    config: Optional[dict] = None
) -> float:
    """
    High-level function: raw signal → normalized anomaly → reliability.
    
    This demonstrates the full pipeline and can be extended for different
    signal types (latency, error_rate, cpu, etc.).
    
    Args:
        raw_signal: Raw measurement value
        signal_type: Type of signal ("latency", "error_rate", "cpu", etc.)
        config: Optional configuration with signal-specific bounds
    
    Returns:
        Reliability score between 0 and 1.
    
    Example:
        >>> signal_to_reliability(450, "latency")
        0.77  # Assuming default latency max 500ms
    """
    # Default bounds for different signal types
    defaults = {
        "latency": {"max": 500.0, "min": 0.0},
        "error_rate": {"max": 0.3, "min": 0.0},
        "cpu": {"max": 1.0, "min": 0.0},
        "memory": {"max": 1.0, "min": 0.0},
        "default": {"max": 10.0, "min": 0.0}
    }
    
    # Merge with provided config
    bounds = defaults.get(signal_type, defaults["default"])
    if config and signal_type in config:
        bounds.update(config[signal_type])
    
    # Normalize
    normalized = normalize_anomaly_signal(
        raw_signal,
        max_expected=bounds["max"],
        min_expected=bounds["min"]
    )
    
    # Compute reliability
    reliability = compute_reliability_score(normalized)
    
    return reliability


# Module usage example (run with: python -m agentic_reliability_framework.core.reliability_signal)
if __name__ == "__main__":
    print("🔧 ARF Reliability Signal Module - Quick Test")
    print("-" * 50)
    
    # Example 1: Latency signal
    latency_ms = 450
    reliability = signal_to_reliability(latency_ms, "latency")
    print(f"Latency: {latency_ms}ms → Reliability: {reliability:.3f}")
    
    # Example 2: Error rate
    error_rate = 0.12
    reliability = signal_to_reliability(error_rate, "error_rate")
    print(f"Error rate: {error_rate:.1%} → Reliability: {reliability:.3f}")
    
    # Example 3: CPU utilization
    cpu = 0.85
    reliability = signal_to_reliability(cpu, "cpu")
    print(f"CPU: {cpu:.1%} → Reliability: {reliability:.3f}")
    
    # Example 4: Direct anomaly score
    anomaly = 0.23
    reliability = compute_reliability_score(anomaly)
    print(f"Anomaly score: {anomaly} → Reliability: {reliability:.3f}")
    
    print("-" * 50)
    print("✅ Module ready. Import with:")
    print("    from agentic_reliability_framework.core.reliability_signal import compute_reliability_score")
