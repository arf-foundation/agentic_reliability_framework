"""
ARF Diagnostic CLI

Provides quick visibility into system reliability signals.
Usage: arf diagnose
"""

import argparse
import sys
from typing import Optional

try:
    from agentic_reliability_framework.core.reliability_signal import (
        compute_reliability_score,
        signal_to_reliability
    )
except ImportError:
    sys.path.insert(0, '.')
    from agentic_reliability_framework.core.reliability_signal import (
        compute_reliability_score,
        signal_to_reliability
    )


def get_risk_level(reliability: float) -> str:
    if reliability >= 0.8:
        return "LOW"
    elif reliability >= 0.5:
        return "MEDIUM"
    elif reliability >= 0.3:
        return "HIGH"
    else:
        return "CRITICAL"


def get_suggested_action(reliability: float) -> str:
    if reliability >= 0.8:
        return "Monitor"
    elif reliability >= 0.5:
        return "Investigate"
    elif reliability >= 0.3:
        return "Intervene"
    else:
        return "IMMEDIATE ACTION REQUIRED"


def run_diagnose(anomaly_score=None, latency=None, error_rate=None):
    if anomaly_score is not None:
        reliability = compute_reliability_score(anomaly_score)
        source = "direct anomaly score"
    elif latency is not None:
        reliability = signal_to_reliability(latency, "latency")
        source = f"latency ({latency}ms)"
    elif error_rate is not None:
        reliability = signal_to_reliability(error_rate, "error_rate")
        source = f"error rate ({error_rate:.1%})"
    else:
        reliability = 0.77
        source = "demo (no input provided)"

    risk_level = get_risk_level(reliability)
    action = get_suggested_action(reliability)

    return {
        "reliability": reliability,
        "risk_level": risk_level,
        "suggested_action": action,
        "source": source,
        "anomaly_score": 1.0 - reliability if anomaly_score is None else anomaly_score
    }


def main():
    parser = argparse.ArgumentParser(description="ARF Diagnostic Tool")
    parser.add_argument("--anomaly", "-a", type=float, help="Anomaly score (0-1)")
    parser.add_argument("--latency", "-l", type=float, help="Latency in milliseconds")
    parser.add_argument("--error-rate", "-e", type=float, help="Error rate (0-1)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")

    args = parser.parse_args()

    result = run_diagnose(
        anomaly_score=args.anomaly,
        latency=args.latency,
        error_rate=getattr(args, 'error_rate')
    )

    print("\n🔍 ARF Diagnostic Report")
    print("=" * 50)
    print(f"Reliability Score:   {result['reliability']:.3f}")
    print(f"Risk Level:          {result['risk_level']}")
    print(f"Suggested Action:    {result['suggested_action']}")
    print(f"Signal Source:       {result['source']}")

    if args.verbose:
        print("\n📊 Detailed Metrics")
        print("-" * 50)
        print(f"Anomaly Score:       {result['anomaly_score']:.3f}")
        print("Risk Thresholds:     LOW ≥0.8, MEDIUM ≥0.5, HIGH ≥0.3")

    print("=" * 50)

    if result['risk_level'] == "CRITICAL":
        return 2
    elif result['risk_level'] == "HIGH":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
