"""
Predictive analytics engine using timestamp‑based linear regression with risk classification.
"""
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Deque
from collections import deque

from agentic_reliability_framework.core.models.event import ForecastResult, ReliabilityEvent
from agentic_reliability_framework.core.config.constants import (
    FORECAST_MIN_DATA_POINTS,
    LATENCY_WARNING, LATENCY_EXTREME, LATENCY_CRITICAL,
    ERROR_RATE_WARNING, ERROR_RATE_CRITICAL,
    CPU_WARNING, CPU_CRITICAL, MEMORY_WARNING, MEMORY_CRITICAL,
    CACHE_EXPIRY_MINUTES, BASE_REVENUE_PER_MINUTE, BASE_USERS
)


class SimplePredictiveEngine:
    """Simple timestamp‑based linear regression predictive engine with caching."""

    def __init__(self, history_window: int = 100):
        self.history_window = history_window
        self.service_history: Dict[str, Deque] = {}
        self.prediction_cache: Dict[str, tuple] = {}  # key -> (ForecastResult, timestamp)
        self.max_cache_age = timedelta(minutes=CACHE_EXPIRY_MINUTES)

    def add_telemetry(self, service: str, metrics: Dict[str, float]):
        """Store a new telemetry point for a service."""
        if service not in self.service_history:
            self.service_history[service] = deque(maxlen=self.history_window)
        point = {
            'timestamp': datetime.now(timezone.utc),
            **metrics
        }
        # Normalise key names
        if 'latency_p99' in point:
            point['latency'] = point.pop('latency_p99')
        self.service_history[service].append(point)
        # Invalidate cache for this service
        keys_to_remove = [k for k in self.prediction_cache if k.startswith(f"{service}_")]
        for key in keys_to_remove:
            del self.prediction_cache[key]

    def _clean_cache(self):
        """Remove stale cached forecasts."""
        now = datetime.now(timezone.utc)
        expired = [k for k, (_, ts) in self.prediction_cache.items()
                   if now - ts > self.max_cache_age]
        for k in expired:
            del self.prediction_cache[k]

    def _forecast_metric(self, history: List[Dict], metric: str, lookahead_minutes: int,
                         lower_bound: float = 0, upper_bound: Optional[float] = None,
                         thresholds: Optional[Dict[str, float]] = None) -> Optional[ForecastResult]:
        """Generic forecasting for a single metric using timestamp regression."""
        if len(history) < FORECAST_MIN_DATA_POINTS:
            return None
        # Extract timestamps and values
        timestamps = [p['timestamp'] for p in history if metric in p]
        values = [p[metric] for p in history if metric in p]
        if len(values) < FORECAST_MIN_DATA_POINTS:
            return None
        # Convert to seconds since first point
        t0 = timestamps[0]
        t_sec = [(ts - t0).total_seconds() for ts in timestamps]
        x = np.array(t_sec)
        y = np.array(values)
        try:
            slope, intercept = np.polyfit(x, y, 1)
            # Forecast at current time + lookahead minutes
            future_seconds = lookahead_minutes * 60
            predicted = intercept + slope * (x[-1] + future_seconds)
            if upper_bound is not None:
                predicted = min(predicted, upper_bound)
            predicted = max(predicted, lower_bound)

            # Trend
            if slope > 0.01 / 60:  # per minute slope > 0.01
                trend = "increasing"
            elif slope < -0.01 / 60:
                trend = "decreasing"
            else:
                trend = "stable"

            # Risk level based on thresholds
            if metric == "latency":
                if predicted >= LATENCY_CRITICAL:
                    risk = "critical"
                elif predicted >= LATENCY_EXTREME:
                    risk = "high"
                elif predicted >= LATENCY_WARNING:
                    risk = "medium"
                else:
                    risk = "low"
            elif metric == "error_rate":
                if predicted >= ERROR_RATE_CRITICAL:
                    risk = "critical"
                elif predicted >= ERROR_RATE_WARNING:
                    risk = "high"
                else:
                    risk = "low"
            else:  # cpu_util, memory_util
                if predicted >= (CPU_CRITICAL if metric == "cpu_util" else MEMORY_CRITICAL):
                    risk = "critical"
                elif predicted >= (CPU_WARNING if metric == "cpu_util" else MEMORY_WARNING):
                    risk = "high"
                else:
                    risk = "low"

            # Confidence based on R²
            residuals = y - (intercept + slope * x)
            r2 = 1 - (np.var(residuals) / np.var(y)) if len(y) > 1 else 0.5
            confidence = max(0.0, min(1.0, r2))

            # Estimate time to threshold (in minutes)
            time_to_threshold = None
            if thresholds and "threshold" in thresholds:
                threshold_val = thresholds["threshold"]
                if slope > 0:
                    last_val = y[-1]
                    if last_val < threshold_val:
                        seconds_needed = (threshold_val - last_val) / max(slope, 1e-6)
                        time_to_threshold = seconds_needed / 60.0

            return ForecastResult(
                metric=metric,
                predicted_value=float(predicted),
                confidence=float(confidence),
                trend=trend,
                risk_level=risk,
                time_to_threshold=time_to_threshold
            )
        except Exception:
            return None

    def forecast_service_health(self, service: str) -> List[ForecastResult]:
        """Return forecasts for all available metrics of a service."""
        self._clean_cache()
        if service not in self.service_history:
            return []
        history = list(self.service_history[service])
        forecasts = []
        for metric in ["latency", "error_rate", "cpu_util", "memory_util"]:
            key = f"{service}_{metric}"
            if key in self.prediction_cache:
                f, _ = self.prediction_cache[key]
                forecasts.append(f)
            else:
                if metric == "latency":
                    f = self._forecast_metric(history, "latency", 10, lower_bound=0,
                                               thresholds={"threshold": LATENCY_CRITICAL})
                elif metric == "error_rate":
                    f = self._forecast_metric(history, "error_rate", 10, lower_bound=0, upper_bound=1,
                                               thresholds={"threshold": ERROR_RATE_CRITICAL})
                elif metric == "cpu_util":
                    f = self._forecast_metric(history, "cpu_util", 10, lower_bound=0, upper_bound=1,
                                               thresholds={"threshold": CPU_CRITICAL})
                elif metric == "memory_util":
                    f = self._forecast_metric(history, "memory_util", 10, lower_bound=0, upper_bound=1,
                                               thresholds={"threshold": MEMORY_CRITICAL})
                if f:
                    self.prediction_cache[key] = (f, datetime.now(timezone.utc))
                    forecasts.append(f)
        return forecasts

    def get_predictive_insights(self, service: str) -> Dict[str, Any]:
        """Return structured insights including warnings and recommendations."""
        forecasts = self.forecast_service_health(service)
        warnings = []
        recommendations = []
        critical_count = 0
        for f in forecasts:
            if f.risk_level == "critical":
                critical_count += 1
                warnings.append(f"{f.metric} is forecast to reach critical levels.")
                recommendations.append(f"Immediate action required on {f.metric}.")
            elif f.risk_level == "high":
                warnings.append(f"{f.metric} is forecast to be high.")
                recommendations.append(f"Consider scaling or reviewing {f.metric}.")
        return {
            "service": service,
            "forecasts": forecasts,
            "warnings": warnings,
            "recommendations": recommendations,
            "critical_risk_count": critical_count,
            "forecast_timestamp": datetime.now(timezone.utc).isoformat()
        }

    # Backward‑compatible methods (wrappers around _forecast_metric)
    def _forecast_latency(self, history: List[Dict], lookahead: int) -> Optional[ForecastResult]:
        return self._forecast_metric(history, "latency", lookahead, lower_bound=0)

    def _forecast_error_rate(self, history: List[Dict], lookahead: int) -> Optional[ForecastResult]:
        return self._forecast_metric(history, "error_rate", lookahead, lower_bound=0, upper_bound=1)

    def _forecast_resources(self, history: List[Dict], lookahead: int) -> List[ForecastResult]:
        forecasts = []
        for metric in ["cpu_util", "memory_util"]:
            f = self._forecast_metric(history, metric, lookahead, lower_bound=0, upper_bound=1)
            if f:
                forecasts.append(f)
        return forecasts


class BusinessImpactCalculator:
    """Calculate business impact from a ReliabilityEvent."""

    def __init__(self, revenue_per_request: float = 0.01):
        self.revenue_per_request = revenue_per_request

    def calculate_impact(self, event: ReliabilityEvent, duration_minutes: int = 5) -> Dict:
        latency = event.latency_p99 if event.latency_p99 is not None else 0.0
        error_rate = event.error_rate  # keep None for fallback
        cpu = event.cpu_util if event.cpu_util is not None else 0.5

        # Handle throughput: use BASE_USERS if missing or zero
        throughput = event.throughput
        if throughput is None or throughput == 0:
            throughput = BASE_USERS

        # NEW: Fallback for missing error_rate
        if error_rate is None:
            revenue_loss = BASE_REVENUE_PER_MINUTE * (duration_minutes / 60)
            affected_users = 0
            # No latency factor, no throughput multiplier
        else:
            # Impact estimation: revenue loss = throughput * revenue_per_request * error_rate * duration
            revenue_loss = throughput * self.revenue_per_request * error_rate * duration_minutes
            # Also include latency impact: higher latency reduces effective throughput
            latency_factor = 1 + (latency / 1000)  # normalized
            revenue_loss *= latency_factor
            affected_users = throughput * error_rate

        # Severity classification uses effective error_rate (0 if None)
        effective_error = error_rate if error_rate is not None else 0.0
        if latency >= LATENCY_CRITICAL or effective_error >= ERROR_RATE_CRITICAL or cpu >= CPU_CRITICAL:
            severity = "CRITICAL"
        elif latency >= LATENCY_EXTREME or effective_error >= ERROR_RATE_WARNING or cpu >= CPU_WARNING:
            severity = "HIGH"
        elif latency >= LATENCY_WARNING or effective_error >= 0.02 or cpu >= 0.7:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        return {
            "revenue_loss_estimate": revenue_loss,
            "affected_users_estimate": affected_users,
            "severity_level": severity,
            "throughput_reduction_pct": effective_error * 100
        }
