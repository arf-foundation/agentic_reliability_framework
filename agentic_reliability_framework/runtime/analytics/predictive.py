"""
Predictive analytics engine for forecasting service health.
"""

import threading
import logging
import datetime
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple, Any

from agentic_reliability_framework.core.models.event import ForecastResult, ReliabilityEvent
from agentic_reliability_framework.core.config.constants import (
    HISTORY_WINDOW, CACHE_EXPIRY_MINUTES, FORECAST_MIN_DATA_POINTS,
    FORECAST_LOOKAHEAD_MINUTES, SLOPE_THRESHOLD_INCREASING,
    SLOPE_THRESHOLD_DECREASING, LATENCY_WARNING, LATENCY_EXTREME,
    ERROR_RATE_WARNING, ERROR_RATE_CRITICAL, CPU_WARNING, CPU_CRITICAL,
    MEMORY_WARNING, MEMORY_CRITICAL
)

logger = logging.getLogger(__name__)


class SimplePredictiveEngine:
    def __init__(self, history_window: int = HISTORY_WINDOW):
        self.history_window = history_window
        self.service_history: Dict[str, deque] = {}
        self.prediction_cache: Dict[str, Tuple[ForecastResult, datetime.datetime]] = {}
        self.max_cache_age = datetime.timedelta(minutes=CACHE_EXPIRY_MINUTES)
        self._lock = threading.RLock()
        logger.info(f"Initialized SimplePredictiveEngine with history_window={history_window}")

    def add_telemetry(self, service: str, event_data: Dict):
        with self._lock:
            if service not in self.service_history:
                self.service_history[service] = deque(maxlen=self.history_window)
            telemetry_point = {
                'timestamp': datetime.datetime.now(datetime.timezone.utc),
                'latency': event_data.get('latency_p99', 0),
                'error_rate': event_data.get('error_rate', 0),
                'throughput': event_data.get('throughput', 0),
                'cpu_util': event_data.get('cpu_util'),
                'memory_util': event_data.get('memory_util')
            }
            self.service_history[service].append(telemetry_point)
            self._clean_cache()

    def _clean_cache(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        expired = [k for k, (_, ts) in self.prediction_cache.items() if now - ts > self.max_cache_age]
        for k in expired:
            del self.prediction_cache[k]

    def forecast_service_health(self, service: str, lookahead_minutes: int = FORECAST_LOOKAHEAD_MINUTES) -> List[ForecastResult]:
        with self._lock:
            if service not in self.service_history or len(self.service_history[service]) < FORECAST_MIN_DATA_POINTS:
                return []
            history = list(self.service_history[service])
        forecasts = []
        latency_forecast = self._forecast_latency(history, lookahead_minutes)
        if latency_forecast:
            forecasts.append(latency_forecast)
        error_forecast = self._forecast_error_rate(history, lookahead_minutes)
        if error_forecast:
            forecasts.append(error_forecast)
        resource_forecasts = self._forecast_resources(history, lookahead_minutes)
        forecasts.extend(resource_forecasts)
        with self._lock:
            for forecast in forecasts:
                cache_key = f"{service}_{forecast.metric}"
                self.prediction_cache[cache_key] = (forecast, datetime.datetime.now(datetime.timezone.utc))
        return forecasts

    def _forecast_latency(self, history: List, lookahead_minutes: int) -> Optional[ForecastResult]:
        try:
            latencies = [point['latency'] for point in history[-20:]]
            if len(latencies) < FORECAST_MIN_DATA_POINTS:
                return None
            x = np.arange(len(latencies))
            slope, intercept = np.polyfit(x, latencies, 1)
            next_x = len(latencies)
            predicted = float(slope * next_x + intercept)  # ensure float
            residuals = latencies - (slope * x + intercept)
            # compute confidence safely
            mean_lat = np.mean(latencies) or 1.0
            conf = 1.0 - (np.std(residuals) / mean_lat)
            confidence = max(0.0, min(1.0, float(conf)))
            if slope > SLOPE_THRESHOLD_INCREASING:
                trend = "increasing"
                risk = "critical" if predicted > LATENCY_EXTREME else "high"
            elif slope < SLOPE_THRESHOLD_DECREASING:
                trend = "decreasing"
                risk = "low"
            else:
                trend = "stable"
                risk = "low" if predicted < LATENCY_WARNING else "medium"
            time_to_threshold = None
            if slope > 0 and predicted < LATENCY_EXTREME:
                denominator = predicted - latencies[-1]
                if abs(denominator) > 0.1:
                    minutes_to = lookahead_minutes * (LATENCY_EXTREME - predicted) / denominator
                    if minutes_to > 0:
                        time_to_threshold = float(minutes_to)
            return ForecastResult(
                metric="latency",
                predicted_value=predicted,
                confidence=confidence,
                trend=trend,
                time_to_threshold=time_to_threshold,
                risk_level=risk
            )
        except Exception as e:
            logger.error(f"Latency forecast error: {e}")
            return None

    def _forecast_error_rate(self, history: List, lookahead_minutes: int) -> Optional[ForecastResult]:
        try:
            error_rates = [point['error_rate'] for point in history[-15:]]
            if len(error_rates) < FORECAST_MIN_DATA_POINTS:
                return None
            alpha = 0.3
            forecast = error_rates[0]
            for rate in error_rates[1:]:
                forecast = alpha * rate + (1 - alpha) * forecast
            predicted = float(forecast)
            recent_trend = np.mean(error_rates[-3:]) - np.mean(error_rates[-6:-3])
            if recent_trend > 0.02:
                trend = "increasing"
                risk = "critical" if predicted > ERROR_RATE_CRITICAL else "high"
            elif recent_trend < -0.01:
                trend = "decreasing"
                risk = "low"
            else:
                trend = "stable"
                risk = "low" if predicted < ERROR_RATE_WARNING else "medium"
            # confidence based on variance
            std_err = np.std(error_rates) if len(error_rates) > 1 else 0.0
            mean_err = np.mean(error_rates) or 0.01
            conf = 1.0 - (std_err / mean_err)
            confidence = max(0.0, min(1.0, float(conf)))
            return ForecastResult(
                metric="error_rate",
                predicted_value=predicted,
                confidence=confidence,
                trend=trend,
                risk_level=risk
            )
        except Exception as e:
            logger.error(f"Error rate forecast error: {e}")
            return None

    def _forecast_resources(self, history: List, lookahead_minutes: int) -> List[ForecastResult]:
        forecasts = []
        cpu_vals = [p['cpu_util'] for p in history if p.get('cpu_util') is not None]
        if len(cpu_vals) >= FORECAST_MIN_DATA_POINTS:
            try:
                predicted = float(np.mean(cpu_vals[-5:]))
                recent = cpu_vals[-1] if cpu_vals else 0
                older = np.mean(cpu_vals[-10:-5]) if len(cpu_vals) >= 10 else predicted
                trend = "increasing" if recent > older else "stable"
                risk = "low"
                if predicted > CPU_CRITICAL:
                    risk = "critical"
                elif predicted > CPU_WARNING:
                    risk = "high"
                elif predicted > 0.7:
                    risk = "medium"
                forecasts.append(ForecastResult(
                    metric="cpu_util",
                    predicted_value=predicted,
                    confidence=0.7,
                    trend=trend,
                    risk_level=risk
                ))
            except Exception as e:
                logger.error(f"CPU forecast error: {e}")
        mem_vals = [p['memory_util'] for p in history if p.get('memory_util') is not None]
        if len(mem_vals) >= FORECAST_MIN_DATA_POINTS:
            try:
                predicted = float(np.mean(mem_vals[-5:]))
                recent = mem_vals[-1] if mem_vals else 0
                older = np.mean(mem_vals[-10:-5]) if len(mem_vals) >= 10 else predicted
                trend = "increasing" if recent > older else "stable"
                risk = "low"
                if predicted > MEMORY_CRITICAL:
                    risk = "critical"
                elif predicted > MEMORY_WARNING:
                    risk = "high"
                elif predicted > 0.7:
                    risk = "medium"
                forecasts.append(ForecastResult(
                    metric="memory_util",
                    predicted_value=predicted,
                    confidence=0.7,
                    trend=trend,
                    risk_level=risk
                ))
            except Exception as e:
                logger.error(f"Memory forecast error: {e}")
        return forecasts

    def get_predictive_insights(self, service: str) -> Dict[str, Any]:
        forecasts = self.forecast_service_health(service)
        critical_risks = [f for f in forecasts if f.risk_level in ["high", "critical"]]
        warnings = []
        recommendations = []
        for f in critical_risks:
            if f.metric == "latency":
                warnings.append(f"📈 Latency expected to reach {f.predicted_value:.0f}ms")
                if f.time_to_threshold:
                    recommendations.append(f"⏰ Critical latency in ~{int(f.time_to_threshold)} minutes")
                recommendations.append("🔧 Consider scaling or optimizing dependencies")
            elif f.metric == "error_rate":
                warnings.append(f"🚨 Errors expected to reach {f.predicted_value*100:.1f}%")
                recommendations.append("🐛 Investigate recent deployments or dependency issues")
            elif f.metric == "cpu_util":
                warnings.append(f"🔥 CPU expected at {f.predicted_value*100:.1f}%")
                recommendations.append("⚡ Consider scaling compute resources")
            elif f.metric == "memory_util":
                warnings.append(f"💾 Memory expected at {f.predicted_value*100:.1f}%")
                recommendations.append("🧹 Check for memory leaks or optimize usage")
        return {
            'service': service,
            'forecasts': [f.model_dump() for f in forecasts],
            'warnings': warnings[:3],
            'recommendations': list(dict.fromkeys(recommendations))[:3],
            'critical_risk_count': len(critical_risks),
            'forecast_timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()
        }


class BusinessImpactCalculator:
    def __init__(self, revenue_per_request: float = 0.01):
        self.revenue_per_request = revenue_per_request
        logger.info("Initialized BusinessImpactCalculator")

    def calculate_impact(self, event: ReliabilityEvent, duration_minutes: int = 5) -> Dict[str, Any]:
        from agentic_reliability_framework.core.config.constants import (
            BASE_REVENUE_PER_MINUTE, BASE_USERS, LATENCY_CRITICAL, CPU_CRITICAL
        )
        impact_multiplier = 1.0
        if event.latency_p99 and event.latency_p99 > LATENCY_CRITICAL:
            impact_multiplier += 0.5
        if event.error_rate and event.error_rate > 0.1:
            impact_multiplier += 0.8
        if event.cpu_util and event.cpu_util > CPU_CRITICAL:
            impact_multiplier += 0.3
        revenue_loss = BASE_REVENUE_PER_MINUTE * impact_multiplier * (duration_minutes / 60)
        user_impact_multiplier = (event.error_rate * 10) + (max(0, (event.latency_p99 or 0) - 100) / 500)
        affected_users = int(BASE_USERS * user_impact_multiplier)
        if revenue_loss > 500 or affected_users > 5000:
            severity = "CRITICAL"
        elif revenue_loss > 100 or affected_users > 1000:
            severity = "HIGH"
        elif revenue_loss > 50 or affected_users > 500:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        logger.info(f"Business impact: ${revenue_loss:.2f} revenue loss, {affected_users} users, {severity} severity")
        return {
            'revenue_loss_estimate': round(revenue_loss, 2),
            'affected_users_estimate': affected_users,
            'severity_level': severity,
            'throughput_reduction_pct': round(min(100, user_impact_multiplier * 100), 1)
        }
