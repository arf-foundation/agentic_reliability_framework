"""
Policy Engine for Automated Healing Actions.
"""

import threading
import logging
import datetime
from collections import OrderedDict
from typing import Dict, List, Optional, Any

from agentic_reliability_framework.core.models.event import HealingAction, ReliabilityEvent
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PolicyCondition(BaseModel):
    """A single condition in a policy."""
    metric: str
    operator: str  # gt, lt, eq, gte, lte
    threshold: float


class HealingPolicy(BaseModel):
    """A policy that defines conditions and actions."""
    name: str
    conditions: List[PolicyCondition]
    actions: List[HealingAction]
    priority: int
    cool_down_seconds: int
    max_executions_per_hour: int
    enabled: bool = True


DEFAULT_HEALING_POLICIES = [
    HealingPolicy(
        name="high_latency_restart",
        conditions=[PolicyCondition(metric="latency_p99", operator="gt", threshold=500.0)],
        actions=[HealingAction.RESTART_CONTAINER, HealingAction.ALERT_TEAM],
        priority=1,
        cool_down_seconds=300,
        max_executions_per_hour=5
    ),
    HealingPolicy(
        name="critical_error_rate_rollback",
        conditions=[PolicyCondition(metric="error_rate", operator="gt", threshold=0.3)],
        actions=[HealingAction.ROLLBACK, HealingAction.CIRCUIT_BREAKER, HealingAction.ALERT_TEAM],
        priority=1,
        cool_down_seconds=600,
        max_executions_per_hour=3
    ),
    HealingPolicy(
        name="high_error_rate_traffic_shift",
        conditions=[PolicyCondition(metric="error_rate", operator="gt", threshold=0.15)],
        actions=[HealingAction.TRAFFIC_SHIFT, HealingAction.ALERT_TEAM],
        priority=2,
        cool_down_seconds=300,
        max_executions_per_hour=5
    ),
    HealingPolicy(
        name="resource_exhaustion_scale",
        conditions=[
            PolicyCondition(metric="cpu_util", operator="gt", threshold=0.9),
            PolicyCondition(metric="memory_util", operator="gt", threshold=0.9)
        ],
        actions=[HealingAction.SCALE_OUT],
        priority=2,
        cool_down_seconds=600,
        max_executions_per_hour=10
    ),
    HealingPolicy(
        name="moderate_latency_circuit_breaker",
        conditions=[PolicyCondition(metric="latency_p99", operator="gt", threshold=300.0)],
        actions=[HealingAction.CIRCUIT_BREAKER],
        priority=3,
        cool_down_seconds=180,
        max_executions_per_hour=8
    )
]


class PolicyEngine:
    """
    Thread‑safe policy engine with cooldown and rate limiting.
    Policies are evaluated in priority order. Each policy has:
      - conditions (AND logic)
      - cooldown per (policy, component)
      - rate limit per hour
    The engine maintains an LRU cache of last execution timestamps
    (using OrderedDict) to bound memory usage.
    """

    def __init__(
        self,
        policies: Optional[List[HealingPolicy]] = None,
        max_cooldown_history: int = 10000,
        max_execution_history: int = 1000
    ):
        """
        Args:
            policies: List of HealingPolicy objects. If None, DEFAULT_HEALING_POLICIES used.
            max_cooldown_history: Maximum number of (policy, component) entries to keep.
            max_execution_history: Maximum number of timestamps stored per policy for rate limiting.
        """
        self.policies = policies or DEFAULT_HEALING_POLICIES
        self._lock = threading.RLock()
        # OrderedDict acts as an LRU cache: last item is most recent.
        self.last_execution: OrderedDict[str, float] = OrderedDict()
        self.max_cooldown_history = max_cooldown_history
        self.execution_timestamps: Dict[str, List[float]] = {}
        self.max_execution_history = max_execution_history
        self.policies = sorted(self.policies, key=lambda p: p.priority)
        logger.info(f"Initialized PolicyEngine with {len(self.policies)} policies")

    def evaluate_policies(self, event: ReliabilityEvent) -> List[HealingAction]:
        """
        Evaluate all policies against the event and return the set of actions
        triggered (deduplicated). Returns [NO_ACTION] if none triggered.
        """
        applicable_actions = []
        current_time = datetime.datetime.now(datetime.timezone.utc).timestamp()
        for policy in self.policies:
            if not policy.enabled:
                continue
            policy_key = f"{policy.name}_{event.component}"
            with self._lock:
                last_exec = self.last_execution.get(policy_key, 0)
                if current_time - last_exec < policy.cool_down_seconds:
                    continue
                if self._is_rate_limited(policy_key, policy, current_time):
                    continue
                if self._evaluate_conditions(policy.conditions, event):
                    applicable_actions.extend(policy.actions)
                    # Update cooldown
                    self.last_execution[policy_key] = current_time
                    self.last_execution.move_to_end(policy_key)  # mark as most recent
                    # Enforce cache size
                    if len(self.last_execution) > self.max_cooldown_history:
                        self.last_execution.popitem(last=False)
                    self._record_execution(policy_key, current_time)

        # Deduplicate actions while preserving order
        seen = set()
        unique = []
        for a in applicable_actions:
            if a not in seen:
                seen.add(a)
                unique.append(a)
        return unique if unique else [HealingAction.NO_ACTION]

    def _evaluate_conditions(self, conditions: List[PolicyCondition], event: ReliabilityEvent) -> bool:
        """Return True if all conditions are satisfied."""
        for cond in conditions:
            metric = cond.metric
            op = cond.operator
            thresh = cond.threshold
            val = getattr(event, metric, None)
            if val is None:
                return False
            if op == "gt":
                if not (val > thresh):
                    return False
            elif op == "lt":
                if not (val < thresh):
                    return False
            elif op == "eq":
                if not (abs(val - thresh) < 1e-6):
                    return False
            elif op == "gte":
                if not (val >= thresh):
                    return False
            elif op == "lte":
                if not (val <= thresh):
                    return False
            else:
                return False
        return True

    def _is_rate_limited(self, key: str, policy: HealingPolicy, now: float) -> bool:
        """
        Check if the policy has exceeded its hourly execution limit.

        This method prunes old timestamps BEFORE checking the rate limit to ensure
        correct behavior. Thread-safety is enforced by the caller (evaluate_policies).

        Args:
            key: Policy key (policy_name_component)
            policy: HealingPolicy with max_executions_per_hour limit
            now: Current timestamp (seconds since epoch)

        Returns:
            bool: True if rate limit exceeded, False otherwise.
        """
        if key not in self.execution_timestamps:
            # First time this policy was triggered for this component
            return False

        # Prune timestamps older than 1 hour
        one_hour_ago = now - 3600.0
        recent_timestamps = [ts for ts in self.execution_timestamps[key] if ts > one_hour_ago]
        self.execution_timestamps[key] = recent_timestamps

        # Check if we've hit the limit
        is_limited = len(recent_timestamps) >= policy.max_executions_per_hour
        return is_limited

    def _record_execution(self, key: str, ts: float):
        """
        Record an execution timestamp for rate limiting.

        Appends timestamp and enforces max history size.
        Thread-safety is enforced by the caller (evaluate_policies).

        Args:
            key: Policy key (policy_name_component)
            ts: Timestamp to record (seconds since epoch)
        """
        if key not in self.execution_timestamps:
            self.execution_timestamps[key] = []
        
        self.execution_timestamps[key].append(ts)
        
        # Trim to latest max_execution_history entries (LRU eviction)
        if len(self.execution_timestamps[key]) > self.max_execution_history:
            self.execution_timestamps[key] = self.execution_timestamps[key][-self.max_execution_history:]
