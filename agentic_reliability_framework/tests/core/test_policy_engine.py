"""
Tests for PolicyEngine cooldown and rate-limiting correctness.

Validates that:
- Cooldown prevents rapid policy execution
- Rate-limiting enforces hourly limits
- Timestamp pruning works correctly
- Thread-safety is maintained
"""

import pytest
import time
import threading
from datetime import datetime, timezone

from agentic_reliability_framework.core.governance.policy_engine import (
    PolicyEngine,
    HealingPolicy,
    PolicyCondition,
)
from agentic_reliability_framework.core.models.event import (
    ReliabilityEvent,
    HealingAction,
)


@pytest.fixture
def test_policy():
    """Create a test policy with cooldown and rate limit."""
    return HealingPolicy(
        name="test_policy",
        conditions=[
            PolicyCondition(metric="latency_p99", operator="gt", threshold=300.0)
        ],
        actions=[HealingAction.RESTART_CONTAINER],
        priority=1,
        cool_down_seconds=60,
        max_executions_per_hour=3,
        enabled=True,
    )


@pytest.fixture
def test_event():
    """Create a test event that triggers the policy."""
    return ReliabilityEvent(
        component="test-service",
        latency_p99=400,  # > 300, triggers the policy
        error_rate=0.05,
        throughput=1000,
    )


def test_cooldown_prevents_rapid_execution(test_policy, test_event):
    """Test that cooldown prevents rapid repeated execution."""
    engine = PolicyEngine(policies=[test_policy])

    # First evaluation should trigger
    actions1 = engine.evaluate_policies(test_event)
    assert HealingAction.RESTART_CONTAINER in actions1

    # Immediate second evaluation should be blocked by cooldown
    actions2 = engine.evaluate_policies(test_event)
    assert HealingAction.NO_ACTION in actions2

    # After cooldown expires, should trigger again
    policy_key = f"{test_policy.name}_{test_event.component}"
    # Simulate time passage
    engine.last_execution[policy_key] = time.time() - 61  # 61 seconds ago

    actions3 = engine.evaluate_policies(test_event)
    assert HealingAction.RESTART_CONTAINER in actions3


def test_rate_limit_enforces_hourly_max(test_policy, test_event):
    """Test that rate limiting enforces hourly execution limits."""
    # Reduce max_executions_per_hour for faster testing
    test_policy.max_executions_per_hour = 2
    test_policy.cool_down_seconds = 0  # Disable cooldown to test rate limit

    engine = PolicyEngine(policies=[test_policy])

    policy_key = f"{test_policy.name}_{test_event.component}"
    current_time = time.time()

    # Manually set up execution timestamps to simulate 2 prior executions
    engine.execution_timestamps[policy_key] = [
        current_time - 500,  # 500 seconds ago
        current_time - 300,  # 300 seconds ago
    ]
    
    # Try to trigger a third execution - should be rate limited
    is_limited = engine._is_rate_limited(policy_key, test_policy, current_time)
    assert is_limited is True, "Should be rate limited at max_executions_per_hour=2"


def test_timestamp_pruning_removes_old_entries(test_policy, test_event):
    """Test that old timestamps are pruned correctly."""
    test_policy.cool_down_seconds = 0
    test_policy.max_executions_per_hour = 100  # High limit to test pruning
    engine = PolicyEngine(policies=[test_policy])

    policy_key = f"{test_policy.name}_{test_event.component}"
    current_time = time.time()

    # Manually add timestamps: some old (>1 hour), some recent
    engine.execution_timestamps[policy_key] = [
        current_time - 7200,  # 2 hours old (should be pruned)
        current_time - 3600,  # 1 hour old (exactly at boundary, should be removed)
        current_time - 1800,  # 30 minutes old (should be kept)
        current_time - 100,   # 1.67 minutes old (should be kept)
    ]

    # Call _is_rate_limited to trigger pruning
    is_limited = engine._is_rate_limited(policy_key, test_policy, current_time)

    # After pruning, should have only 2 recent timestamps
    recent = engine.execution_timestamps[policy_key]
    assert len(recent) == 2
    assert all(ts > (current_time - 3600) for ts in recent)


def test_rate_limit_reset_after_hour(test_policy, test_event):
    """Test that rate limit resets after an hour."""
    test_policy.max_executions_per_hour = 1
    test_policy.cool_down_seconds = 0
    engine = PolicyEngine(policies=[test_policy])

    policy_key = f"{test_policy.name}_{test_event.component}"
    current_time = time.time()

    # Set up a timestamp from 90 minutes ago (outside the 1-hour window)
    old_timestamp = current_time - 5400  # 90 minutes ago
    engine.execution_timestamps[policy_key] = [old_timestamp]

    # At current_time, the old timestamp should be pruned (>3600 seconds old)
    # So _is_rate_limited should return False
    is_limited = engine._is_rate_limited(policy_key, test_policy, current_time)
    
    # The old timestamp should be pruned, leaving us with 0 timestamps
    # So we should NOT be rate limited
    assert is_limited is False, "Should not be rate limited after old timestamp is pruned"

    # Verify that the old timestamp was indeed pruned
    assert len(engine.execution_timestamps[policy_key]) == 0


def test_deduplication_of_actions(test_policy, test_event):
    """Test that duplicate actions are deduplicated."""
    # Create two policies with same action
    policy1 = HealingPolicy(
        name="policy1",
        conditions=[PolicyCondition(metric="latency_p99", operator="gt", threshold=300)],
        actions=[HealingAction.RESTART_CONTAINER, HealingAction.ALERT_TEAM],
        priority=1,
        cool_down_seconds=0,
        max_executions_per_hour=10,
    )
    policy2 = HealingPolicy(
        name="policy2",
        conditions=[PolicyCondition(metric="latency_p99", operator="gt", threshold=350)],
        actions=[HealingAction.RESTART_CONTAINER],  # Duplicate action
        priority=2,
        cool_down_seconds=0,
        max_executions_per_hour=10,
    )

    engine = PolicyEngine(policies=[policy1, policy2])

    actions = engine.evaluate_policies(test_event)

    # Count occurrences of RESTART_CONTAINER
    restart_count = actions.count(HealingAction.RESTART_CONTAINER)
    assert restart_count == 1, "Duplicate actions should be deduplicated"

    # Should have both RESTART_CONTAINER and ALERT_TEAM
    assert HealingAction.RESTART_CONTAINER in actions
    assert HealingAction.ALERT_TEAM in actions


def test_thread_safety_of_policy_evaluation(test_policy, test_event):
    """Test thread-safety during concurrent policy evaluations."""
    engine = PolicyEngine(policies=[test_policy])

    results = []
    errors = []

    def evaluate_in_thread():
        try:
            actions = engine.evaluate_policies(test_event)
            results.append(actions)
        except Exception as e:
            errors.append(e)

    # Create 10 concurrent threads
    threads = [threading.Thread(target=evaluate_in_thread) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Errors in threads: {errors}"
    assert len(results) == 10


def test_cooldown_and_rate_limit_interaction(test_policy, test_event):
    """Test that cooldown and rate limit work together."""
    test_policy.cool_down_seconds = 30
    test_policy.max_executions_per_hour = 2

    engine = PolicyEngine(policies=[test_policy])

    # First execution
    actions1 = engine.evaluate_policies(test_event)
    assert HealingAction.RESTART_CONTAINER in actions1

    # Cooldown blocks second attempt
    actions2 = engine.evaluate_policies(test_event)
    assert HealingAction.NO_ACTION in actions2

    # After cooldown expiration, second execution allowed
    policy_key = f"{test_policy.name}_{test_event.component}"
    engine.last_execution[policy_key] = time.time() - 31

    actions3 = engine.evaluate_policies(test_event)
    assert HealingAction.RESTART_CONTAINER in actions3

    # After cooldown again, rate limit should block third attempt
    engine.last_execution[policy_key] = time.time() - 31

    actions4 = engine.evaluate_policies(test_event)
    assert HealingAction.NO_ACTION in actions4


def test_no_action_when_conditions_unmet(test_policy, test_event):
    """Test that NO_ACTION returned when policy conditions not met."""
    # Create event that doesn't trigger policy (latency < 300)
    non_trigger_event = ReliabilityEvent(
        component="test-service",
        latency_p99=200,  # < 300, doesn't trigger
        error_rate=0.05,
        throughput=1000,
    )

    engine = PolicyEngine(policies=[test_policy])
    actions = engine.evaluate_policies(non_trigger_event)

    assert actions == [HealingAction.NO_ACTION]


def test_cooldown_history_cleanup(test_policy, test_event):
    """Test that cooldown history is cleaned up to prevent memory leak."""
    engine = PolicyEngine(policies=[test_policy], max_cooldown_history=3)

    # Create many events with different components
    for i in range(5):
        event_copy = ReliabilityEvent(
            component=f"service-{i}",
            latency_p99=400,
            error_rate=0.05,
            throughput=1000,
        )
        engine.evaluate_policies(event_copy)

    # Should only keep latest 3
    assert len(engine.last_execution) <= 3
