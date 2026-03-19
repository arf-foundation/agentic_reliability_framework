# agentic_reliability_framework/infrastructure/__init__.py
"""
ARF Infrastructure Governance Module
OSS Edition - Advisory analysis for Azure infrastructure
"""

import os
from agentic_reliability_framework.core.config.constants import MAX_POLICY_VIOLATIONS

from .intents import (
    ProvisionResourceIntent,
    DeployConfigurationIntent,
    GrantAccessIntent,
    ResourceType,
    Environment,
    PermissionLevel,
)
from .policies import (
    Policy,
    RegionAllowedPolicy,
    ResourceTypeRestrictedPolicy,
    MaxPermissionLevelPolicy,
    CostThresholdPolicy,
    PolicyEvaluator,
)
from .cost_estimator import CostEstimator
from .risk_engine import RiskEngine, RiskFactor
from .healing_intent import (
    HealingIntent,
    IntentSource,
    IntentStatus,
    RecommendedAction,
    create_infrastructure_healing_intent,
)
from .azure.azure_simulator import AzureInfrastructureSimulator


def validate_infrastructure_config(config: dict) -> dict:
    """
    Validate infrastructure‑specific configuration.

    Args:
        config: Dictionary with infrastructure config (e.g., pricing_file, policy definitions)

    Returns:
        Dictionary with keys:
            - valid: bool
            - warnings: list of warning messages
            - errors: list of error messages (if invalid)
    """
    warnings = []
    errors = []

    # Check pricing file existence
    pricing_file = config.get("pricing_file")
    if pricing_file and not os.path.exists(pricing_file):
        warnings.append(f"Pricing file not found: {pricing_file}")

    # Check policy limits
    max_policies = config.get("max_policies")
    if max_policies and max_policies > MAX_POLICY_VIOLATIONS:
        warnings.append(
            f"max_policies ({max_policies}) exceeds OSS limit {MAX_POLICY_VIOLATIONS}"
        )

    # (Add more infrastructure‑specific checks as needed)

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }

from .governance_loop import GovernanceLoop

__all__ = [
    # Intents
    "ProvisionResourceIntent",
    "DeployConfigurationIntent",
    "GrantAccessIntent",
    "ResourceType",
    "Environment",
    "PermissionLevel",

    # Policies
    "Policy",
    "RegionAllowedPolicy",
    "ResourceTypeRestrictedPolicy",
    "MaxPermissionLevelPolicy",
    "CostThresholdPolicy",
    "PolicyEvaluator",

    # Cost
    "CostEstimator",

    # Risk
    "RiskEngine",
    "RiskFactor",

    # HealingIntent
    "HealingIntent",
    "create_infrastructure_healing_intent",
    "IntentSource",
    "IntentStatus",
    "RecommendedAction",

    # Simulator
    "AzureInfrastructureSimulator",

    # Validation
    "validate_infrastructure_config",

    # Governance Loop
    "GovernanceLoop",
]
