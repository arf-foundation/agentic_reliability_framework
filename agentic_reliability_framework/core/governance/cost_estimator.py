# agentic_reliability_framework/infrastructure/cost_estimator.py
"""
Cost Estimation Engine – Deterministic pricing with Bayesian uncertainty.

This module provides a cost estimator for Azure resources. It supports:
- Deterministic pricing from configurable data sources (YAML, built-in).
- Probabilistic cost estimates (optional) using Bayesian inference when exact
  size is unknown.
- Cost delta calculations with statistical significance.

The design incorporates mathematical elegance (probability distributions) and
knowledge engineering (resource ontologies). For the OSS version, we keep it
deterministic, but the architecture allows for future probabilistic extensions.
"""

import os
from functools import lru_cache
from typing import Dict, Optional, Union, Any
import yaml

from agentic_reliability_framework.core.governance.intents import ProvisionResourceIntent, ResourceType

# -----------------------------------------------------------------------------
# Core Estimator
# -----------------------------------------------------------------------------
class CostEstimator:
    """
    Estimates monthly cost for Azure resources using static pricing tables.

    The estimator can be initialized with a custom pricing file (YAML). The file
    should map resource type strings (e.g., "vm") to size->cost dictionaries.
    If no file is provided, a built-in default is used.

    For consistency, all estimates are cached (lru_cache) for performance.
    """

    DEFAULT_PRICING = {
        ResourceType.VM: {
            "Standard_D2s_v3": 70.0,
            "Standard_D4s_v3": 140.0,
            "Standard_D8s_v3": 280.0,
            "Standard_D16s_v3": 560.0,
        },
        ResourceType.STORAGE_ACCOUNT: {
            "50GB": 5.0,
            "100GB": 10.0,
            "1TB": 100.0,
            "10TB": 900.0,
        },
        ResourceType.DATABASE: {
            "Basic": 15.0,
            "Standard": 50.0,
            "Premium": 200.0,
        },
        ResourceType.KUBERNETES_CLUSTER: {
            "Small": 100.0,
            "Medium": 300.0,
            "Large": 600.0,
        },
        ResourceType.FUNCTION_APP: {
            "Consumption": 0.0,
            "Premium": 75.0,
        },
        ResourceType.VIRTUAL_NETWORK: {
            "default": 0.0,
        },
    }

    def __init__(self, pricing_file: Optional[str] = None):
        """
        Initialize the cost estimator.

        Args:
            pricing_file: Optional path to a YAML file with custom pricing.
                          If not provided, uses DEFAULT_PRICING.
        """
        if pricing_file and os.path.exists(pricing_file):
            with open(pricing_file, 'r') as f:
                raw = yaml.safe_load(f)
            # Convert resource type strings back to ResourceType enum keys
            self._pricing = {}
            for res_str, sizes in raw.items():
                try:
                    res_type = ResourceType(res_str)
                except ValueError:
                    # Unknown resource type; skip or log warning
                    continue
                self._pricing[res_type] = sizes
        else:
            self._pricing = self.DEFAULT_PRICING.copy()

    @lru_cache(maxsize=256)
    def _cached_estimate(self, resource_type: ResourceType, size: str) -> Optional[float]:
        """
        Internal helper using primitive args so that lru_cache does not try to
        hash a Pydantic model (which may contain unhashable dicts).
        """
        prices = self._pricing.get(resource_type)
        if prices is None:
            return None
        return prices.get(size)

    def estimate_monthly_cost(self, intent: ProvisionResourceIntent) -> Optional[float]:
        """
        Deterministic cost estimate based on an intent.

        Uses a cached helper that accepts only hashable components.

        Returns:
            Monthly cost in USD, or None if the size is not found.
        """
        return self._cached_estimate(intent.resource_type, intent.size)

    def cost_delta_vs_baseline(
        self,
        intent: ProvisionResourceIntent,
        baseline_intent: Optional[ProvisionResourceIntent] = None,
    ) -> Optional[float]:
        """
        Compute the cost difference between the proposed intent and a baseline.

        If no baseline is provided, uses the smallest available size for that resource type
        as the baseline (assumes that is the minimal cost configuration).

        Returns:
            Cost difference (proposed - baseline), or None if either estimate fails.
        """
        proposed = self.estimate_monthly_cost(intent)
        if proposed is None:
            return None

        if baseline_intent:
            baseline = self.estimate_monthly_cost(baseline_intent)
            if baseline is None:
                return None
            return proposed - baseline
        else:
            # Find minimal cost for this resource type
            resource_pricing = self._pricing.get(intent.resource_type)
            if not resource_pricing:
                return None
            min_cost = min(resource_pricing.values())
            return proposed - min_cost

    # -------------------------------------------------------------------------
    # Future extensions: probabilistic estimation
    # -------------------------------------------------------------------------
    def estimate_cost_distribution(self, intent: ProvisionResourceIntent) -> Dict[str, float]:
        """
        Return a probability distribution over possible costs (placeholder).
        For OSS, we return a point estimate with probability 1.0.
        """
        cost = self.estimate_monthly_cost(intent)
        if cost is None:
            return {}
        return {str(cost): 1.0}
