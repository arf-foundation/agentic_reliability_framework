"""
Bayesian Risk Scoring Engine – Conjugate priors (online) + HMC (offline) + Hyperpriors (optional).

This engine combines two or three Bayesian approaches:
1. Conjugate Beta priors for each action category – fast online updates.
2. A Hamiltonian Monte Carlo logistic regression (using NUTS) that learns
   complex patterns (time of day, user role, environment) from historical data.
3. Optional hyperpriors (hierarchical Beta) that share statistical strength across categories.

At runtime, the final risk score can be a weighted average of:
   - conjugate mean (simple)
   - hyperprior posterior mean (if enabled)
   - HMC prediction (if available)

The weights are determined by the amount of data available.
"""

import os
import json
import threading
import logging
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum

import pymc as pm
import arviz as az
from sklearn.preprocessing import StandardScaler

# Pyro and torch are optional; import them conditionally
try:
    import pyro
    import pyro.distributions as dist
    from pyro.infer import SVI, Trace_ELBO, Predictive
    import torch
    PYRO_AVAILABLE = True
except ImportError:
    PYRO_AVAILABLE = False
    # Create dummy modules to avoid NameError if someone tries to use them
    pyro = None
    dist = None
    SVI = None
    Trace_ELBO = None
    Predictive = None
    torch = None
    logger = logging.getLogger(__name__)
    logger.warning("Pyro not installed; hyperprior functionality will be disabled.")

from agentic_reliability_framework.core.governance.intents import (
    InfrastructureIntent,
    ProvisionResourceIntent,
    GrantAccessIntent,
    DeployConfigurationIntent,
    ResourceType,
    PermissionLevel,
    Environment,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Action Categories
# =============================================================================
class ActionCategory(str, Enum):
    DATABASE = "database"
    NETWORK = "network"
    COMPUTE = "compute"
    SECURITY = "security"
    DEFAULT = "default"


# =============================================================================
# Conjugate Beta Priors (Online Model)
# =============================================================================
PRIORS = {
    ActionCategory.DATABASE: (1.5, 8.0),   # Beta(α, β) – more pessimistic
    ActionCategory.NETWORK: (1.2, 10.0),
    ActionCategory.COMPUTE: (1.0, 12.0),
    ActionCategory.SECURITY: (2.0, 10.0),
    ActionCategory.DEFAULT: (1.0, 10.0),
}

class BetaStore:
    """Thread‑safe store of Beta posterior parameters per category."""
    def __init__(self):
        self._data: Dict[ActionCategory, Tuple[float, float]] = {
            cat: PRIORS[cat] for cat in ActionCategory
        }
        self._lock = threading.RLock()

    def get(self, category: ActionCategory) -> Tuple[float, float]:
        with self._lock:
            return self._data[category]

    def update(self, category: ActionCategory, success: bool):
        with self._lock:
            alpha, beta = self._data[category]
            if success:
                alpha += 1
            else:
                beta += 1
            self._data[category] = (alpha, beta)


# =============================================================================
# Hyperprior Beta Model (Hierarchical)
# =============================================================================
class HyperpriorBetaStore:
    """
    Hierarchical Beta model where each category's parameters share a common hyperprior.
    Uses Pyro for variational inference (SVI) to learn the posterior.
    """
    def __init__(self, num_categories: int = len(ActionCategory)):
        self.num_categories = num_categories
        self.category_names = list(ActionCategory)
        self.category_indices = {cat: i for i, cat in enumerate(self.category_names)}
        self._history: List[Tuple[int, float]] = []  # (category_idx, success)
        self._initialized = False
        self._lock = threading.RLock()
        if PYRO_AVAILABLE:
            self._init_model()
        else:
            logger.warning("HyperpriorBetaStore: Pyro not available; store will be a no-op.")

    def _init_model(self):
        """Initialize Pyro parameters if available."""
        if not PYRO_AVAILABLE:
            return

        # Hyperpriors (Gamma for alpha, beta)
        self.alpha0 = pyro.param("alpha0", torch.tensor(2.0), constraint=dist.constraints.positive)
        self.beta0 = pyro.param("beta0", torch.tensor(2.0), constraint=dist.constraints.positive)
        # Category‑specific parameters
        self.p_alpha = pyro.param("p_alpha", torch.ones(self.num_categories) * 2.0, constraint=dist.constraints.positive)
        self.p_beta = pyro.param("p_beta", torch.ones(self.num_categories) * 2.0, constraint=dist.constraints.positive)
        self._initialized = True

    def model(self, observations=None):
        """Pyro model for hyperprior Beta."""
        if not PYRO_AVAILABLE or not self._initialized:
            return

        # Global hyperprior (concentration parameters)
        alpha0 = pyro.sample("alpha0", dist.Gamma(2.0, 1.0))
        beta0 = pyro.sample("beta0", dist.Gamma(2.0, 1.0))

        with pyro.plate("categories", self.num_categories):
            # Category‑specific success probabilities drawn from Beta(alpha0, beta0)
            p = pyro.sample("p", dist.Beta(alpha0, beta0))

        if observations is not None:
            cat_idx = torch.tensor([obs[0] for obs in observations])
            successes = torch.tensor([obs[1] for obs in observations])
            with pyro.plate("data", len(observations)):
                pyro.sample("obs", dist.Bernoulli(p[cat_idx]), obs=successes)

    def guide(self, observations=None):
        """Variational guide for hyperprior model."""
        if not PYRO_AVAILABLE or not self._initialized:
            return

        # Variational parameters for hyperpriors
        alpha0_q = pyro.param("alpha0_q", torch.tensor(2.0), constraint=dist.constraints.positive)
        beta0_q = pyro.param("beta0_q", torch.tensor(2.0), constraint=dist.constraints.positive)
        pyro.sample("alpha0", dist.Gamma(alpha0_q, 1.0))
        pyro.sample("beta0", dist.Gamma(beta0_q, 1.0))

        with pyro.plate("categories", self.num_categories):
            p_alpha = pyro.param("p_alpha", torch.ones(self.num_categories) * 2.0, constraint=dist.constraints.positive)
            p_beta = pyro.param("p_beta", torch.ones(self.num_categories) * 2.0, constraint=dist.constraints.positive)
            pyro.sample("p", dist.Beta(p_alpha, p_beta))

    def update(self, category: ActionCategory, success: bool):
        """Record an observation and optionally run SVI."""
        if not PYRO_AVAILABLE or not self._initialized:
            return

        cat_idx = self.category_indices[category]
        with self._lock:
            self._history.append((cat_idx, 1.0 if success else 0.0))
            # Run a few steps of SVI if we have enough data
            if len(self._history) > 5:
                self._run_svi(steps=10)

    def _run_svi(self, steps=50):
        """Run variational inference on observed data."""
        if not PYRO_AVAILABLE or not self._initialized or len(self._history) == 0:
            return

        optimizer = pyro.optim.Adam({"lr": 0.01})
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())
        for step in range(steps):
            loss = svi.step(self._history)
            if step % 10 == 0:
                logger.debug(f"Hyperprior SVI step {step}, loss: {loss}")

    def get_risk_summary(self, category: ActionCategory) -> Dict[str, float]:
        """Return posterior predictive risk metrics for a category."""
        if not PYRO_AVAILABLE or not self._initialized or len(self._history) == 0:
            # Fall back to uniform prior
            return {"mean": 0.5, "p5": 0.1, "p50": 0.5, "p95": 0.9}

        cat_idx = self.category_indices[category]
        # Generate posterior samples for p[cat_idx]
        predictive = Predictive(self.model, guide=self.guide, num_samples=500)
        samples = predictive(self._history)
        p_samples = samples["p"][:, cat_idx].detach().numpy()

        return {
            "mean": float(p_samples.mean()),
            "p5": float(np.percentile(p_samples, 5)),
            "p50": float(np.percentile(p_samples, 50)),
            "p95": float(np.percentile(p_samples, 95))
        }


# =============================================================================
# HMC Model (Offline Bayesian Logistic Regression)
# =============================================================================
class HMCModel:
    """[unchanged from original]"""
    def __init__(self, model_path: str = "hmc_model.json"):
        self.model_path = model_path
        self.coefficients: Optional[Dict[str, float]] = None
        self.feature_scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[List[str]] = None
        self.is_ready = False
        self._load()

    def _load(self):
        """Load pre‑trained coefficients from JSON."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "r") as f:
                    data = json.load(f)
                self.coefficients = data.get("coefficients", {})
                self.feature_names = data.get("feature_names", [])
                scaler_params = data.get("scaler")
                if scaler_params:
                    self.feature_scaler = StandardScaler()
                    self.feature_scaler.mean_ = np.array(scaler_params["mean"])
                    self.feature_scaler.scale_ = np.array(scaler_params["scale"])
                self.is_ready = True
                logger.info(f"HMC model loaded from {self.model_path}")
            except Exception as e:
                logger.warning(f"Could not load HMC model: {e}")

    def _save(self, trace, feature_names, scaler):
        """Save posterior means as coefficients to JSON."""
        coeffs = {}
        for var in trace.posterior.data_vars:
            if var.startswith("beta_") or var == "alpha":
                mean_val = float(trace.posterior[var].mean().values)
                coeffs[var] = mean_val
        data = {
            "coefficients": coeffs,
            "feature_names": feature_names,
            "scaler": {
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
            }
        }
        with open(self.model_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"HMC model saved to {self.model_path}")

    def predict(self, intent: InfrastructureIntent, context: Dict[str, Any]) -> Optional[float]:
        """Return HMC‑predicted probability (unchanged)."""
        if not self.is_ready or self.coefficients is None or self.feature_names is None:
            return None

        hour = datetime.datetime.now().hour
        sin_hour = np.sin(2 * np.pi * hour / 24)
        cos_hour = np.cos(2 * np.pi * hour / 24)
        env_prod = 1.0 if hasattr(intent, "environment") and intent.environment == "prod" else 0.0
        user_role = 0.0  # placeholder

        cat = self._categorize_intent(intent)
        feature_dict = {
            'sin_hour': sin_hour,
            'cos_hour': cos_hour,
            'env_prod': env_prod,
            'user_role': user_role,
            'cat_database': 1.0 if cat == ActionCategory.DATABASE else 0.0,
            'cat_network': 1.0 if cat == ActionCategory.NETWORK else 0.0,
            'cat_compute': 1.0 if cat == ActionCategory.COMPUTE else 0.0,
            'cat_security': 1.0 if cat == ActionCategory.SECURITY else 0.0,
            'cat_default': 1.0 if cat == ActionCategory.DEFAULT else 0.0,
        }

        x = np.array([feature_dict.get(name, 0.0) for name in self.feature_names]).reshape(1, -1)

        if self.feature_scaler:
            x = self.feature_scaler.transform(x)

        lin = self.coefficients.get('alpha', 0.0)
        for i, name in enumerate(self.feature_names):
            beta_name = f'beta_{name}'
            if beta_name in self.coefficients:
                lin += self.coefficients[beta_name] * x[0, i]

        prob = 1.0 / (1.0 + np.exp(-lin))
        return float(prob)

    def _categorize_intent(self, intent: InfrastructureIntent) -> ActionCategory:
        """Helper to categorize intent."""
        if isinstance(intent, ProvisionResourceIntent):
            if intent.resource_type in (ResourceType.DATABASE, ResourceType.STORAGE_ACCOUNT):
                return ActionCategory.DATABASE
            elif intent.resource_type == ResourceType.VM:
                return ActionCategory.COMPUTE
            elif intent.resource_type == ResourceType.VIRTUAL_NETWORK:
                return ActionCategory.NETWORK
        elif isinstance(intent, GrantAccessIntent):
            return ActionCategory.SECURITY
        elif isinstance(intent, DeployConfigurationIntent):
<<<<<<< HEAD
            # configuration deployments are generic; classify as DEFAULT by default
=======
            if "database" in intent.service_name.lower():
                return ActionCategory.DATABASE
            # For any other configuration change, use the baseline (DEFAULT) category
>>>>>>> adf837024fd6d06c8d3dd61a120b662cc49a2c77
            return ActionCategory.DEFAULT
        return ActionCategory.DEFAULT


# =============================================================================
# Helper: categorize intent (public version)
# =============================================================================
def categorize_intent(intent: InfrastructureIntent) -> ActionCategory:
    if isinstance(intent, ProvisionResourceIntent):
        if intent.resource_type in (ResourceType.DATABASE, ResourceType.STORAGE_ACCOUNT):
            return ActionCategory.DATABASE
        elif intent.resource_type == ResourceType.VM:
            return ActionCategory.COMPUTE
        elif intent.resource_type == ResourceType.VIRTUAL_NETWORK:
            return ActionCategory.NETWORK
    elif isinstance(intent, GrantAccessIntent):
        return ActionCategory.SECURITY
    elif isinstance(intent, DeployConfigurationIntent):
<<<<<<< HEAD
        # configuration deployments should count as DEFAULT category
=======
        if "database" in intent.service_name.lower():
            return ActionCategory.DATABASE
        # For any other configuration change, use the baseline (DEFAULT) category
>>>>>>> adf837024fd6d06c8d3dd61a120b662cc49a2c77
        return ActionCategory.DEFAULT
    return ActionCategory.DEFAULT


# =============================================================================
# Combined Risk Engine with Hyperprior Option
# =============================================================================
class RiskEngine:
    """
    Bayesian risk engine that combines online conjugate priors, optional hyperpriors,
    and offline HMC.

    The final risk score is a weighted average of up to three components:
        risk = w_conj * conjugate_mean + w_hyper * hyperprior_mean + w_hmc * hmc_prediction
    where weights are determined by data availability.

    Hyperpriors provide hierarchical shrinkage, sharing statistical strength across categories.
    """

    def __init__(
        self,
        hmc_model_path: str = "hmc_model.json",
        n0: int = 1000,
        use_hyperpriors: bool = False,
        hyperprior_weight: float = 0.3
    ):
        """
        Args:
            hmc_model_path: Path to HMC model JSON.
            n0: Threshold for HMC confidence (incidents needed for full weight).
            use_hyperpriors: Whether to enable the hyperprior model.
            hyperprior_weight: Base weight for hyperprior when data is available.
        """
        self.beta_store = BetaStore()
        self.hmc_model = HMCModel(hmc_model_path)
        self.hyperprior_store = HyperpriorBetaStore() if use_hyperpriors and PYRO_AVAILABLE else None
        self.n0 = n0
        self.hyperprior_weight = hyperprior_weight
        self.total_incidents = 0
        self._lock = threading.RLock()
        self.use_hyperpriors = use_hyperpriors and PYRO_AVAILABLE

        if use_hyperpriors and not PYRO_AVAILABLE:
            logger.warning("Hyperpriors requested but Pyro not installed; disabling.")

    def calculate_risk(
        self,
        intent: InfrastructureIntent,
        cost_estimate: Optional[float],
        policy_violations: List[str],
    ) -> Tuple[float, str, Dict[str, float]]:
        """
        Compute combined Bayesian risk score with optional hyperpriors.
        """
        category = categorize_intent(intent)
        alpha, beta = self.beta_store.get(category)
        conjugate_risk = alpha / (alpha + beta)

        # Hyperprior risk (if enabled)
        hyper_risk_summary = None
        if self.use_hyperpriors and self.hyperprior_store:
            hyper_risk_summary = self.hyperprior_store.get_risk_summary(category)
            hyper_risk = hyper_risk_summary["mean"]
        else:
            hyper_risk = None

        # HMC prediction (if available)
        hmc_risk = self.hmc_model.predict(intent, {"cost": cost_estimate, "violations": policy_violations})

        # Determine weights based on data availability
        with self._lock:
            n = self.total_incidents

        # Weight calculation
        if hmc_risk is None and hyper_risk is None:
            # Only conjugate available
            final_risk = conjugate_risk
            weights = {"conjugate": 1.0, "hyper": 0.0, "hmc": 0.0}
        elif hmc_risk is None:
            # Conjugate + hyperprior
            weight_hyper = min(self.hyperprior_weight, n / 100) if n > 0 else 0
            weight_hyper = min(weight_hyper, 0.5)  # Cap hyperprior influence
            weight_conj = 1.0 - weight_hyper
            final_risk = weight_conj * conjugate_risk + weight_hyper * hyper_risk
            weights = {"conjugate": weight_conj, "hyper": weight_hyper, "hmc": 0.0}
        elif hyper_risk is None:
            # Conjugate + HMC (original behavior)
            weight_hmc = min(1.0, n / self.n0)
            final_risk = (1 - weight_hmc) * conjugate_risk + weight_hmc * hmc_risk
            weights = {"conjugate": 1 - weight_hmc, "hyper": 0.0, "hmc": weight_hmc}
        else:
            # All three available
            weight_hmc = min(0.6, n / self.n0)  # Cap HMC influence
            weight_hyper = min(self.hyperprior_weight, n / 100) * (1 - weight_hmc)
            weight_hyper = min(weight_hyper, 0.3)  # Cap hyperprior
            weight_conj = 1.0 - weight_hmc - weight_hyper
            final_risk = weight_conj * conjugate_risk + weight_hyper * hyper_risk + weight_hmc * hmc_risk
            weights = {"conjugate": weight_conj, "hyper": weight_hyper, "hmc": weight_hmc}

        # Context multiplier
        multiplier = self._context_multiplier(intent)
        final_risk = min(final_risk * multiplier, 1.0)

        # Build explanation
        explanation_parts = [
            f"Bayesian risk for category '{category.value}': "
            f"conjugate mean = {conjugate_risk:.3f} (α={alpha:.1f}, β={beta:.1f})."
        ]
        if hyper_risk is not None:
            explanation_parts.append(
                f"Hyperprior mean = {hyper_risk:.3f} (weight={weights['hyper']:.2f})."
            )
        if hmc_risk is not None:
            explanation_parts.append(
                f"HMC prediction = {hmc_risk:.3f} (weight={weights['hmc']:.2f})."
            )
        explanation_parts.append(f"Context multiplier: {multiplier:.2f}. Final risk: {final_risk:.3f}.")
        explanation = " ".join(explanation_parts)

        contributions = {
            "conjugate_mean": conjugate_risk,
            "conjugate_alpha": alpha,
            "conjugate_beta": beta,
            "hyper_mean": hyper_risk if hyper_risk is not None else 0.0,
            "hmc_prediction": hmc_risk if hmc_risk is not None else 0.0,
            "weights": weights,
            "context_multiplier": multiplier,
        }
        if hyper_risk_summary:
            contributions["hyper_summary"] = hyper_risk_summary

        return final_risk, explanation, contributions

    def update_outcome(self, intent: InfrastructureIntent, success: bool) -> None:
        """
        Update all models with the outcome of an executed intent.
        """
        category = categorize_intent(intent)
        self.beta_store.update(category, success)
        if self.use_hyperpriors and self.hyperprior_store:
            self.hyperprior_store.update(category, success)
        with self._lock:
            self.total_incidents += 1

    def train_hmc(self, incidents_df: pd.DataFrame) -> None:
        """
        Train the HMC model on historical incident data.
        """
        self.hmc_model.train(incidents_df)

    def _context_multiplier(self, intent: InfrastructureIntent) -> float:
        """Compute multiplier based on environment, user role, time, etc."""
        mult = 1.0
<<<<<<< HEAD
        # environment is a literal string ("prod"/"dev" etc.)
        if hasattr(intent, "environment") and intent.environment == "prod":
            mult *= 1.5
        # some intents (e.g. DeployConfigurationIntent) use `deployment_target`
=======
        # Check for production environment in various intent fields
        if hasattr(intent, "environment") and intent.environment == "prod":
            mult *= 1.5
>>>>>>> adf837024fd6d06c8d3dd61a120b662cc49a2c77
        elif hasattr(intent, "deployment_target") and intent.deployment_target == "prod":
            mult *= 1.5
        # Additional factors could be added
        return mult


# For backward compatibility
class RiskFactor:
    """Placeholder for compatibility – not used."""
    def __init__(self, *args, **kwargs):
        pass
