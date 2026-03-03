"""
Hamiltonian Monte Carlo Learner for Probabilistic Risk Assessment.
Uses PyMC's No-U-Turn Sampler (NUTS) to model the probability of critical incidents.
"""

import logging
import pymc as pm
import numpy as np
import arviz as az
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple

from agentic_reliability_framework.core.config.constants import MAX_RISK_FACTORS

logger = logging.getLogger(__name__)


class HMCRiskLearner:
    """
    Bayesian logistic regression model using NUTS to estimate incident probability.

    Features: latency, error_rate, throughput, cpu_util, memory_util (optional).
    Target: binary indicator of critical/high severity incident.

    The model can be trained on historical incident data and then used to predict
    the probability of a new event being critical.
    """

    def __init__(self):
        self.model: Optional[pm.Model] = None
        self.trace: Optional[az.InferenceData] = None
        self.is_ready = False
        self.feature_names: List[str] = []
        self._feature_scales: Dict[str, Tuple[float, float]] = {}  # (mean, std) for scaling

    def _prepare_features(self, incident_data: List[Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert list of incident dictionaries to feature matrix and target vector.
        Assumes each dict has 'severity' (or 'target') key indicating if incident was critical.
        For demo, we'll create a synthetic target if not present.
        """
        df = pd.DataFrame(incident_data)
        # If no target, create a synthetic one for demonstration (replace with real labels in production)
        if 'target' not in df.columns:
            # For demo, label as critical if latency > 300 or error_rate > 0.15
            df['target'] = ((df.get('latency_p99', 0) > 300) |
                            (df.get('error_rate', 0) > 0.15)).astype(int)

        # Select feature columns
        feature_cols = ['latency_p99', 'error_rate', 'throughput', 'cpu_util', 'memory_util']
        # Keep only those present
        available_features = [c for c in feature_cols if c in df.columns]
        X = df[available_features].fillna(0).values.astype(float)
        y = df['target'].values.astype(int)

        # Store feature names and compute scaling (for later prediction)
        self.feature_names = available_features
        self._feature_scales = {name: (df[name].mean(), df[name].std()) for name in available_features}

        # Scale features (important for HMC convergence)
        X_scaled = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        # Replace NaNs (if std=0) with 0
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        return X_scaled, y

    def build_model(self, X: np.ndarray, y: np.ndarray):
        """Build the Bayesian logistic regression model."""
        n_features = X.shape[1]
        with pm.Model() as self.model:
            # Priors for coefficients (regularizing)
            alpha = pm.Normal('alpha', mu=0, sigma=2)  # intercept
            beta = pm.Normal('beta', mu=0, sigma=2, shape=n_features)

            # Linear predictor
            mu = alpha + pm.math.dot(X, beta)

            # Likelihood (Bernoulli for binary outcome)
            likelihood = pm.Bernoulli('y', logit_p=mu, observed=y)

            # Compute risk probability as deterministic
            pm.Deterministic('risk_prob', pm.math.sigmoid(mu))

        logger.info("Bayesian logistic regression model built with %d features.", n_features)

    def train(self, incident_data: List[Dict[str, float]],
              draws: int = 1000,
              tune: int = 1000,
              chains: int = 2,
              target_accept: float = 0.9):
        """
        Train the model on historical incident data using NUTS.

        Args:
            incident_data: List of dictionaries, each containing metrics and optionally 'target'.
            draws: Number of posterior draws.
            tune: Number of tuning steps (adaptation).
            chains: Number of MCMC chains.
            target_accept: Target acceptance rate for NUTS (higher values lead to smaller step sizes).
        """
        if len(incident_data) < 10:
            logger.warning("Insufficient data for training (need at least 10 incidents).")
            self.is_ready = False
            return

        X, y = self._prepare_features(incident_data)
        self.build_model(X, y)

        with self.model:
            logger.info("Sampling with NUTS (target_accept=%.2f)...", target_accept)
            # Explicitly use NUTS sampler with specified target acceptance
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                step=pm.NUTS(target_accept=target_accept),
                progressbar=False,
                return_inferencedata=True,
                idata_kwargs={'log_likelihood': True}  # optional, for later model comparison
            )
            self.is_ready = True
            logger.info("NUTS sampling completed.")

    def posterior_predictive(self, metrics: Dict[str, float]) -> np.ndarray:
        """
        Generate posterior samples of the risk probability for a new observation.

        Args:
            metrics: Dictionary with keys like 'latency_p99', 'error_rate', etc.

        Returns:
            Array of risk probability samples (length = number of posterior draws).
        """
        if not self.is_ready or self.trace is None:
            logger.warning("Model not trained. Returning default risk [0.5].")
            return np.array([0.5])

        # Build feature vector for new observation
        x = []
        for name in self.feature_names:
            val = metrics.get(name, 0.0)
            mean, std = self._feature_scales.get(name, (0.0, 1.0))
            # Scale using training statistics
            scaled = (val - mean) / std if std != 0 else 0.0
            x.append(scaled)
        x = np.array(x).reshape(1, -1)

        # Extract posterior samples of coefficients
        alpha_samples = self.trace.posterior['alpha'].values.flatten()
        beta_samples = self.trace.posterior['beta'].values.reshape(-1, len(self.feature_names))

        # Compute linear predictor for each sample
        mu = alpha_samples + np.dot(beta_samples, x.T).flatten()
        # Convert to probability via sigmoid
        risk_probs = 1 / (1 + np.exp(-mu))
        return risk_probs

    def predict_risk_summary(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Return summary statistics of the posterior risk probability.
        """
        samples = self.posterior_predictive(metrics)
        return {
            'mean_risk': float(np.mean(samples)),
            'std_risk': float(np.std(samples)),
            'median_risk': float(np.median(samples)),
            'q05_risk': float(np.percentile(samples, 5)),
            'q95_risk': float(np.percentile(samples, 95)),
            'samples': samples.tolist()[:10]  # first 10 samples for inspection
        }

    def get_feature_importance(self) -> Dict[str, Any]:
        """
        Return summary of posterior coefficient distributions.
        """
        if not self.is_ready or self.trace is None:
            return {}
        coeffs = {}
        for i, name in enumerate(self.feature_names):
            beta_samples = self.trace.posterior['beta'].values[..., i].flatten()
            coeffs[name] = {
                'mean': float(np.mean(beta_samples)),
                'std': float(np.std(beta_samples)),
                'q05': float(np.percentile(beta_samples, 5)),
                'q95': float(np.percentile(beta_samples, 95)),
                'p_gt_0': float((beta_samples > 0).mean())
            }
        return coeffs

    def predict(self, metrics: Dict[str, float]) -> float:
        """
        Predict the risk probability for a new observation (scalar prediction).

        This is the main entry point for integration with risk_engine.py.
        Returns a single scalar float in [0,1] representing the probability
        of a critical/high-severity incident based on the provided metrics.

        Args:
            metrics: Dictionary with optional keys 'latency_p99', 'error_rate',
                     'throughput', 'cpu_util', 'memory_util'. Missing keys default to 0.

        Returns:
            float: Risk probability in [0,1]. Returns 0.5 if model not trained.

        Example:
            >>> hmc = HMCRiskLearner()
            >>> hmc.train([...historical_data...])
            >>> risk = hmc.predict({
            ...     'latency_p99': 350,
            ...     'error_rate': 0.08,
            ...     'throughput': 900
            ... })
            >>> print(f"Risk score: {risk:.3f}")  # e.g., 0.623
        """
        if not self.is_ready:
            logger.debug("HMC model not trained; returning default risk 0.5")
            return 0.5

        try:
            samples = self.posterior_predictive(metrics)
            # Return the mean of the posterior distribution
            mean_risk = float(np.mean(samples))
            # Ensure result is in [0,1]
            return min(max(mean_risk, 0.0), 1.0)
        except Exception as e:
            logger.error(f"HMC predict failed: {e}, returning default 0.5")
            return 0.5
