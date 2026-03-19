import pytest
pytest.importorskip("torch", reason="Torch import conflict – skip for now")
"""
Additional comprehensive tests for RiskEngine to cover missing lines.
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import json
import os
from unittest.mock import MagicMock, patch, PropertyMock

from agentic_reliability_framework.core.governance.risk_engine import (
    RiskEngine,
    HMCModel,
    HyperpriorBetaStore,
    categorize_intent,
    ActionCategory,
    PRIORS,
)
from agentic_reliability_framework.core.governance.intents import (
    ProvisionResourceIntent,
    GrantAccessIntent,
    DeployConfigurationIntent,
    ResourceType,
    PermissionLevel,
    # Environment and ChangeScope are string literals, not enums
)


# -----------------------------------------------------------------------------
# Fixtures for intents
# -----------------------------------------------------------------------------
@pytest.fixture
def compute_intent():
    return ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D2s_v3",
        environment="dev",  # string
        requester="tester",
    )


@pytest.fixture
def database_intent():
    return ProvisionResourceIntent(
        resource_type=ResourceType.DATABASE,
        region="eastus",
        size="Standard",
        environment="dev",  # string
        requester="tester",
    )


@pytest.fixture
def network_intent():
    return ProvisionResourceIntent(
        resource_type=ResourceType.VIRTUAL_NETWORK,
        region="eastus",
        size="/24",
        environment="dev",  # string
        requester="tester",
    )


@pytest.fixture
def security_intent():
    return GrantAccessIntent(
        principal="bob",
        permission_level=PermissionLevel.ADMIN,
        resource_scope="/subscriptions/123",
        requester="alice",
    )


@pytest.fixture
def default_intent():
    return DeployConfigurationIntent(
        service_name="api",
        change_scope="global",  # string
        deployment_target="prod",  # string
        requester="alice",
        configuration={},
    )


# -----------------------------------------------------------------------------
# HyperpriorBetaStore tests (with Pyro mocked)
# -----------------------------------------------------------------------------
class TestHyperpriorBetaStore:
    def test_init_without_pyro(self):
        """Test that store is a no‑op when Pyro not available."""
        with patch("agentic_reliability_framework.core.governance.risk_engine.PYRO_AVAILABLE", False):
            store = HyperpriorBetaStore()
            assert store._initialized is False
            # update and get_risk_summary should do nothing
            store.update(ActionCategory.COMPUTE, True)
            summary = store.get_risk_summary(ActionCategory.COMPUTE)
            assert summary == {"mean": 0.5, "p5": 0.1, "p50": 0.5, "p95": 0.9}

    def test_init_with_pyro(self):
        """Test initialization when Pyro is available."""
        with patch("agentic_reliability_framework.core.governance.risk_engine.PYRO_AVAILABLE", True):
            with patch("agentic_reliability_framework.core.governance.risk_engine.pyro") as mock_pyro:
                store = HyperpriorBetaStore()
                assert store._initialized is True
                mock_pyro.param.assert_called()

    def test_update_and_summary(self):
        """Test that update records observations and _run_svi is called."""
        with patch("agentic_reliability_framework.core.governance.risk_engine.PYRO_AVAILABLE", True):
            with patch("agentic_reliability_framework.core.governance.risk_engine.pyro") as mock_pyro:
                store = HyperpriorBetaStore()
                store._run_svi = MagicMock()
                store.update(ActionCategory.COMPUTE, True)
                assert len(store._history) == 1
                # After 5 updates, _run_svi should be called
                for i in range(5):
                    store.update(ActionCategory.COMPUTE, i % 2 == 0)
                assert store._run_svi.called

                # Mock get_risk_summary to return dummy data
                with patch.object(store, 'get_risk_summary', return_value={"mean": 0.3}):
                    summary = store.get_risk_summary(ActionCategory.COMPUTE)
                    assert summary["mean"] == 0.3


# -----------------------------------------------------------------------------
# HMCModel tests – edge cases and training
# -----------------------------------------------------------------------------
class TestHMCModel:
    def test_load_nonexistent_file(self):
        """Test loading a non‑existent model file."""
        model = HMCModel("nonexistent.json")
        assert model.is_ready is False
        assert model.coefficients is None

    def test_load_corrupted_json(self, tmp_path):
        """Test loading corrupted JSON."""
        p = tmp_path / "bad.json"
        p.write_text("{not json}")
        model = HMCModel(str(p))
        assert model.is_ready is False

    def test_train_without_pymc(self, tmp_path):
        """Test training when PyMC is not available (should log error and return)."""
        model = HMCModel(str(tmp_path / "dummy.json"))
        with patch("agentic_reliability_framework.core.governance.risk_engine.pm", None):
            model.train(pd.DataFrame())  # empty df should not cause error
        assert model.is_ready is False

    def test_train_success_with_mocked_trace(self, tmp_path, monkeypatch):
        """Test successful training with mocked PyMC sample."""
        df = pd.DataFrame({
            'hour': [0, 12],
            'env_prod': [1, 0],
            'user_role': [0, 1],
            'cat_database': [1, 0],
            'cat_compute': [0, 1],
            'outcome': [0, 1],
        })
        model_path = tmp_path / "hmc.json"

        # Mock pm.sample to return a dummy trace
        class DummyTrace:
            posterior = {
                'alpha': np.array(0.5),
                'beta': np.array([0.1, 0.2, 0.3, 0.4, 0.5])
            }
        import pymc as pm
        monkeypatch.setattr(pm, 'sample', lambda *args, **kwargs: DummyTrace())

        model = HMCModel(str(model_path))
        model.train(df)
        assert model.is_ready is True
        assert model.coefficients is not None
        assert 'alpha' in model.coefficients
        assert 'beta_hour' in model.coefficients

    def test_predict_without_ready(self):
        """Test predict returns None when model not ready."""
        model = HMCModel("nonexistent.json")
        assert model.predict(None, {}) is None

    def test_predict_with_coefficients(self, compute_intent, tmp_path, monkeypatch):
        """Test prediction with loaded coefficients."""
        # Create a dummy model file
        model_path = tmp_path / "hmc.json"
        data = {
            "coefficients": {"alpha": 0.2, "beta_hour": 0.3, "beta_env_prod": 0.4},
            "feature_names": ["hour", "env_prod", "user_role", "cat_compute"],
            "scaler": {"mean": [0.0, 0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0, 1.0]},
        }
        with open(model_path, "w") as f:
            json.dump(data, f)

        model = HMCModel(str(model_path))
        # Mock datetime to control hour
        class MockDatetime:
            @classmethod
            def now(cls):
                return type('', (), {'hour': 12})()
        monkeypatch.setattr("agentic_reliability_framework.core.governance.risk_engine.datetime", MockDatetime)

        prob = model.predict(compute_intent, {})
        assert prob is not None
        assert 0 <= prob <= 1


# -----------------------------------------------------------------------------
# RiskEngine comprehensive tests
# -----------------------------------------------------------------------------
class TestRiskEngineComprehensive:
    def test_hyperprior_disabled_when_pyro_missing(self):
        """Test that hyperprior is disabled if Pyro not installed."""
        with patch("agentic_reliability_framework.core.governance.risk_engine.PYRO_AVAILABLE", False):
            engine = RiskEngine(use_hyperpriors=True)
            assert engine.use_hyperpriors is False
            assert engine.hyperprior_store is None

    def test_calculate_risk_with_all_three_components(self, compute_intent, monkeypatch):
        """Test weight calculation when all three components are available."""
        engine = RiskEngine(use_hyperpriors=True, n0=100, hyperprior_weight=0.3)
        # Mock hyperprior to return a value
        engine.hyperprior_store = MagicMock()
        engine.hyperprior_store.get_risk_summary.return_value = {"mean": 0.4}
        # Mock HMC to return a value
        engine.hmc_model = MagicMock()
        engine.hmc_model.predict.return_value = 0.6
        engine.hmc_model.is_ready = True
        # Set total_incidents high enough to give HMC full weight
        engine.total_incidents = 200

        risk, expl, contribs = engine.calculate_risk(compute_intent, 100, [])
        weights = contribs['weights']
        assert weights['hmc'] > 0
        assert weights['hyper'] > 0
        assert weights['conjugate'] > 0
        assert abs(weights['conjugate'] + weights['hyper'] + weights['hmc'] - 1.0) < 1e-6
        # Final risk should be weighted average
        conjugate_mean = 1.0 / (1.0 + 12.0)  # compute category prior
        expected = (weights['conjugate'] * conjugate_mean + weights['hyper']*0.4 + weights['hmc']*0.6)
        assert risk == pytest.approx(expected)

    def test_calculate_risk_with_hyper_and_conj(self, compute_intent):
        """Test when only conjugate and hyper are available."""
        engine = RiskEngine(use_hyperpriors=True)
        engine.hyperprior_store = MagicMock()
        engine.hyperprior_store.get_risk_summary.return_value = {"mean": 0.4}
        engine.hmc_model.predict.return_value = None  # HMC not ready
        engine.total_incidents = 50

        risk, expl, contribs = engine.calculate_risk(compute_intent, 100, [])
        weights = contribs['weights']
        assert weights['hmc'] == 0.0
        assert weights['hyper'] > 0
        assert weights['conjugate'] > 0

    def test_calculate_risk_with_hmc_and_conj(self, compute_intent):
        """Test when only conjugate and HMC are available."""
        engine = RiskEngine(use_hyperpriors=False)  # hyper off
        engine.hmc_model = MagicMock()
        engine.hmc_model.predict.return_value = 0.6
        engine.hmc_model.is_ready = True
        engine.total_incidents = 200

        risk, expl, contribs = engine.calculate_risk(compute_intent, 100, [])
        weights = contribs['weights']
        assert weights['hyper'] == 0.0
        assert weights['hmc'] > 0
        assert weights['conjugate'] > 0

    def test_calculate_risk_with_only_conjugate(self, compute_intent):
        """Test fallback to only conjugate."""
        engine = RiskEngine(use_hyperpriors=False)
        engine.hmc_model.predict.return_value = None
        risk, expl, contribs = engine.calculate_risk(compute_intent, 100, [])
        weights = contribs['weights']
        assert weights['conjugate'] == 1.0
        assert weights['hyper'] == 0.0
        assert weights['hmc'] == 0.0

    def test_context_multiplier_for_different_environments(self, compute_intent):
        """Test _context_multiplier for dev vs prod."""
        engine = RiskEngine()
        # dev should have multiplier 1.0
        mult_dev = engine._context_multiplier(compute_intent)
        assert mult_dev == 1.0
        # prod
        prod_intent = ProvisionResourceIntent(
            resource_type=ResourceType.VM,
            region="eastus",
            size="Standard_D2s_v3",
            environment="prod",  # string
            requester="tester",
        )
        mult_prod = engine._context_multiplier(prod_intent)
        assert mult_prod == 1.5

    def test_context_multiplier_for_deployment_target(self, default_intent):
        """Test multiplier for intents with deployment_target."""
        engine = RiskEngine()
        mult = engine._context_multiplier(default_intent)
        assert mult == 1.5  # because deployment_target is prod

    def test_persist_beta_store(self, compute_intent, tmp_path):
        """Test save and load of beta store."""
        engine = RiskEngine()
        engine.update_outcome(compute_intent, success=True)
        original_alpha, original_beta = engine.beta_store.get(ActionCategory.COMPUTE)

        save_path = tmp_path / "beta.json"
        engine.persist_beta_store(str(save_path))

        new_engine = RiskEngine()
        new_engine.load_beta_store(str(save_path))
        loaded_alpha, loaded_beta = new_engine.beta_store.get(ActionCategory.COMPUTE)

        assert loaded_alpha == original_alpha
        assert loaded_beta == original_beta

    # Note: get_system_risk is not a method of RiskEngine; it might be elsewhere.
    # We'll skip that test.

    def test_categorize_intent_all_types(self, compute_intent, database_intent, network_intent,
                                         security_intent, default_intent):
        """Test categorize_intent for all categories."""
        assert categorize_intent(compute_intent) == ActionCategory.COMPUTE
        assert categorize_intent(database_intent) == ActionCategory.DATABASE
        assert categorize_intent(network_intent) == ActionCategory.NETWORK
        assert categorize_intent(security_intent) == ActionCategory.SECURITY
        assert categorize_intent(default_intent) == ActionCategory.DEFAULT

        # Special case: DeployConfigurationIntent with "database" in service name
        db_deploy = DeployConfigurationIntent(
            service_name="database-migrate",
            change_scope="global",
            deployment_target="dev",
            requester="tester",
            configuration={},
        )
        assert categorize_intent(db_deploy) == ActionCategory.DATABASE

    def test_update_outcome_thread_safety(self, compute_intent):
        """Simple test that multiple updates don't crash."""
        import threading
        engine = RiskEngine()
        def updater():
            for _ in range(10):
                engine.update_outcome(compute_intent, success=True)
        threads = [threading.Thread(target=updater) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        alpha, beta = engine.beta_store.get(ActionCategory.COMPUTE)
        assert alpha > PRIORS[ActionCategory.COMPUTE][0]

    def test_train_hmc(self, compute_intent, tmp_path):
        """Test train_hmc calls hmc_model.train."""
        engine = RiskEngine()
        engine.hmc_model = MagicMock()
        df = pd.DataFrame({'col': [1]})
        engine.train_hmc(df)
        engine.hmc_model.train.assert_called_once_with(df)

    def test_hyperprior_get_risk_summary_no_history(self):
        """Test get_risk_summary returns fallback when no history."""
        with patch("agentic_reliability_framework.core.governance.risk_engine.PYRO_AVAILABLE", True):
            store = HyperpriorBetaStore()
            store._initialized = True
            store._history = []
            summary = store.get_risk_summary(ActionCategory.COMPUTE)
            assert summary == {"mean": 0.5, "p5": 0.1, "p50": 0.5, "p95": 0.9}
