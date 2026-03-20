# API Reference

ARF provides a Python API for evaluating infrastructure intents, computing risk, and retrieving healing recommendations. All public classes and methods are documented below.

## Core Classes

### `InfrastructureIntent` (abstract)

Base class for all infrastructure requests. Concrete subclasses:

- `ProvisionResourceIntent` – request to create a cloud resource.
- `GrantAccessIntent` – request to grant permissions.
- `DeployConfigurationIntent` – request to change configuration.

All intents have common fields: `service_name`, `environment`, `requester`, and optional `provenance`.

### `HealingIntent`

Immutable container for a recommendation. Created by `GovernanceLoop.run()`.

**Key fields**:

- `action` (`str`) – one of `"approve"`, `"deny"`, `"escalate"`.
- `component` (`str`) – the affected service/component.
- `risk_score` (`float`) – calibrated failure probability (0–1).
- `risk_factors` (`Dict[str, float]`) – additive contributions from each Bayesian component.
- `confidence` (`float`) – derived from epistemic uncertainty.
- `policy_violations` (`List[str]`) – list of violated policy rules.
- `recommended_action` (`RecommendedAction`) – enum with `APPROVE`, `DENY`, `ESCALATE`.
- `metadata` (`Dict`) – stores decision trace, forecasts, epistemic breakdown, etc.
- `source` (`IntentSource`) – indicates origin (`INFRASTRUCTURE_ANALYSIS`).
- `infrastructure_intent_id` – link to the original infrastructure intent.
- `ancestor_chain` – tuple of parent intent IDs for full traceability.

**Methods**:

- `to_dict(include_oss_context: bool) -> Dict` – serialise to JSON‑compatible dict.
- `with_execution_result(...)` – create a new intent with execution outcome (enterprise use).
- `sign(private_key)` – cryptographic signing (optional).

### `GovernanceLoop`

Orchestrator that integrates all components.

```python
loop = GovernanceLoop(
    policy_evaluator=PolicyEvaluator(),
    cost_estimator=CostEstimator(),
    risk_engine=RiskEngine(),
    memory=RAGGraphMemory(...),
    enable_epistemic=True,
    hallucination_probe=HallucinationRisk(),
    predictive_engine=SimplePredictiveEngine(),
)

intent = loop.run(infrastructure_intent, context={...})
```

**Parameters**:

*   policy\_evaluator – evaluates policy rules.
    
*   cost\_estimator – projects monthly cost.
    
*   risk\_engine – computes Bayesian risk.
    
*   memory – (optional) RAG memory for similar incidents.
    
*   enable\_epistemic – if True, uses hallucination probe and predictive uncertainty.
    
*   hallucination\_probe – ECLIPSE probe for epistemic risk.
    
*   predictive\_engine – forecasts service health.
    

**Methods**:

*   run(intent, context) -> HealingIntent – single evaluation.
    
*   run\_batch(intents, contexts) -> List\[HealingIntent\] – batch evaluation.
    

### RiskEngine

Bayesian risk scoring.

```python
risk_engine = RiskEngine(hmc_model_path="hmc_model.json", use_hyperpriors=True)

risk, explanation, contributions = risk_engine.calculate_risk(
    intent=intent,
    cost_estimate=100.0,
    policy_violations=["Region not allowed"]
)
```

**Returns**:

*   risk – final risk score.
    
*   explanation – human‑readable string.
    
*   contributions – dict with weights, conjugate\_mean, hyper\_mean, hmc\_prediction, and Beta parameters.
    

**Methods**:

*   update\_outcome(intent, success) – update online priors with observed outcome.
    
*   train\_hmc(incidents\_df) – train the HMC model on historical data.
    

### SimplePredictiveEngine

Lightweight forecasting using linear regression.

```python
engine = SimplePredictiveEngine(history_window=100)
engine.add_telemetry("payment-service", {"latency": 120, "error_rate": 0.03})
forecasts = engine.forecast_service_health("payment-service")
# Returns list of ForecastResult objects
```

### HealingIntentSerializer

JSON serialisation and deserialisation.

```python
# Serialise
data = HealingIntentSerializer.serialize(intent, version="2.1.0")
json_str = HealingIntentSerializer.to_json(intent)

# Deserialise
intent2 = HealingIntentSerializer.from_json(json_str)

# Convert to enterprise payload
payload = intent.to_enterprise_request()
```

Constants and Configuration
All OSS limits and Bayesian decision coefficients are defined in core/config/constants.py. Important ones:

EPISTEMIC_ESCALATION_THRESHOLD – default 0.5.

COST_FP, COST_FN, COST_REVIEW – cost coefficients for expected loss.

USE_EPISTEMIC_GATE – if True, escalate on high epistemic uncertainty.

Exceptions
ValidationError – raised if a HealingIntent fails validation.

IntegrityError – signature verification failed.

OSSBoundaryError – configuration violates OSS limits.

For a complete list of modules and functions, see the source code documentation. https://arf-foundation.github.io/arf-spec/
