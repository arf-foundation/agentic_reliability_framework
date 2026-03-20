# Examples

This page shows how to use ARF for different scenarios. All examples assume you have installed the package and are working in a Python environment.

## Basic Usage

### Evaluate a Provisioning Request

```python
from agentic_reliability_framework.core.governance.intents import ProvisionResourceIntent, ResourceType
from agentic_reliability_framework.core.governance.governance_loop import GovernanceLoop
from agentic_reliability_framework.core.governance.policies import PolicyEvaluator
from agentic_reliability_framework.core.governance.cost_estimator import CostEstimator
from agentic_reliability_framework.core.governance.risk_engine import RiskEngine

# Create an infrastructure intent
intent = ProvisionResourceIntent(
    resource_type=ResourceType.VM,
    region="eastus",
    size="Standard_D2s_v3",
    requester="dev-team",
    environment="prod"
)

# Set up components
policy_evaluator = PolicyEvaluator()
cost_estimator = CostEstimator()
risk_engine = RiskEngine()

loop = GovernanceLoop(
    policy_evaluator=policy_evaluator,
    cost_estimator=cost_estimator,
    risk_engine=risk_engine
)

# Run the loop
healing_intent = loop.run(intent, context={"incident_id": "demo-123"})

print(f"Action: {healing_intent.action}")
print(f"Risk: {healing_intent.risk_score:.3f}")
print(f"Risk factors: {healing_intent.risk_factors}")
print(f"Justification: {healing_intent.justification}")
```

### Simulate a High‑Risk Scenario

```python
# Manually set the risk engine to return high risk
risk_engine.calculate_risk = lambda intent, cost, violations: (0.95, "High risk", {})

healing_intent = loop.run(intent)
print(f"Recommended action: {healing_intent.action}")  # should be "deny"
```

### Using the Predictive Engine

```python
from agentic_reliability_framework.runtime.analytics.predictive import SimplePredictiveEngine

predictive = SimplePredictiveEngine()

# Add historical telemetry
predictive.add_telemetry("payment-service", {"latency": 150, "error_rate": 0.02})
predictive.add_telemetry("payment-service", {"latency": 180, "error_rate": 0.03})
# ... more points

forecasts = predictive.forecast_service_health("payment-service")
for f in forecasts:
    print(f"{f.metric}: predicted {f.predicted_value:.0f}, confidence {f.confidence:.2f}, risk {f.risk_level}")
```

### Incorporating Epistemic Uncertainty

```python
from agentic_reliability_framework.core.research.eclipse_probe import HallucinationRisk

hallucination_probe = HallucinationRisk()

loop = GovernanceLoop(
    policy_evaluator=policy_evaluator,
    cost_estimator=cost_estimator,
    risk_engine=risk_engine,
    enable_epistemic=True,
    hallucination_probe=hallucination_probe,
    predictive_engine=predictive
)

context = {
    "service_name": "payment-service",
    "query": "high latency in eastus",
    "evidence": "previous incidents show network congestion",
    "entropy": 0.7,
    "evidence_lift": 0.4,
    "contradiction": 0.2,
    "latency_p99": 350,
    "error_rate": 0.12,
    "throughput": 10000
}

intent = loop.run(sample_intent, context=context)
print(f"Epistemic uncertainty: {intent.metadata['epistemic_breakdown']['hallucination']:.3f}")
print(f"Final confidence: {intent.confidence:.3f}")
```

Working with the Risk Engine Directly
-------------------------------------

### Update Conjugate Priors with Outcomes

```python
# After an approval was executed
risk_engine.update_outcome(intent, success=True)   # True if the action succeeded

# The posterior parameters are updated automatically
alpha, beta = risk_engine.beta_store.get(category)
print(f"New posterior: α={alpha:.1f}, β={beta:.1f}")
```

### Train HMC Model on Historical Data

```python
import pandas as pd

# Assume you have a DataFrame with columns: hour, env_prod, user_role, category, outcome, and one‑hot encoded category columns
df = pd.read_csv("historical_incidents.csv")
risk_engine.train_hmc(df)
```

Using Semantic Memory
---------------------

```python
from agentic_reliability_framework.runtime.memory import create_faiss_index, RAGGraphMemory

faiss_index = create_faiss_index(dim=384)
memory = RAGGraphMemory(faiss_index)

# Add incidents (in production you would embed them with a model)
memory.add_incident(incident_id="inc1", component="db", metrics={"latency": 200})
memory.add_incident(incident_id="inc2", component="db", metrics={"latency": 150})

# Retrieve similar incidents
similar = memory.find_similar(event, analysis, k=3)
```

Customising the Governance Loop
-------------------------------

You can replace any component with your own implementation as long as it follows the expected interface.

### Custom Policy Evaluator

```python
class MyPolicyEvaluator:
    def evaluate(self, intent, context):
        violations = []
        if intent.requester == "untrusted":
            violations.append("Requester not allowed")
        if context.get("cost_estimate", 0) > 500:
            violations.append("Cost exceeds budget")
        return violations
```

### Custom Risk Engine

```python
class MyRiskEngine:
    def calculate_risk(self, intent, cost_estimate, policy_violations):
        # Custom logic
        return 0.5, "My explanation", {"custom_factor": 0.3}
```

Full Example: End‑to‑End with Feedback
--------------------------------------

```python
# Evaluate
healing = loop.run(intent, context={...})

if healing.action == "approve":
    # Execute the action (in a real system you would call cloud API)
    success = True
else:
    success = False

# Provide feedback to update priors
risk_engine.update_outcome(intent, success)
```

Running the Interactive Demo
----------------------------

The repository includes a small demo that uses the Gradio interface. To run it locally:

```bash
python -m gradio app.py
```

Then open [http://localhost:7860](http://localhost:7860/).

For more examples, see the [Tutorial](https://tutorial.md/).
