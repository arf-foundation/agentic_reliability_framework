# Getting Started with ARF

This guide will walk you through installing ARF, running your first evaluation, and understanding the output.

## Installation

### Using pip

```bash
pip install agentic-reliability-framework
```

### From source (for development)

```python
git clone https://github.com/petter2025us/agentic-reliability-framework.git
cd agentic-reliability-framework
pip install -e .
```

Your First Evaluation
---------------------

Let’s evaluate a request to provision a virtual machine.

```python
# 1. Import necessary classes
from agentic_reliability_framework.core.governance.intents import ProvisionResourceIntent, ResourceType
from agentic_reliability_framework.core.governance.governance_loop import GovernanceLoop
from agentic_reliability_framework.core.governance.policies import PolicyEvaluator
from agentic_reliability_framework.core.governance.cost_estimator import CostEstimator
from agentic_reliability_framework.core.governance.risk_engine import RiskEngine

# 2. Create the infrastructure intent
intent = ProvisionResourceIntent(
    resource_type=ResourceType.VM,
    region="eastus",
    size="Standard_D2s_v3",
    requester="dev-team",
    environment="prod"
)

# 3. Set up the governance components (defaults are fine for a first run)
policy_evaluator = PolicyEvaluator()
cost_estimator = CostEstimator()
risk_engine = RiskEngine()

# 4. Create the governance loop
loop = GovernanceLoop(
    policy_evaluator=policy_evaluator,
    cost_estimator=cost_estimator,
    risk_engine=risk_engine
)

# 5. Run the evaluation
healing_intent = loop.run(intent, context={"incident_id": "first-run"})

# 6. Inspect the result
print(f"Action: {healing_intent.action}")
print(f"Risk score: {healing_intent.risk_score:.3f}")
print(f"Risk factors: {healing_intent.risk_factors}")
print(f"Justification: {healing_intent.justification}")
```

If you run this, you’ll see something like:

```text
Action: approve
Risk score: 0.120
Risk factors: {'conjugate': 0.120}
Justification: Bayesian risk for category 'compute': conjugate mean = 0.120 (α=1.0, β=12.0). Context multiplier: 1.00. Final risk: 0.120.
```

Understanding the Output
------------------------

The HealingIntent contains:

*   **action**: one of approve, deny, escalate – the recommended next step.
    
*   **risk\_score**: the estimated probability of failure (0–1). Lower is better.
    
*   **risk\_factors**: how much each model component contributed. In this case, only the conjugate prior contributed (since we didn’t train an HMC model and hyperpriors are off).
    
*   **justification**: a human‑readable explanation.
    
*   **metadata**: additional data like decision trace, epistemic breakdown, and (if used) forecasts.
    

Going Further
-------------

### Adding a Custom Policy

Policies are evaluated by PolicyEvaluator. Suppose you want to reject any request from the “test” user. You can subclass PolicyEvaluator or modify the default one:

```python
class MyPolicyEvaluator(PolicyEvaluator):
    def evaluate(self, intent, context):
        violations = super().evaluate(intent, context)
        if intent.requester == "test":
            violations.append("Test user not allowed")
        return violations

loop = GovernanceLoop(
    policy_evaluator=MyPolicyEvaluator(),
    cost_estimator=cost_estimator,
    risk_engine=risk_engine
)
```

### Feeding Back Outcomes

After an action is executed (in your own system), you should update the risk engine so it learns:

```python
# Assume you executed the action and it succeeded
success = True
risk_engine.update_outcome(intent, success)
```

This updates the conjugate priors for the category of the intent, making future risk estimates more accurate.

### Using the Predictive Engine

Add telemetry data and retrieve forecasts:

```python
from agentic_reliability_framework.runtime.analytics.predictive import SimplePredictiveEngine

predictive = SimplePredictiveEngine()
predictive.add_telemetry("payment-service", {"latency": 120, "error_rate": 0.02})
forecasts = predictive.forecast_service_health("payment-service")
```

Then pass the predictive engine to GovernanceLoop to incorporate forecasting into the risk score.

### Enabling Epistemic Uncertainty

To include uncertainty from hallucination risk and forecast confidence, turn on the epistemic gate:

```python
from agentic_reliability_framework.core.research.eclipse_probe import HallucinationRisk

loop = GovernanceLoop(
    policy_evaluator=policy_evaluator,
    cost_estimator=cost_estimator,
    risk_engine=risk_engine,
    enable_epistemic=True,
    hallucination_probe=HallucinationRisk(),
    predictive_engine=predictive
)
```

Now the risk score will be adjusted, and confidence in the HealingIntent will be 1 - epistemic\_uncertainty.

Running the Interactive Demo
----------------------------

The repository includes a small Gradio app that demonstrates the risk dashboard. To run it:

```python
git clone https://github.com/arf-foundation/agentic_reliability_framework.git
cd agentic-reliability-framework
pip install -e .
python -m gradio app.py
```

Open [http://localhost:7860](http://localhost:7860/) to see the UI.

Next Steps
----------

*   Read the [API reference](https://api_reference.md/) for a detailed look at all classes and methods.
    
*   Check out [examples.md](https://examples.md/) for more advanced usage patterns.
    
*   Join the community on [GitHub Discussions](https://github.com/petter2025us/agentic-reliability-framework/discussions) to ask questions and share your experiences.
    

We’re excited to see what you build with ARF!
