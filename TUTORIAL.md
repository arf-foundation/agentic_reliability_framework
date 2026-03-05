# Quick‑Start Tutorial for ARF

This tutorial walks a new user through the Agentic Reliability Framework
(ARF) core concepts and typical workflows. All snippets are runnable; copy
and paste into a Python REPL or notebook after installing the package.

A live, web‑based demo of the engine is available at
https://huggingface.co/spaces/petter2025/Agentic-Reliability-Framework-v4.

---

## 1. Installation & Environment

```bash
# clone the repo and create a virtual environment (see CONTRIBUTING.md for details)
git clone git@github.com:petter2025us/agentic-reliability-framework.git
cd agentic-reliability-framework
conda env create -f environment.yml        # or python -m venv .venv
conda activate arf
pip install -e .                          # installs ARF and dependencies
```

You can also install via `pip install agentic-reliability-framework` once a
release is published.

Open a Python shell to follow the rest of the examples:
```bash
python -i
```

---

## 2. Basic Concepts

### **ReliabilityEvent & HealingIntent**

- A `ReliabilityEvent` is an observation (latency spike, error rate surge,
etc.) that ARF analyzes.
- A `HealingIntent` is ARF's recommendation for a corrective action
  (restart a container, scale up, etc.). In OSS it is advisory only; the
  enterprise layer executes it.

### **RiskEngine**

Responsible for computing a risk score using hybrid Bayesian models: fast
conjugate priors online and heavier HMC predictors offline. See
`core/governance/risk_engine.py` for details. Note: the OSS engine uses per‑category default Beta priors (see the code) rather than a single uniform prior.

### **Policy Algebra**

The `Policy` class provides composable building blocks (AND, OR, NOT) and
`PolicyEvaluator`/`ProbabilisticPolicyEvaluator` handle deterministic and
uncertain inputs.

---

## 3. A Step‑by‑Step Example

This example creates a provisioning intent, evaluates policies, computes risk,
and produces a healing intent.

```python
from agentic_reliability_framework.core.governance.policies import (
    RegionAllowedPolicy, CostThresholdPolicy, PolicyEvaluator
)
from agentic_reliability_framework.core.governance.risk_engine import RiskEngine
from agentic_reliability_framework.core.governance.healing_intent import HealingIntent
from agentic_reliability_framework.core.governance.intents import ProvisionResourceIntent, ResourceType

# 1. define some policies
region_policy = RegionAllowedPolicy({"eastus", "westeurope"})
cost_policy = CostThresholdPolicy(500.0)
root = region_policy & cost_policy
pe = PolicyEvaluator(root)

# 2. create an intent
intent = ProvisionResourceIntent(
    resource_type=ResourceType.VM,
    region="eastus",
    size="Standard_D2s_v3",
    requester="alice",
    environment="dev",
)

# 3. evaluate policies
det_violations = pe.evaluate(intent, {"cost_estimate": 320.0})
print("violations", det_violations)  # [] if all good

# 4. compute risk
engine = RiskEngine()
score, explanation, contributions = engine.calculate_risk(intent, cost_estimate=320.0, policy_violations=det_violations)
print("risk", score)
print(explanation)

# 5. assemble a healing intent
healing = HealingIntent(
    action="provision_vm",
    component="web-cluster",
    parameters={"size": intent.size},
    justification=f"Risk score {score:.2f}; {det_violations or 'no violations'}",
    confidence=1 - score,
)
print(healing)
```

The `HealingIntent` can then be serialized, stored, or sent to an enterprise
MCP server.

---

## 4. Multi‑Agent Analysis with `EnhancedReliabilityEngine`

The runtime module orchestrates multiple agents (detective, diagnostician,
predictive, etc.) and optionally queries an LLM (Claude). Here's how to run
it programmatically:

```python
import asyncio
from agentic_reliability_framework.runtime.engine import EnhancedReliabilityEngine

async def main():
    engine = EnhancedReliabilityEngine()
    result = await engine.process_event_enhanced(
        component="api-service",
        latency=450,
        error_rate=0.12,
    )
    print(result['multi_agent_analysis'])
    print(result['healing_actions'])
    print(result.get('claude_synthesis'))

asyncio.run(main())
```

The JSON result contains severity, anomaly flag, HMC analysis (if trained),
and a synthesized summary.

---

## 5. Interpreting Risk Scores and Credible Intervals

The `RiskEngine.calculate_risk` returns a float in [0,1] plus an explanatory
string. You can inspect the conjugate posterior parameters (alpha, beta) from the engine's beta store and compute credible intervals if desired. Example:

```python
from agentic_reliability_framework.core.governance.risk_engine import categorize_intent
from scipy.stats import beta

# compute category for an intent (use the same categorization the engine uses)
category = categorize_intent(intent)
alpha, beta_param = engine.beta_store.get(category)
ci = beta.interval(0.95, alpha, beta_param)
print("95% interval", ci)
```

Note: the OSS engine uses tuned, per‑category priors by default (see `core/governance/risk_engine.py`). The mapping from risk score to an action (approve/deny/escalate) is typically performed by applying deterministic thresholds (DPT) in the calling layer.

---

## 6. Running the Interactive Demo Locally

A lightweight Gradio app is available in the companion repository. To run it:

```bash
pip install gradio
# assume demo script `app.py` lives under examples/ (create one if needed)
python examples/app.py
```

Visit `http://localhost:7860` to interact with the engine via a browser. The
demo lets you type intents, view semantic memory results, and inspect risk
components live.

---

That concludes the quick tour! Dive into the code, read the docs at
https://docs.agentic-reliability-framework.io, and don't hesitate to open an
issue or contribute a PR. Happy hacking!
