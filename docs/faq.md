# Frequently Asked Questions

## General

### What is ARF?

Agentic Reliability Framework (ARF) is an open‑source advisory engine that uses Bayesian inference to evaluate cloud infrastructure requests. It returns a recommendation – approve, deny, or escalate – with a calibrated risk score and a full explanation.

### Is ARF production‑ready?

The OSS core is stable and used in internal testing. For production use with actual cloud providers (Azure, AWS, etc.), we recommend the Enterprise edition, which adds enforcement, audit trails, and support.

### What’s the difference between OSS and Enterprise?

- **OSS**: advisory only, returns `HealingIntent` with recommendations but does not execute anything. It uses in‑memory storage and limited FAISS index types.  
- **Enterprise**: adds autonomous execution, persistent storage, approval workflows, audit trails, and support for advanced FAISS indices (IVF, HNSW) and embedding models.

### How do I get the Enterprise edition?

Contact us at [petter2025us@outlook.com](mailto:petter2025us@outlook.com) or book a call via [Calendly](https://calendly.com/petter2025us/30min).

## Installation & Setup

### What are the system requirements?

Python 3.10 or later. The OSS core has minimal dependencies; for HMC you need `pymc` and `arviz`, and for optional embedding you need `sentence-transformers` and `torch`.

### How do I install from source?

```bash
git clone https://github.com/petter2025us/agentic-reliability-framework.git
cd agentic-reliability-framework
pip install -e .
```

### Why do I get an import error for torch or pymc?

Those are optional dependencies. If you don’t need HMC or advanced embeddings, you can ignore the error. To install them, run:

```bash
pip install pymc arviz torch sentence-transformers
```

Usage
-----

### How do I interpret the risk\_factors?

risk\_factors is a dictionary where keys are model components (conjugate, hyperprior, hmc) and values are additive contributions to the total risk score. They sum to the risk\_score (within floating‑point precision). This allows you to see which part of the model dominated the decision.

### What is the epistemic uncertainty and how is it computed?

Epistemic uncertainty is computed from three sources:

*   Hallucination risk from the ECLIPSE probe (if enabled)
    
*   Forecast uncertainty (1 – average confidence of forecasts)
    
*   Data sparsity (exponential decay based on history length)
    

The final value is 1 - ∏(1 - u\_i), which can be interpreted as the probability that the model’s knowledge is insufficient. A high value triggers escalation if the epistemic gate is enabled.

### Can I use my own cost data?

Yes. The CostEstimator can be subclassed or replaced. The default uses a YAML file (pricing.yml) that you can edit. See the example in the README.

### How do I add custom policies?

Policies are evaluated by PolicyEvaluator. You can add rules by editing evaluate\_policies in core/governance/policy\_engine.py or by implementing your own evaluator and passing it to GovernanceLoop.

Performance & Limitations
-------------------------

### How many intents can ARF process per second?

The OSS core is not optimized for high throughput; it is meant for advisory analysis. In benchmarks, it can handle ~10–20 requests per second on a typical CPU. For higher throughput, use the Enterprise edition with caching and parallel processing.

### What are the memory limits for the RAG graph?

OSS limits:

*   Max incident nodes: 1000
    
*   Max outcome nodes: 5000
    
*   FAISS index type: IndexFlatL2 only
    

These are hard‑coded in constants.py. If you need larger graphs, consider the Enterprise edition.

### Can I train the HMC model on‑the‑fly?

Yes, you can call risk\_engine.train\_hmc(df) at any time. The model is serialised to hmc\_model.json and reloaded automatically.

Contributing & Support
----------------------

### How can I contribute?

See [development.md](https://development.md/) for guidelines. We welcome bug reports, documentation improvements, and feature requests.

### Where do I report a bug?

Open an issue on [GitHub](https://github.com/petter2025us/agentic-reliability-framework/issues).

### Is there a community Slack?

Not yet, but we plan to create one. For now, use GitHub Discussions.

Licensing
---------

### What license does ARF use?

Apache 2.0. See the [LICENSE](https://github.com/petter2025us/agentic-reliability-framework/blob/main/LICENSE) file.

### Can I use ARF in a commercial product?

Yes, the OSS edition is free. If you need enterprise features, a commercial license is required.

### Can I modify the code?

Yes, under the terms of Apache 2.0. You must retain the original copyright notices.

Troubleshooting
---------------

### I get ValidationError: Execution not allowed in OSS edition

This happens if you try to set execution\_allowed=True or set a status that implies execution. In OSS, all HealingIntent are automatically marked OSS\_ADVISORY\_ONLY with execution\_allowed=False. This is a safety measure.

### My forecasts are empty / not showing

Ensure you have added enough telemetry points (FORECAST\_MIN\_DATA\_POINTS = 5 by default). Also check that the metric names match (e.g., latency vs latency\_p99).

### The risk score is always 0.5

This typically means the risk engine has no data and no HMC model. Add some training data or use the update mechanism to feed outcomes.

### How do I enable hyperpriors?

Hyperpriors require pyro and torch. Install them, then pass use\_hyperpriors=True when creating RiskEngine. The hyperprior component will contribute to risk when enough data is available.

For more help, search existing [GitHub issues](https://github.com/petter2025us/agentic-reliability-framework/issues) or open a new one.
