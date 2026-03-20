# Development Guide

This guide explains how to set up a development environment, run tests, and contribute to ARF.

## Prerequisites

- Python 3.10 or later
- Git
- (Optional) `conda` if you prefer Conda environments

## Setting Up the Development Environment

1. **Clone the repository**

   ```bash
   git clone https://github.com/petter2025us/agentic-reliability-framework.git
   cd agentic-reliability-framework
   ```
2. **Create a virtual environment**
   
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -e ".[dev]"
```

The \[dev\] extras include pytest, black, ruff, and other development tools.

**Optional: install HMC dependencies** (if you plan to train models)

Running Tests
-------------

```bash
pytest tests/
```

To run a specific test:
-----------------------

```bash
pytest tests/core/governance/test_governance_loop.py -k test_risk_factors
```

We use GitHub Actions for CI; the configuration is in .github/workflows/python-package-conda.yml.

Code Style
----------

*   Formatting: [Black](https://black.readthedocs.io/) with default settings.
    
*   Linting: [Ruff](https://docs.astral.sh/ruff/) (configured in pyproject.toml).
    
*   Type hints: we use mypy for static type checking.
    

Run formatter and linter before committing:

```bash
black src/ tests/
ruff check src/ tests/
```

Project Structure
-----------------

```text
agentic_reliability_framework/
├── core/
│   ├── governance/         # Risk engine, policies, intents, healing intent
│   ├── config/             # Constants and validation
│   ├── models/             # Data models (ForecastResult, ReliabilityEvent)
│   └── research/           # ECLIPSE hallucination probe
├── runtime/
│   ├── analytics/          # Predictive engine, business impact calculator
│   └── memory/             # RAG graph memory, FAISS index
└── __init__.py
```

Adding a New Policy
Policies are evaluated by PolicyEvaluator. To add a custom policy:

Subclass Policy (if you want a new rule type) or add a method to PolicyEvaluator.

Ensure the policy returns a list of violation strings.

The GovernanceLoop passes those violations to the risk engine and includes them in the HealingIntent.

Example:

```python
class MyCustomPolicy(Policy):
    def evaluate(self, intent, context):
        violations = []
        if intent.requester == "anonymous":
            violations.append("Anonymous requester not allowed")
        return violations
```

Modifying the Bayesian Risk Engine
----------------------------------

The risk engine is in core/governance/risk\_engine.py. It consists of:

*   BetaStore – conjugate prior storage.
    
*   HyperpriorBetaStore – hierarchical Beta with Pyro (optional).
    
*   HMCModel – offline HMC logistic regression.
    

To change the prior distributions, modify the PRIORS dictionary. To alter the blending weights, adjust the n0 and hyperprior\_weight parameters in RiskEngine.\_\_init\_\_.

Contributing
------------

We welcome contributions! Please follow these steps:

1.  Open an issue describing the change you want to make.
    
2.  Fork the repository and create a new branch.
    
3.  Write code and tests.
    
4.  Run the full test suite.
    
5.  Submit a pull request with a clear description.
    

For larger changes, please discuss with maintainers beforehand.

Building Documentation
----------------------

This documentation is written in Markdown and hosted on GitHub. To preview locally, you can use any Markdown viewer. We plan to migrate to a static site generator later.

Releasing a New Version
-----------------------

1.  Update \_\_version\_\_ in \_\_init\_\_.py.
    
2.  Update CHANGELOG.md.
    
3.  Tag the commit: git tag vX.Y.Z.
    
4.  Push tag: git push origin vX.Y.Z.
    
5.  The GitHub Actions workflow will build and publish to PyPI.
    

License
-------

Apache 2.0 – see [LICENSE](https://github.com/petter2025us/agentic-reliability-framework/blob/main/LICENSE).
