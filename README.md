# Agentic Reliability Framework (ARF) – OSS Edition v4

[![PyPI](https://img.shields.io/pypi/v/agentic-reliability-framework.svg)](https://pypi.org/project/agentic-reliability-framework/)
[![Python Versions](https://img.shields.io/pypi/pyversions/agentic-reliability-framework.svg)](https://pypi.org/project/agentic-reliability-framework/)
[![Tests](https://github.com/petter2025us/agentic-reliability-framework/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/petter2025us/agentic-reliability-framework/actions/workflows/python-package-conda.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Hugging Face Space](https://img.shields.io/badge/demo%20space-v4-orange?logo=huggingface)](https://huggingface.co/spaces/petter2025/Agentic-Reliability-Framework-v4)
[![Reliable AI Systems Stack](https://img.shields.io/badge/🤗-Reliable%20AI%20Stack-blue)](https://huggingface.co/collections/petter2025/reliable-ai-systems-stack)

## 🎯 Overview

**Agentic Reliability Framework (ARF)** is an open‑source advisory engine for cloud infrastructure governance. It provides **provably safe, mathematically grounded recommendations**—approve, deny, or escalate—when users request provisioning, configuration, or access changes.

Powered by **Bayesian probabilistic models** (conjugate priors + HMC sampling), **composable policy algebra**, and **semantic memory** (FAISS + embeddings), ARF delivers:

- **Transparent decisions** with audit trails and human-readable justifications
- **Fast online learning** using beta-binomial conjugate priors  
- **Deep pattern discovery** via Hamiltonian Monte Carlo (HMC/NUTS)
- **Contextual analysis** by retrieving similar past incidents from a semantic graph
- **Multi-agent orchestration** with anomaly detection, root cause analysis, and forecasting

This repository contains the **OSS core** for inference and advisory analysis. Enterprise customers layer enforcement, audit trails, and proprietary safety heuristics on top.

### Key Insight: Deterministic Probability Thresholding (DPT)

Instead of credible intervals, p-values, or fuzzy logic, ARF uses **transparent thresholds**:

- **Approve** if P(failure) < 0.2
- **Deny** if P(failure) > 0.8  
- **Escalate** if 0.2 ≤ P(failure) ≤ 0.8

This makes decisions auditable, reproducible, and immune to the "credibility paradox."

---

## 🔍 Problem Statement

Modern cloud platforms support thousands of perturbable resources and users.
Traditional policy engines devolve into brittle rule‑sprawl; naive ML models
either overfit or offer no audit trail. ARF bridges this gap with a
**bayesian‑compatible, hybrid‑learning advisory engine** grounded in clear
decision rules and open semantics.

---

## 🚀 High‑Level Overview

ARF is structured as a **hybrid intelligence engine** coupling fast online
updates with offline pattern learning. A simulator takes an
`InfrastructureIntent` (e.g. provision a VM, deploy configuration, grant access)
and returns a `HealingIntent` recommendation. Internally it evaluates:

1. **Cost** using built‑in pricing tables or user‑supplied YAML.
2. **Policies** expressed as composable predicates.
3. **Risk score** from our Bayesian engine.
4. **Semantic context** retrieved from past incidents.

These pieces are assembled by `AzureInfrastructureSimulator` (or other cloud
adapters) and produce human‑readable justifications.

## Architecture diagram:  

```mermaid
flowchart TD
    subgraph Input["🔌 Input Sources"]
        Services[Agents / Services]
        Metrics[Metrics / Logs]
    end

    Services --> Signals[Observability Signals]
    Metrics --> Signals

    Signals --> Interpreter[ARF Reliability Interpreter]
    
    subgraph Engine["⚙️ ARF Core Engine"]
        Interpreter --> Risk[Bayesian Risk Engine]
        Risk --> Intent[Healing Intent Engine]
    end
    
    Intent --> Recovery[Recovery Actions]
    
    style Interpreter fill:#e1f5fe,stroke:#01579b
    style Risk fill:#fff3e0,stroke:#e65100
    style Intent fill:#e8f5e8,stroke:#1b5e20
```

---

## 📘 Quick‑Start Tutorial

For a hands‑on introduction to ARF, check out the **[Tutorial](TUTORIAL.md)**.
It covers installation, core concepts, step‑by‑step examples, multi‑agent
analysis, and how to run the interactive demo. A live demo is available at
[Hugging Face Space v4](https://huggingface.co/spaces/petter2025/Agentic-Reliability-Framework-v4).

---

## 🧠 Intelligence Engine

ARF’s core innovation lies in its **intelligence engine**, which blends
on‑the‑fly learning with deep offline analysis and contextual memory.

### Bayesian Online Learning with Conjugate Priors

To update risk continuously as outcomes arrive, ARF maintains a
Beta–Binomial model. The implementation uses per‑category Beta priors (OSS defaults are non‑uniform), and updates are performed by counting successes (s) and failures (f) and computing the posterior Beta parameters.

The posterior mean (predicted failure probability) is:

    E[p] = (alpha + s) / (alpha + beta + s + f)

Important: the OSS implementation ships with tuned, category‑specific priors (not a uniform α=β=1). Example defaults in the codebase include:

- database: Beta(α=1.5, β=8.0)
- network:  Beta(α=1.2, β=10.0)
- compute:  Beta(α=1.0, β=12.0)
- security: Beta(α=2.0, β=10.0)
- default:  Beta(α=1.0, β=10.0)

These priors are configurable in the code and chosen to reflect pessimistic priors for high‑impact categories (see `core/governance/risk_engine.py`).

### Offline Pattern Discovery via HMC (NUTS)

Certain factors (time of day, user role, environment, etc.) exhibit complex,
nonlinear interactions that are hard to capture online. For these we train a
logistic regression using Hamiltonian Monte Carlo with the No‑U‑Turn Sampler:

\[
\Pr(\text{failure}\mid x) = \sigma(w_0 + w_{\text{role}} + w_{\text{env}} 
  + w_{\sin\omega t} + w_{\cos\omega t} + \cdots),
\]

where the cyclical encoding `(sin, cos)` of time `t` captures daily
rhythms. The HMC implementation (via PyMC/ArviZ) learns a posterior over
weights; the model is serialized to `hmc_model.json` and hot‑loaded by the
simulator. The offline engine can be retrained periodically with new data.

### Hybrid Architecture

At lookup time the final risk score is a weighted blend of the online beta
estimate, an optional hierarchical hyperprior component, and the offline HMC prediction. Unlike a single fixed mixing parameter (λ), the OSS engine computes blending weights dynamically based on available data and configurable parameters. The runtime logic considers:

- the conjugate posterior mean (always available),
- an optional hyperprior summary (enabled only if Pyro is installed and configured),
- an HMC prediction (available when an HMC model has been trained and loaded).

Weights are determined by the amount of historical incidents and parameters exposed by the `RiskEngine` such as `n0` (HMC confidence threshold) and `hyperprior_weight`. See `core/governance/risk_engine.py` for the exact weighting rules and defaults.

### Semantic Memory

In the Enterprise edition incidents are typically embedded using
sentence‑transformers (for example `all-MiniLM-L6-v2`) and indexed with FAISS for hybrid vector search. In this OSS repository the RAG/graph memory uses a simplified, deterministic embedding fallback (no heavy ML dependencies) and an in‑memory FAISS IndexFlatL2 implementation for similarity search. This keeps the OSS package lightweight and runnable without large ML libraries; it also enforces an in‑memory FAISS backend and fixed embedding dim (see `core/config/constants.py`).

If you need full sentence‑transformers embeddings locally, install the optional dependencies (e.g., `sentence-transformers`, `torch`) or use the Enterprise integration that supports advanced embedding backends and FAISS index types (IVF/HNSW/PQ).

### Deterministic Probability Thresholding (DPT)

We introduce **DPT**, a decision rule that compares the posterior failure probability directly against fixed thresholds (for example `τ_low=0.2`, `τ_high=0.8`). The OSS core returns risk scores and explanations; how those scores are mapped to final actions (approve/deny/escalate) is typically applied by the calling layer (simulator or enterprise enforcement) using deterministic thresholds.

### Cyclical Time Encoding

Daily and weekly cycles are encoded using sine and cosine transforms

\[
\sin(2\pi t/24),\qquad \cos(2\pi t/24)
\]

to give the offline model sensitivity to time‑of‑day effects without
introducing discontinuities at midnight.

---

## 🔒 OSS vs Enterprise

The repository you are reading is the **OSS** slice, which implements all
advisory logic and exposes the public API. The enterprise layer (in a
different repo) wraps ARF with actual Azure/AWS/… clients, enforces decisions,
stores outcomes securely, and adds proprietary safety heuristics. To keep the
open‑source core safe:

- Intent outputs are marked `OSS_ADVISORY_ONLY` when external enforcement is
  required.
- Cost and policy modules contain no cloud credentials.
- Critical constants (e.g. `MAX_POLICY_VIOLATIONS`) are reviewed by the
  security team.
- **Enterprise features** include audit trails, approval workflows, blast
  radius limits, and integration with corporate logging. Contact us for docs
  and a demo.

Anyone can fork and run the OSS engine, but production enforcement requires
enterprise integration. For commercial inquiries or support, please reach out
via the **Contact** section below.

---

## 🔗 Links

- **Live demo:** https://huggingface.co/spaces/petter2025/Agentic-Reliability-Framework-v4
- **Legacy API demo:** https://huggingface.co/spaces/petter2025/Agentic-Reliability-Framework-API
- **Reliable AI Systems Stack (curated collection):** https://huggingface.co/collections/petter2025/reliable-ai-systems-stack
- **Full documentation:** docs/ (online version coming soon at
  https://docs.agentic-reliability-framework.io)


### 📚 Citation

If you use ARF in research, please cite:

```bibtex
@misc{arf2026,
  title={Agentic Reliability Framework (ARF)},
  author={Juan, Petter and contributors},
  year={2026},
  howpublished={\url{https://github.com/arf-foundation/agentic_reliability_framework}}
}
```


## 📬 Contact

- Email: [petter2025us@outlook.com](mailto:petter2025us@outlook.com)
- LinkedIn: [petterjuan](https://www.linkedin.com/in/petterjuan/)
- Book a call: [Calendly – 30 min](https://calendly.com/petter2025us/30min)


## 📦 Additional File: Example Pricing YAML (`pricing.yml`)

```yaml
vm:
  Standard_D2s_v3: 70.0
  Standard_D4s_v3: 140.0
  custom_extra_large: 999.0
storage_account:
  "50GB": 5.0
  "1TB": 100.0
```
