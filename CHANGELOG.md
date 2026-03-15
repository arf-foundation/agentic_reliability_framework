# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.2.0+oss] - 2026-03-15

### Added
- **Canonical governance loop** (`governance_loop.py`) – a single entry point orchestrating policy, cost, reliability signals, and Bayesian risk into a comprehensive `HealingIntent`.
- **Expanded `HealingIntent` (v2.1.0)** – adds 30+ optional fields for:
  - Bayesian uncertainty (`posterior_mean`, `credible_interval`, `epistemic_uncertainty`)
  - Evidence provenance (`evidence_lift`, `contradiction_score`, `evidence_sources`)
  - Human factors (`ambiguity_score`, `operator_attention_required`, `explanation_depth`)
  - Learning hooks (`feedback_expected`, `replay_key`, `experiment_tags`)
  - Full backward compatibility – old intents still deserialize correctly.
- **Deterministic Probability Thresholds (DPT)** – new constants `DPT_LOW = 0.2` and `DPT_HIGH = 0.8` for transparent approve/deny/escalate decisions.
- **Backward‑compatible event model** – `ReliabilityEvent` now accepts legacy fields (`component`, `latency_p99`, `error_rate`, etc.) and maps them into the canonical `metrics` dictionary. Existing code that reads `event.latency_p99` continues to work via property aliases.
- **Metric resolver** – `resolve_metric(event, name)` safely retrieves numeric values even when the attribute is a property, eliminating type errors in comparisons and formatting.
- **`requirements.txt`** – lists all core dependencies for easy installation.
- **`CITATION.cff`** – added for academic citation.

### Changed
- `HealingIntent` version bumped to `2.1.0` (backward‑compatible).
- Improved test coverage to 50% overall, with critical paths >80%.
- Updated `README.md` with v4.2.0 highlights.

### Fixed
- Type errors when comparing or formatting event metrics (now uses `resolve_metric`).
- Import errors in `governance_loop.py` (deferred research import).
- `frozen_instance` errors by using pre‑validator instead of post‑validation mutation.
- `None` handling in `rag_graph.py` for deterministic incident IDs and embeddings.

- # Changelog

All notable changes to this project will be documented in this file.

---

## [4.0.1+oss] - 2026-03-05

### Added
- FAISS/RAG memory module (`runtime/memory/`) for incident storage, similarity search, and outcome tracking.
- OSS limits enforced via `core/config/constants.py` (max incident nodes, in-memory only).
- 9 unit tests for memory module.

### Changed
- Updated `README.md` and `TUTORIAL.md` to reflect per-category priors and dynamic blending.
- Version bumped to `4.0.1+oss` (PEP440-compliant).

### Fixed
- Deprecated `asyncio.get_event_loop()` calls in async methods.
- Cache memory leak in `rag_graph.py` and `oss_client.py` (LRU eviction).
- Circular import in `oss_config.py`.

---

## [4.0.0] - 2026-02-28

### Initial Release
- Core governance framework
- MCP client integration
- Multi-agent runtime architecture


# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.2.0] - 2026-03-15

### Added
- Canonical governance loop (`governance_loop.py`) orchestrating policy, cost, reliability, and risk signals.
- 30+ new optional fields in `HealingIntent` for Bayesian uncertainty, evidence provenance, human factors, and learning hooks (v2.1.0).
- Deterministic Probability Thresholds (`DPT_LOW`, `DPT_HIGH`) in `constants.py`.
- Backward‑compatible `ReliabilityEvent` with property aliases for legacy fields.
- `resolve_metric` helper to safely retrieve numeric metric values from events.
- `requirements.txt` listing core dependencies.
- `CITATION.cff` for academic citation.

### Changed
- `HealingIntent` version bumped to `2.1.0` (backward‑compatible).
- Improved test coverage to 50% overall.
- Updated `README.md` with v4.2.0 highlights.

### Fixed
- Type errors when comparing or formatting event metrics (now uses `resolve_metric`).
- Import errors in `governance_loop.py` (deferred research import).
- `frozen_instance` errors by using pre‑validator instead of post‑validation mutation.
- None handling in `rag_graph.py` for deterministic incident IDs and embeddings.
