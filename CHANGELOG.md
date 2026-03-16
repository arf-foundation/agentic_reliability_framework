# Changelog

All notable changes to the Agentic Reliability Framework (ARF) are documented in this file.

The format follows Keep a Changelog principles and Semantic Versioning.

---

## [4.2.0+oss] - 2026-03-16

### Added
Canonical governance loop for advisory decision workflows:

Intent → Simulation → Risk Evaluation → Policy Analysis → Decision → HealingIntent

Expanded `HealingIntent` schema with 30+ fields to capture:

- uncertainty metrics
- supporting evidence
- remediation parameters
- escalation indicators
- decision justification
- governance metadata

Formalized Deterministic Probability Thresholding (DPT):

Approve if P(failure) < 0.2  
Escalate if 0.2 ≤ P(failure) ≤ 0.8  
Deny if P(failure) > 0.8

Additional predictive engine tests and stability improvements.

### Improved

Risk engine explanations now provide clearer breakdowns of probabilistic reasoning.

Governance pipeline produces more structured advisory output.

Predictive analytics module stability improvements.

Improved validation logic for infrastructure intents.

### Testing

Expanded coverage across governance and predictive modules.

Current coverage approximately 50%.

All existing tests pass successfully.

### Compatibility

Fully backward compatible with v4.0.x.

---

## [4.0.1+oss] - 2026-03-05

### Added

FAISS/RAG memory module (`runtime/memory/`) for:

- incident storage
- similarity search
- outcome tracking

OSS runtime limits enforced via `core/config/constants.py`.

9 unit tests added for the memory subsystem.

### Changed

README.md and TUTORIAL.md updated with improved explanations of per-category Bayesian priors and hybrid model blending.

Version bumped to 4.0.1+oss (PEP440 compliant).

### Fixed

Deprecated `asyncio.get_event_loop()` usage replaced with `get_running_loop()`.

Cache memory leak fixed in `rag_graph.py` and `oss_client.py`.

Circular import issue resolved in `oss_config.py`.

---

## [4.0.0] - 2026-02-28

### Initial Release

Core governance framework.

Hybrid Bayesian risk engine.

MCP client integration.

Multi-agent runtime architecture.
