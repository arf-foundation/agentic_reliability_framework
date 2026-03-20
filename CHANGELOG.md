# Changelog

All notable changes to the Agentic Reliability Framework (ARF) are documented in this file.

The format follows Keep a Changelog principles and Semantic Versioning.

---

# ARF Update Summary 2026-03-20

## What changed today

The governance loop was aligned with the finalized `HealingIntent` contract. The loop now creates intents through `from_infrastructure_intent`, which preserves the full provenance trail from the original `InfrastructureIntent` into the immutable advisory output.

The predictive analytics engine and test suite were already compatible with the new contract. The remaining work was confirmation, not redesign: the decision flow, metadata shape, and validation behavior are now consistent across the governance layer, predictive layer, and tests.

## Key impact

The system now preserves a complete audit chain from input intent to advisory output. That matters because ARF depends on traceability at the boundary between analysis and execution. The updated flow keeps the advisory layer deterministic, immutable, and explainable while retaining all infrastructure context needed for downstream review.

## Validation status

The latest code path is consistent with the finalized contract. The workflow history shows the most recent test update run and the earlier predictive baseline both completed successfully. The current state is green.

## Practical meaning

This update strengthens ARF as a control plane, not just a risk scorer. It improves the quality of decision evidence, makes audits more reliable, and reduces the chance of provenance loss between components.


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
