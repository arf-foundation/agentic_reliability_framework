# Changelog

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
