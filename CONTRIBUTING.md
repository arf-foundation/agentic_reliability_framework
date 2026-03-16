# Contributing to ARF

Thank you for your interest in contributing to the Agentic Reliability
Framework! This document outlines how to get started, coding conventions, and
the workflow for submitting changes.

If you have questions at any point, feel free to reach out to the project
maintainer:

- **Email:** petter2025us@outlook.com
- **LinkedIn:** https://www.linkedin.com/in/petterjuan/

---

## 🛠️ Development Environment

1. **Clone the repository**
   ```bash
   git clone git@github.com:petter2025us/agentic-reliability-framework.git
   cd agentic-reliability-framework
   ```

2. **Create and activate a Python environment** (the `environment.yml` file
describes a conda environment):
   ```bash
   conda env create -f environment.yml
   conda activate arf
   pip install -e .
   ```
   Alternatively, a plain `venv` is fine:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

3. **Install additional tooling** (optional but recommended):
   - `pre-commit` for formatting and linting hooks
   - `pytest` for running tests

4. **Run the test suite** to verify the base environment is working:
   ```bash
   pytest tests/infrastructure/  # or `pytest` for everything once name conflicts are fixed
   ```

---

## 📏 Coding Standards

The project follows these general guidelines:

* **Imports** should be absolute and start with
  `agentic_reliability_framework`. Avoid relative imports.
* **Type hints** are encouraged, especially on public functions and methods.
* **Logging** should use the standard `logging` module; do not print directly.
* **Immutability**: many domain objects are Pydantic models or frozen
  dataclasses; if you need a modified copy, use `model_copy(update={...})` or
  create a new instance.
* **Policy algebra** and other fluent APIs should return new objects rather
  than mutating existing ones.
* **Docstrings** should follow the NumPy/Google style. Public APIs should have
  docstrings that appear in the generated documentation.
* **Tests** should use `pytest` and the fixtures in
  `tests/infrastructure/conftest.py` when possible. New files go under the
  appropriate `tests/core/...`, `tests/infrastructure/...`, or
  `tests/runtime/...` directory.
* **Constants** (e.g. in `core/config/constants.py`) must be validated and
  documented; add new entries with care as they may affect OSS boundaries.

Refer to `.github/copilot-instructions.md` for additional hints about coding
patterns used in this repository.

---

## ✅ Testing Requirements

* Run the full suite locally before opening a pull request:
  ```bash
  pytest tests/infrastructure/  # includes core and runtime by default
  ```
* Aim for high coverage on modified modules; tests should exercise both
  happy paths and edge cases.
* Use `pytest.mark.asyncio` for asynchronous tests.
* When adding tests that depend on external heavy libraries (PyMC, ArviZ),
  consider mocking those modules to keep the suite fast.
* The CI configuration (GitHub Actions) runs `pytest -q`; ensure your tests
  pass in the editable install.

---

## 🔄 Workflow & Pull Requests

1. **Branch naming**: use `feature/<short-description>` or
   `bugfix/<short-description>`.
2. **Keep commits atomic** and write clear commit messages. Use the
   imperative tense (e.g. "Add policy combinator tests").
3. **Update documentation** when adding public APIs or changing behavior
   (README, docs/**/*.md, or inline comments).
4. **Run formatting/linting hooks** (if configured) before pushing:
   ```bash
   pre-commit run --all-files
   ```
5. **Open a pull request** against the `main` branch. Include a description of
   the change, why it's needed, and any relevant issue numbers.
6. **Address review comments** promptly. Small changes may be pushed to the
   same branch; larger refactors can be split into follow‑on PRs.
7. **Merge criteria**: at least one approval from a core maintainer, CI tests
   green, and no merge conflicts.

---

## 📚 Additional Resources

* Issue tracker: (https://github.com/arf-foundation/agentic-reliability-framework/issues)
* Discussions/roadmap: see the GitHub Discussions tab
* Architecture overview: refer to `agentic_reliability_framework/.github/copilot-instructions.md`

Thanks again for contributing! Your work helps make ARF more reliable and
accessible for everyone. 🚀

