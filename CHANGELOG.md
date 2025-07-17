# Changelog

All notable changes to **RuneLog** will be documented in this file.

---

## [0.1.0] – 2025-07-10

### 🎉 Initial Release

- Implemented core `RuneLog` class for managing experiments and runs.
- Support for:
  - Logging parameters and metrics
  - Logging and retrieving artifacts
  - Model saving with `joblib`
- Added:
  - Run context management (`start_run`)
  - Experiment creation and listing
  - Run data retrieval (`get_run_details`, `load_results`)
- Model Registry:
  - Register model with versioning
  - Load specific or latest model version
  - Add and retrieve model tags
  - List models and their versions
- Utilities:
  - `get_run`, `get_model_versions`, `list_registered_models`
- Structure:
  - `runelog/runelog.py`: full implementation
  - `runelog/__init__.py`: exposed `Tracker`
  - `runelog/cli.py`: placeholder for upcoming CLI tools
- Examples:
  - `examples/train_model.py` (full pipeline)
  - `examples/minimal_tracking.py` (quick demo)

---

## [Unreleased]

### 🔜 Planned

- CLI support via `cli.py`
- Streamlit UI dashboard for browsing runs and metrics
- Full test suite with `pytest`
- Integration with Git metadata (hash, commit time, branch)
