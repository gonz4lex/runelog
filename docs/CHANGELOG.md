# Changelog

All notable changes to **RuneLog** will be documented in this file.

---

## [Unreleased]

### Planned

- Integration with Git metadata (hash, commit time, branch)
- `runelog serve` command to deploy modfels as a local API
- Lightweight feature store implementation
- More visualizations options, i.e. ROC curves, feature importances, confusion matrices
- Extensible plugin architecture for custom trackers or visualizations

## [0.1.1] - 2025-08-04

### Added
- The `start_run` method now accepts an `experiment_name` directly, simplifying the most common user workflow.
- Added `delete_run` method to the core library for better cleanup and management.
- Added `runs delete` command with interactive confirmation prompts for safety.
- Improved empty state of the Streamlit UI for fresh runs, showing a brief quickstart when no experiments exist yet.

### Fixed
- Corrected the development installation instructions in the `README.md` and contribution guides.

### Changed
- Refactored `pyproject.toml` to use a pure `hatchling` build backend and a new `Hatch` environments for `docs`.
- Changed usage examples in `examples/` to reflect API changes.


## [0.1.0] – 2025-07-30
### 🎉 Initial Release
#### Core Library
- **Experiment Tracking**: `RuneLog` class for managing experiments and runs. Supports logging parameters, metrics, artifacts, and models.
- **Model Registry**: Full-featured model registry with versioning and tagging.
- **Sweep Runner**: `run_sweep` function for automated experiments from a flexible YAML configuration file.
- **Custom Exceptions**: A full suite of specific exceptions for robust error handling.

#### Command-Line Interface (CLI)
- A full-featured CLI powered by `Typer` and `rich`.
- **`runelog experiments`**: `list`, `get`, `delete` or `export` experiments to CSV.
- **`runelog runs`**: `list`, `get`, `compare` runs side-by-side, and `download-artifact`.
- **`runelog registry`**: `list` models, `get-versions`, `register` a model, and `tag` versions.
- **`runelog sweep`**: Execute a sweep from a config file.
- **`runelog ui`**: Launch the web UI.
- **`runelog examples`**: Commands to run example scripts.

#### Web UI (Streamlit)
- **Experiment Explorer**: View experiments and runs with a detailed drill-down view.
- **Visual Run Comparison**: Select multiple runs to see an interactive bar chart comparing their performance.
- **Artifact Previewer**: Render common artifact types like images and text files directly in the UI.
- **Model Registry Viewer**: Browse registered models and their versions.
- **Register from UI**: A button in the run detail view to register a model directly.

#### Project & Development
- **Professional Project Structure**: Uses a `src`-layout managed by `Hatch`.
- **Testing**: Comprehensive test suite using `pytest`, including unit and integration tests.
- **Docker Support**: `Dockerfile` and `docker-compose.yml` to easily build and share the UI.
- **Documentation**: A full documentation site built with `mkdocs`.
- **Community Files**: `LICENSE`, `CONTRIBUTING.md`, and `CODE_OF_CONDUCT.md`.
