# Changelog

All notable changes to **RuneLog** will be documented in this file.

---

## [Unreleased]

### Planned

- Integration with Git metadata (hash, commit time, branch)
- `runelog serve` command to deploy modfels as a local API


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
