# RuneLog

[![Build Status](https://github.com/gonz4lex/runelog/actions/workflows/tests.yml/badge.svg)](https://github.com/gonz4lex/runelog/actions/workflows/tests.yml)
[![Codecov](https://codecov.io/gh/gonz4lex/runelog/branch/develop/graph/badge.svg)](https://codecov.io/gh/gonz4lex/runelog)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/gonz4lex/runelog)
[![Docs](https://github.com/gonz4lex/runelog/actions/workflows/deploy-docs.yml/badge.svg)](https://github.com/gonz4lex/runelog/actions/workflows/deploy-docs.yml)

<!-- [![PyPI version](https://badge.fury.io/py/runelog.svg)](https://badge.fury.io/py/runelog) -->

## Lightweight ML Tracker

A simple, file-based Python library for tracking machine learning experiments, inspired by MLflow.

**RuneLog is in active development**. The core API is functional but subject to change.

The name *RuneLog* is a play on words. It's reminiscent of the common `run.log()` command used to log an experiment in tracking systems, while also treating these powerful, and sometimes mysterious, models as modern-day mystical writings: a "log of runes".


##  Why RuneLog?

- **Zero-Overhead Setup**: start tracking runs in a single line
- **Local-First, Lightweight**: perfect for solo devs or small teams
- **Portable & Transparent**: data is stored in simple folders/files

##  Installation

### User Setup

This is the recommended way to install `runelog` if you just want to use it in your projects.

1. Make sure you have Python 3.10+ installed.
2. Install the library from PyPI using pip:

```bash
pip install runelog
```

That's it! You can now import it into your Python scripts.

#### Quickstart

Start tracking your ML experiments in just a few lines of code:

```python
from runelog import get_tracker

# 1. Initialize the tracker
tracker = get_tracker(experiment_name="my-first-experiment")

# 2. Start a run
with tracker.start_run():
    # 3. Your training code and logging calls go here
    tracker.log_metric("accuracy", 0.95)
```

Check the detailed [Quickstart Guide](./docs/quickstart.md) for for a complete runnable example.

### Development Setup

1. Clone the repository:

```bash
git clone https://github.com/gonz4lex/runelog.git
cd runelog
```
2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
3. Install in Editable Mode:
Install the package and all development dependencies from your pyproject.toml file.

```bash
pip install -e .[dev]
```

#### Development Workflow

RuneLog uses **[Hatch](https://hatch.pypa.io/)** to manage dependencies, environments, and scripts.

1. Environment Setup: the primary command to enter a fully configured development environment is `hatch shell`. This will create a virtual environment, install all dependencies listed in `pyproject.toml`, and activate it.

```bash
hatch shell
```

2. Formatting Code

RuneLog uses **Black** for automatic code formatting. To format all project files, run the following command:

```bash
hatch run lint:format
```

3. Running Tests

The test suite is run using `pytest` and the `test` environment is configured to automatically run tests with coverage:

```bash
hatch run test
```

This command is a shortcut for `pytest --cov=src/runelog --cov-report=term-missing --cov-report=xml`.

4. Building the Documentation

To preview the documentation site locally with live reloading, you can add a `docs` environment to your `pyproject.toml`.

```bash
hatch run docs:serve
```

5. Building the Package

To build the distributable package files (`.whl` and `.tar.gz`) just run:

```bash
hatch build
```

The files will be created in the `dist/` directory, ready for publishing.


### Usage Examples
You can find example scripts in the `examples/ directory`:

#### `train_model.py`

Full pipeline example with:
* logging parameters and metrics
* saving and registering models
* tagging and retrieving models

```bash
python examples/train_model.py  # or use the CLI: runelog examples train
```

#### `minimal_tracking.py`

Minimal working example with only metric logging.

```bash
python examples/minimal_tracking.py  # or use the CLI: runelog examples minimal
```

#### `sweep/sweep.py`

Sweep example with configuration file to define runs.

```bash
python examples/sweep/sweep.py # or use the CLI: runelog examples sweep
```

---

### Features
- **Core Tracking API**: Experiments, runs, parameters, metrics.
- **Artifact Logging**: Save model files, plots, and other artifacts.
- **Model Registry**: Version and tag models.
- **Streamlit UI**: Interactive dashboard to explore runs and the registry.
- **Command-Line Interface (CLI)** for programmatic interaction.
- **Full Test Coverage**: Comprehensive unit and integration tests.


### 🐳 Running the UI with Docker

The easiest way to run the Runelog web UI without setting up a local Python environment is with Docker. You must have [Docker](https://www.docker.com/products/docker-desktop/) installed and running.

#### Instructions

1.  Build the Docker image from the root of the project directory:
    ```bash
    docker build -t runelog-app .
    ```

2.  Use `docker-compose` to start the application:
    ```bash
    docker-compose up
    ```

3.  To access the UI, open your web browser and navigate to:
    **[http://localhost:8501](http://localhost:8501)**

4.  To stop the application, press `Ctrl+C` in the terminal, and then run:
    ```bash
    docker-compose down
    ```

