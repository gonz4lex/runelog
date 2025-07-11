# Runelog
## Lightweight ML Tracker

A simple, file-based Python library for tracking machine learning experiments, inspired by MLflow.

**Current Status**: 🚧 In active development. The core API is functional but subject to change.

The name *Runelog* is a play on words. It evokes the common `run.log()` command used to log an experiment, while also treating your powerful, and sometimes mysterious, models as modern-day mystical writings: a "log of runes".

## Guiding Philosophy

This project is guided by a local-first and lightweight philosophy. The goal is to provide a simple, intuitive, and dependency-light tool for individual developers and small teams to track experiments without the overhead of a database, a complex server, or cloud services. Every design choice prioritizes simplicity and ease of use.

To maintain the project's lightweight nature, there are several features I will deliberately not implement. If you need these, a more feature-rich tool like the full MLflow is a better choice.

- **No Database**: The library will only ever support the local file system for storing tracking data. This avoids heavy dependencies like SQLAlchemy and the complexity of database migrations.
- **No Users**: This is a local tool, not a multi-tenant service.
- **No Complex Web Server**: The optional UI will be a simple Streamlit application, not a persistent, production-grade server like Flask or Django.
- **No Cloud Integration**: All artifacts are stored locally. The library will not have built-in support for saving to S3, GCS, or Azure Blob Storage.

## Why Runelog?

- Zero-Overhead Setup: start tracking runs within a single line of code
- Ideal for Local Development and Learning: perfect for practitioners working on solo or small projects in their local machines
- Full Transparency and Portability: data is stored in simple files and folders that users can see, understand and even version control.

## Features Roadmap

### Implemented
- Core Tracking API: Create experiments and runs on the local file system.
- Comprehensive Logging: Log parameters, metrics, model files, and other artifacts (e.g., plots, data files).
- Results API: Load experiment results directly into pandas DataFrames for easy analysis and comparison.
- Model Registry: A simple, file-based registry to version and manage models.
    - Model Tagging: Organize runs by arbitrary dimensions.
- Streamlit UI: An interactive dashboard to visualize experiments, compare runs, and view the model registry.

### Planned 

- Testing: A full suite of unit and integration tests.


## Setup & Installation

To get started, clone the repository and set up the Python environment.

1.  **Create and activate a virtual environment**:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

-----

## Quickstart

Use the `Tracker` class to log your model training process. The following example demonstrates how to create an experiment, start a run, log parameters and metrics, and save a model.

```python
# train.py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from runelog import Tracker

# 1. Initialize the tracker
tracker = Tracker()

# 2. Create or get an experiment
experiment_id = tracker.create_experiment("Example")

# 3. Start a new run
with tracker.start_run(experiment_id=experiment_id):
    
    ## ML Code
    X, y = make_classification(n_samples=1000, n_features=20)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Define and log model parameters
    params = {"C": 0.5, "solver": "liblinear"}
    tracker.log_param("C", params["C"])
    tracker.log_param("solver", params["solver"])

    # Train the model
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    # Make predictions and log metrics
    accuracy = accuracy_score(y_test, model.predict(X_test))
    tracker.log_metric("accuracy", accuracy)
    
    # Log the trained model
    tracker.log_model(model, "logistic_regression_model.pkl")

    # Register a model
    registered_model_name = "winner-model"
    tracker.register_model(run_id, model_artifact_name, registered_model_name)

    # Load a registered model
    loaded_model = tracker.load_registered_model(registered_model_name)
    print("Model loaded successfully:", loaded_model)

print("Training script finished.")

# Load and display the results for the experiment
results = tracker.load_results(experiment_id)
print(results)
```