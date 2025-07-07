## Runelog
## Lightweight ML Tracker

A simple, file-based Python library for tracking machine learning experiments, inspired by MLflow.

**Current Status**: 🚧 In active development. The core API is functional but subject to change.

-----

## Core Features

  * **Experiment Tracking**: Organize your work into experiments and runs.
  * **Logging**: Save model parameters, metrics, artifacts, and trained models to your local file system.
  * **Results API**: Load results from an experiment into a pandas DataFrame for easy analysis and comparison.

-----

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
from ml_tracker import Tracker

# 1. Initialize the tracker
tracker = Tracker()

# 2. Create or get an experiment
experiment_id = tracker.create_experiment("Example")

# 3. Start a new run
with tracker.start_run(experiment_id=experiment_id):
    
    # --- ML Code ---
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

print("Training script finished.")

# Load and display the results for the experiment
results = tracker.load_results(experiment_id)
print(results)
```