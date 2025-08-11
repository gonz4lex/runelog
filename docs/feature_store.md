# The Lightweight Feature Store Pattern

This guide explains how to use `runelog` to create a simple, file-based "feature store" implementation. This pattern allows you to version your feature sets and create a clear, traceable lineage between your models and the exact features they were trained on, using only the core library features. While not as powerful as enterprise-level alternatives, it's a highly effective and storage-efficient pattern.

### Workflow

The process involves two decoupled scripts:

- A **feature engineering** script that generates a feature set, saves it to a known location, and logs a unique fingerprint (a SHA256 hash) of the data to a run.

- A **model training** script that discovers the latest feature set via a tag, verifies the integrity of the local data file against the logged fingerprint, and then trains a model.

This approach avoids duplicating large datasets, as only the metadata and hash are stored with the run, making it truly "lightweight."

#### The Fingerprint Method (Recommended)
##### Feature Engineering Script

This script processes data and uses log_dataset to record the feature set's metadata and hash. It then tags the run as latest for easy discovery.

**`examples/feature_store/make_features.py`**

```python
import os
from sklearn.datasets import load_iris
from runelog import get_tracker


def main():
    tracker = get_tracker()

    with tracker.start_run(experiment_name="feature-sets") as run_id:
        print(f"Creating a new feature set in run: {run_id}")

        # Load and process data
        iris = load_iris(as_frame=True)
        df = iris.frame
        df["sepal_area"] = df["sepal length (cm)"] * df["sepal width (cm)"]

        # Create a dedicated feature store folder
        output_dir = os.path.join("fstore", run_id)
        os.makedirs(output_dir, exist_ok=True)
        feature_path = os.path.join(output_dir, "iris_features.parquet")

        # Save the feature set to the new, specific path
        df.to_parquet(feature_path)

        # Log the artifact from that path
        tracker.log_artifact(feature_path)

        print(f"Feature set saved to '{feature_path}' and logged.")

        # Optional: Tag this run as the latest
        tracker.set_run_tags({"latest_feature_set": True})


if __name__ == "__main__":
    main()
```

Run this script with `python examples/feature_store/make_features.py` or use the CLI: `runelog examples features`. This will create a new run with a unique ID that represents a versioned feature set.

#### Create the Model Training Script

This script discovers the latest feature run, reads its logged dataset metadata, and verifies the local data file against the stored hash before training.

**`examples/feature_store/train_with_fs.py`**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from runelog import get_tracker, exceptions

# The experiment where feature generation runs are stored
FEATURE_EXPERIMENT = "feature-sets"

# The consistent filename of the artifact we want to use
ARTIFACT_FILENAME = "iris_features.parquet"


def find_latest_feature_run(tracker) -> str:
    """Finds the most recent run in the feature experiment."""
    try:
        all_runs_df = tracker.load_results(FEATURE_EXPERIMENT)
        if all_runs_df.empty:
            print(f"Error: No runs found in experiment '{FEATURE_EXPERIMENT}'.")
            return None

        all_runs_df["start_time"] = pd.to_datetime(all_runs_df["start_time"])
        sorted_runs = all_runs_df.sort_values(by="start_time", ascending=False)

        latest_run_id = sorted_runs.index[0]
        print(f"Found latest feature set from run: {latest_run_id}")
        return latest_run_id

    except exceptions.ExperimentNotFound:
        print(f"Error: Experiment '{FEATURE_EXPERIMENT}' not found.")
        return None


def main():
    """
    Finds the latest feature set, trains a model, and logs the results.
    """
    tracker = get_tracker()

    feature_run_id = find_latest_feature_run(tracker)
    if not feature_run_id:
        print("Could not find a feature set to train on. Exiting.")
        return

    # Download the artifact using its consistent filename
    print(f"Downloading artifact '{ARTIFACT_FILENAME}' from run '{feature_run_id}'...")
    feature_path = tracker.download_artifact(feature_run_id, ARTIFACT_FILENAME)
    df = pd.read_parquet(feature_path)

    # Start a new run for the model training
    with tracker.start_run(experiment_name="production-models"):

        # Track lineage using the artifact's filename
        tracker.log_input_run(
            name="feature_set", run_id=feature_run_id, artifact_name=ARTIFACT_FILENAME
        )

        # Train a model on the versioned data
        X = df.drop(columns=["target"])
        y = df["target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        params = {"C": 50, "max_iter": 10_000}

        model = LogisticRegression(**params).fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))

        tracker.log_metric("accuracy", accuracy)
        print(f"Model trained with accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
```

Run this script with `python examples/feature_store/train_with_fs.py` or use the CLI: `runelog examples train-fs`.

You now have a model training run that is permanently and traceably linked to the exact version of the features it was trained on.

#### Alternative Workflows
##### Using Full Artifacts
For smaller datasets or when you need fully self-contained runs, you can log the entire dataset as an artifact.

In `make_features.py`, replace `tracker.log_dataset(...)` with `tracker.log_artifact(...)`:

```python
# make_features.py
# ...
df.to_parquet("iris_features.parquet")
# Log the entire file as an artifact
tracker.log_artifact("iris_features.parquet")
```

The training script would then download the artifact directly instead of verifying a local file.

```python
# In train_with_fs.py
# ...
# Download the artifact instead of verifying a local file
feature_path = tracker.download_artifact(feature_run_id, "iris_features.parquet")
df = pd.read_parquet(feature_path)

with tracker.start_run(experiment_name=MODEL_EXPERIMENT):
    # Use log_input_run for a direct artifact-to-run link
    tracker.log_input_run(
        name="feature_set",
        run_id=feature_run_id,
        artifact_name="iris_features.parquet"
    )
    # ... rest of training code
```
##### Integrating with DVC

If you already use DVC to version your data, you can use `log_dvc_input` to create the link. This method finds the `.dvc` file, extracts the version hash (MD5), and logs it.

Simply replace log_dataset in your feature script:

```python
# In make_features.py, after saving the data and running `dvc add data/iris_features.parquet`

# tracker.log_dataset(FEATURE_PATH, name="iris_features") # Old method
tracker.log_dvc_input(FEATURE_PATH, name="iris_features") # New method
print(f"DVC version info for '{FEATURE_PATH}' was logged.")
```
