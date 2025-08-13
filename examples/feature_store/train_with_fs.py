"""
Example script for "consumer" side of the Lightweight Feature Store pattern.

This script trains a model using a versioned feature set created by a
separate script (e.g., make_features.py). It demonstrates how to create a
traceable lineage between a model and the exact data it was trained on.
"""

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
