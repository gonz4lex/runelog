"""Example script for the "producer" side of the Lightweight Feature Store pattern.

This script demonstrates how to create a versioned feature set. It performs
the following steps:
1.  Loads the raw Iris dataset.
2.  Performs a simple feature engineering step (calculating sepal area).
3.  Saves the resulting DataFrame to a unique, traceable folder in `fstore/`
    named after the run's ID.
4.  Uses runelog to log the feature set as a versioned artifact.
5.  Tags the run as the 'latest_feature_set' for easy retrieval by other scripts.
"""

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
        fstore_dir = "../data/fstore/"
        output_dir = os.path.join(fstore_dir, run_id)
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
