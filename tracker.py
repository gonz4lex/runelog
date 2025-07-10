import os
import json
import uuid
import joblib
import shutil
from contextlib import contextmanager


class Tracker:
    """
    A lightweight tracker for ML experiments that handles creating experiments,
    managing runs, and logging parameters, metrics, and artifacts to the
    local filesystem.
    """

    def __init__(self, path="."):
        """
        Initializes the tracker.

        Args:
            path (str): The root directory for storing experiments.
        """
        self.root_path = os.path.abspath(path)
        self._mlruns_dir = os.path.join(self.root_path, ".mlruns")
        self._registry_dir = os.path.join(self.root_path, ".registry")
        self._active_run_id = None
        self._active_experiment_id = None

        os.makedirs(self._mlruns_dir, exist_ok=True)

    def _get_run_path(self):
        """Helper to get the path of the current active run."""
        if not self._active_run_id:
            raise Exception("No active run. Use start_run() context manager.")
        return os.path.join(
            self._mlruns_dir, self._active_experiment_id, self._active_run_id
        )

    # Experiments and runs

    def get_or_create_experiment(self, name: str) -> str:
        """
        Creates a new experiment and returns its ID. If an experiment with
        the same name already exists, it returns the existing ID.
        """
        for experiment_id in os.listdir(self._mlruns_dir):
            meta_path = os.path.join(self._mlruns_dir, experiment_id, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    if json.load(f).get("name") == name:
                        return experiment_id

        experiment_id = str(len(os.listdir(self._mlruns_dir)))
        experiment_path = os.path.join(self._mlruns_dir, experiment_id)
        os.makedirs(experiment_path, exist_ok=True)

        meta = {"experiment_id": experiment_id, "name": name}
        with open(os.path.join(experiment_path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=4)

        return experiment_id

    @contextmanager
    def start_run(self, experiment_id: str = "0"):
        """
        Starts a new run within an experiment using a context manager.
        """
        # Ensure the default experiment '0' exists
        default_experiment_path = os.path.join(self._mlruns_dir, "0")
        if not os.path.exists(default_experiment_path):
            self.get_or_create_experiment("default")

        self._active_experiment_id = experiment_id
        self._active_run_id = uuid.uuid4().hex[:8]  # Short unique ID

        run_path = self._get_run_path()
        os.makedirs(os.path.join(run_path, "params"), exist_ok=True)
        os.makedirs(os.path.join(run_path, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(run_path, "artifacts"), exist_ok=True)

        meta = {
            "run_id": self._active_run_id,
            "experiment_id": self._active_experiment_id,
            "status": "RUNNING",
        }
        with open(os.path.join(run_path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=4)

        try:
            yield self._active_run_id
        finally:
            meta["status"] = "FINISHED"
            with open(os.path.join(run_path, "meta.json"), "w") as f:
                json.dump(meta, f, indent=4)
            self._active_run_id = None
            self._active_experiment_id = None

    # Logging

    def log_param(self, key: str, value):
        """Logs a single parameter for the active run."""
        run_path = self._get_run_path()
        param_path = os.path.join(run_path, "params", f"{key}.json")
        with open(param_path, "w") as f:
            json.dump({"value": value}, f, indent=4)

    def log_metric(self, key: str, value: float):
        """Logs a single metric for the active run."""
        run_path = self._get_run_path()
        metric_path = os.path.join(run_path, "metrics", f"{key}.json")
        with open(metric_path, "w") as f:
            json.dump({"value": value}, f, indent=4)

    def log_artifact(self, local_path: str):
        """Logs a local file as an artifact of the active run."""
        run_path = self._get_run_path()
        artifact_dir = os.path.join(run_path, "artifacts")
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Artifact not found at: {local_path}")
        shutil.copy(local_path, artifact_dir)

    def log_model(self, model, name: str):
        """Logs a trained model as an artifact of the active run."""
        run_path = self._get_run_path()
        model_path = os.path.join(run_path, "artifacts", name)
        joblib.dump(model, model_path)

    # Reading

    def get_run(self, run_id: str):
        """
        Loads the parameters and metrics for a specific run.
        NOTE: Assumes run_id is unique across all experiments for now.
        """
        for experiment_id in os.listdir(self._mlruns_dir):
            run_path = os.path.join(self._mlruns_dir, experiment_id, run_id)
            if os.path.isdir(run_path):
                # Load params
                params = {}
                params_path = os.path.join(run_path, "params")
                for param_file in os.listdir(params_path):
                    key = os.path.splitext(param_file)[0]
                    with open(os.path.join(params_path, param_file), "r") as f:
                        params[f"param_{key}"] = json.load(f)["value"]

                # Load metrics
                metrics = {}
                metrics_path = os.path.join(run_path, "metrics")
                for metric_file in os.listdir(metrics_path):
                    key = os.path.splitext(metric_file)[0]
                    with open(os.path.join(metrics_path, metric_file), "r") as f:
                        metrics[key] = json.load(f)["value"]

                return {"run_id": run_id, **params, **metrics}
        return None

    def load_results(self, experiment_id: str):
        """
        Loads all run metrics and parameters from a given experiment into a
        pandas DataFrame for comparison.
        """
        import pandas as pd

        experiment_path = os.path.join(self._mlruns_dir, experiment_id)
        if not os.path.exists(experiment_path):
            raise FileNotFoundError(f"Experiment with ID '{experiment_id}' not found.")

        all_runs_data = []
        for run_id in os.listdir(experiment_path):
            # Skip metadata file, only process run directories
            if os.path.isdir(os.path.join(experiment_path, run_id)):
                run_data = self.get_run(run_id)
                if run_data:
                    all_runs_data.append(run_data)

        if not all_runs_data:
            return pd.DataFrame()

        return pd.DataFrame(all_runs_data).set_index("run_id").sort_index()

    # Model Registry

    def register_model(self, run_id: str, artifact_name: str, model_name: str, tags: dict = None):
        """
        Registers a model from a run's artifacts to the model registry.

        Args:
            run_id (str): The ID of the run where the model artifact is stored.
            artifact_name (str): The filename of the model artifact (e.g., "model.pkl").
            model_name (str): The name to register the model under.
        """
        # Find the model artifact
        run_path = None
        for exp_id in os.listdir(self._mlruns_dir):
            path = os.path.join(self._mlruns_dir, exp_id, run_id)
            if os.path.isdir(path):
                run_path = path
                break

        if not run_path:
            raise FileNotFoundError(f"Run with ID '{run_id}' not found.")

        source_artifact_path = os.path.join(run_path, "artifacts", artifact_name)
        if not os.path.exists(source_artifact_path):
            raise FileNotFoundError(
                f"Artifact '{artifact_name}' not found in run '{run_id}'."
            )

        # Create the destination directory in the registry
        registry_model_path = os.path.join(self._registry_dir, model_name)
        os.makedirs(registry_model_path, exist_ok=True)

        # Determine the new version number
        existing_versions = [d for d in os.listdir(registry_model_path) if d.isdigit()]
        new_version = str(max([int(v) for v in existing_versions] or [0]) + 1)

        version_path = os.path.join(registry_model_path, new_version)
        os.makedirs(version_path, exist_ok=True)

        # Copy the model and generate metadata
        shutil.copy(source_artifact_path, os.path.join(version_path, "model.joblib"))

        meta = {
            "model_name": model_name,
            "version": new_version,
            "source_run_id": run_id,
            "registration_timestamp": __import__("datetime").datetime.now().isoformat(),
            "tags": tags or {}
        }
        with open(os.path.join(version_path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=4)

        print(
            f"Successfully registered model '{model_name}' with version {new_version}."
        )
        return new_version

    def load_registered_model(self, model_name: str, version: str = "latest"):
        """
        Loads a model from the model registry.

        Args:
            model_name (str): The name of the registered model.
            version (str): The version to load ('latest' or a specific version number).

        Returns:
            The loaded model object.
        """
        model_path = os.path.join(self._registry_dir, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model '{model_name}' not found in registry.")

        if version == "latest":
            versions = [d for d in os.listdir(model_path) if d.isdigit()]
            if not versions:
                raise FileNotFoundError(f"No versions found for model '{model_name}'.")
            latest_version = str(max([int(v) for v in versions]))
            version_to_load = latest_version
        else:
            version_to_load = version

        final_model_path = os.path.join(model_path, version_to_load, "model.joblib")
        if not os.path.exists(final_model_path):
            raise FileNotFoundError(
                f"Version '{version_to_load}' not found for model '{model_name}'."
            )

        return joblib.load(final_model_path)
    

    def add_model_tags(self, model_name: str, version: str, tags: dict):
        """
        Adds tags to an existing registered model version.

        Args:
            model_name (str): The name of the registered model.
            version (str): The model version to add tags to.
            tags (dict): A dictionary of tags to add or update.
        """
        version_path = os.path.join(self._registry_dir, model_name, version)
        meta_path = os.path.join(version_path, "meta.json")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Model '{model_name}' version '{version}' not found.")

        with open(meta_path, "r+") as f:
            meta = json.load(f)
            if "tags" not in meta:
                meta["tags"] = {}
            meta["tags"].update(tags) # Add or overwrite tags
            
            f.seek(0) # Rewind to the beginning of the file
            json.dump(meta, f, indent=4)
            f.truncate() # Remove any trailing content if the new file is shorter


    def get_model_tags(self, model_name: str, version: str) -> dict:
        """
        Retrieves the tags for a specific registered model version.

        Args:
            model_name (str): The name of the registered model.
            version (str): The model version to add tags to.
        """
        version_path = os.path.join(self._registry_dir, model_name, version)
        meta_path = os.path.join(version_path, "meta.json")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Model '{model_name}' version '{version}' not found.")

        with open(meta_path, "r") as f:
            meta = json.load(f)
            return meta.get("tags", {})
