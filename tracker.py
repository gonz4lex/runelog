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
        self._active_run_id = None
        self._active_experiment_id = None
        
        os.makedirs(self._mlruns_dir, exist_ok=True)

    def _get_run_path(self):
        """Helper to get the path of the current active run."""
        if not self._active_run_id:
            raise Exception("No active run. Use start_run() context manager.")
        return os.path.join(self._mlruns_dir, self._active_experiment_id, self._active_run_id)

    # Experiments and runs

    def create_experiment(self, name: str) -> str:
        """
        Creates a new experiment and returns its ID. If an experiment with
        the same name already exists, it returns the existing ID.
        """
        for experiment_id in os.listdir(self._mlruns_dir):
            meta_path = os.path.join(self._mlruns_dir, experiment_id, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    if json.load(f).get('name') == name:
                        return experiment_id

        experiment_id = str(len(os.listdir(self._mlruns_dir)))
        experiment_path = os.path.join(self._mlruns_dir, experiment_id)
        os.makedirs(experiment_path, exist_ok=True)

        meta = {"experiment_id": experiment_id, "name": name}
        with open(os.path.join(experiment_path, "meta.json"), 'w') as f:
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
            self.create_experiment("Default Experiment")

        self._active_experiment_id = experiment_id
        self._active_run_id = uuid.uuid4().hex[:8] # Short unique ID

        run_path = self._get_run_path()
        os.makedirs(os.path.join(run_path, "params"), exist_ok=True)
        os.makedirs(os.path.join(run_path, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(run_path, "artifacts"), exist_ok=True)
        
        meta = {"run_id": self._active_run_id, "experiment_id": self._active_experiment_id, "status": "RUNNING"}
        with open(os.path.join(run_path, "meta.json"), 'w') as f:
            json.dump(meta, f, indent=4)
        
        try:
            yield self._active_run_id
        finally:
            meta["status"] = "FINISHED"
            with open(os.path.join(run_path, "meta.json"), 'w') as f:
                json.dump(meta, f, indent=4)
            self._active_run_id = None
            self._active_experiment_id = None
            
    # Logging

    def log_param(self, key: str, value):
        """Logs a single parameter for the active run."""
        run_path = self._get_run_path()
        param_path = os.path.join(run_path, "params", f"{key}.json")
        with open(param_path, 'w') as f:
            json.dump({'value': value}, f, indent=4)

    def log_metric(self, key: str, value: float):
        """Logs a single metric for the active run."""
        run_path = self._get_run_path()
        metric_path = os.path.join(run_path, "metrics", f"{key}.json")
        with open(metric_path, 'w') as f:
            json.dump({'value': value}, f, indent=4)

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
                    with open(os.path.join(params_path, param_file), 'r') as f:
                        params[f"param_{key}"] = json.load(f)['value']

                # Load metrics
                metrics = {}
                metrics_path = os.path.join(run_path, "metrics")
                for metric_file in os.listdir(metrics_path):
                    key = os.path.splitext(metric_file)[0]
                    with open(os.path.join(metrics_path, metric_file), 'r') as f:
                        metrics[key] = json.load(f)['value']
                
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

        return pd.DataFrame(all_runs_data).set_index('run_id').sort_index()