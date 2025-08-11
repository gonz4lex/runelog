import os
import sys
import json
import uuid
import yaml
import joblib
import shutil
import hashlib
import platform
import subprocess
import warnings
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

from typing import Any, Dict, List, Optional, Generator

from . import exceptions


class RuneLog:
    """
    A lightweight tracker for ML experiments.

    This class handles the creation of experiments, management of runs, and logging
    of parameters, metrics, and artifacts to the local filesystem. It also
    provides a model registry for versioning and managing models.
    """

    def __init__(self, path="."):
        """Initializes the tracker and creates required directories.

        Args:
            path (str, optional): The root directory for storing all tracking
                data. Defaults to the current directory.
        """
        self.root_path = os.path.abspath(path)
        self._mlruns_dir = os.path.join(self.root_path, ".mlruns")
        self._registry_dir = os.path.join(self.root_path, ".registry")
        self._active_run_id = None
        self._active_experiment_id = None

        os.makedirs(self._mlruns_dir, exist_ok=True)
        os.makedirs(self._registry_dir, exist_ok=True)

    def _get_run_path(self):
        """Helper to get the absolute path of the current active run.

        Returns:
            str: The absolute path to the active run's directory.

        Raises:
            exceptions.NoActiveRun: If called outside of an active run context.
        """
        if not self._active_run_id:
            raise exceptions.NoActiveRun()
        return os.path.join(
            self._mlruns_dir, self._active_experiment_id, self._active_run_id
        )

    def _get_run_path_by_id(self, run_id: str) -> Optional[str]:
        """Finds the full path to a run directory from its ID."""
        for exp_id in os.listdir(self._mlruns_dir):
            exp_path = os.path.join(self._mlruns_dir, exp_id)
            if not os.path.isdir(exp_path):
                continue

            run_path = os.path.join(exp_path, run_id)
            if os.path.isdir(run_path):
                return run_path
        return None

    def _log_git_metadata(self):
        """
        Logs Git metadata to a dedicated source_control.json file in the
        active run's directory if the root directory is Git repo
        """
        try:
            commit_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
            )
            branch_name = (
                subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
                .strip()
                .decode()
            )
            is_dirty = bool(
                subprocess.check_output(["git", "status", "--porcelain"]).strip()
            )

            git_meta = {
                "commit_hash": commit_hash,
                "branch": branch_name,
                "is_dirty": is_dirty,
            }

            meta_path = os.path.join(self._get_run_path(), "source_control.json")
            with open(meta_path, "w") as f:
                json.dump(git_meta, f, indent=4)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fail silently if not in a git repo or git is not installed
            pass

    def _log_environment(self):
        """
        Logs environment info to a structured JSON file and creates a
        pip-installable requirements.txt artifact.
        """
        try:
            reqs_raw = (
                subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
                .decode()
                .strip()
            )

            # Installable requirements.txt artifact
            artifact_path = os.path.join(
                self._get_run_path(), "artifacts", "requirements.txt"
            )
            with open(artifact_path, "w") as f:
                f.write(reqs_raw)

            # Structured data for the UI
            packages = {
                pkg.split("==")[0]: pkg.split("==")[1]
                for pkg in reqs_raw.split("\n")
                if "==" in pkg
            }
            env_meta = {
                "python_version": sys.version,
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                },
                "packages": packages,
            }
            meta_path = os.path.join(self._get_run_path(), "environment.json")
            with open(meta_path, "w") as f:
                json.dump(env_meta, f, indent=4)

        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fails silently
            pass

    def _log_source_code(self):
        """
        Detects the executing script and logs it as an artifact.
        Warns the user if run in an interactive environment like a Jupyter notebook.
        """
        if "ipykernel" in sys.modules:
            import warnings

            warnings.warn(
                "Automatic code logging is not supported in interactive environments "
                "like Jupyter notebooks. Please log your notebook manually using "
                "tracker.log_artifact('path/to/your/notebook.ipynb')."
            )
            return

        script_path = os.path.abspath(sys.argv[0])
        if os.path.exists(script_path):
            self.log_artifact(script_path)

    def _hash_file(self, file_path: str) -> str:
        """Calculates the SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read and update hash in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    # Experiments and runs

    def get_or_create_experiment(self, name: str) -> str:
        """Gets an existing experiment by name or creates a new one.

        If an experiment with the given name already exists, its ID is returned.
        Otherwise, a new experiment is created.

        Args:
            name (str): The name of the experiment.

        Returns:
            str: The unique ID of the new or existing experiment.
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

    def list_experiments(self) -> List[Dict]:
        """Lists all available experiments.

        Returns:
            List[Dict]: A list of dictionaries, where each dictionary contains
                the metadata of an experiment (e.g., name and ID).
        """
        experiments = []
        for experiment_id in os.listdir(self._mlruns_dir):
            meta_path = os.path.join(self._mlruns_dir, experiment_id, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    experiments.append(meta)
        return experiments

    def delete_experiment(self, experiment_name_or_id: str) -> None:
        """Deletes an experiment and all of its associated runs and artifacts.
        This is a destructive operation and cannot be undone.

        Args:
            experiment_name_or_id (str): The name or ID of the experiment to delete.

        Raises:
            exceptions.ExperimentNotFound: If no experiment with the given
                name or ID is found.
        """
        _, experiment_path = self._resolve_experiment_id(experiment_name_or_id)

        shutil.rmtree(experiment_path)

    @contextmanager
    def start_run(
        self,
        experiment_name: str = None,
        experiment_id: str = "0",
        log_git_meta: bool = True,
        log_env: bool = False,
        log_code: bool = True,
    ) -> Generator[str, None, None]:
        """Starts a new run within an experiment as a context manager.

        This is the primary method for creating a new run. It handles run creation,
        status updates, and can optionally log metadata about the execution
        environment for reproducibility.

        Args:
            experiment_name (str, optional): The name of the experiment. If it
                doesn't exist, it will be created. This takes precedence over
                `experiment_id`. Defaults to None.
            experiment_id (str, optional): The ID of the experiment. If neither
                name nor ID is provided, it defaults to "0" (the "Default"
                experiment). Defaults to None.
            log_git_meta (bool, optional): If True, logs Git metadata
                (commit hash, branch, dirty status) to a `source_control.json` file.
                Defaults to True.
            log_env (bool, optional): If True, logs the Python environment
                (version, platform, packages) to `environment.json` and a
                `requirements.txt` artifact. Defaults to False.
            log_code (bool, optional): If True, logs the source code file as an artifact.
                Defaults to True.

        Yields:
            str: The unique ID of the newly created run.

        Example:
            >>> tracker = get_tracker()
            >>> with tracker.start_run(
            ...     experiment_name="example-experiment",
            ...     log_env=True
            ... ) as run_id:
            ...     tracker.log_metric("accuracy", 0.95)
        """
        if experiment_name:
            exp_id = self.get_or_create_experiment(experiment_name)
        elif experiment_id:
            exp_id = experiment_id
        else:
            exp_id = "0"

        if exp_id == "0" and not os.path.exists(
            os.path.join(self._mlruns_dir, "0", "meta.json")
        ):
            self.get_or_create_experiment("default-experiment")

        self._active_experiment_id = exp_id
        self._active_run_id = uuid.uuid4().hex[:8]  # Short unique ID

        run_path = self._get_run_path()
        os.makedirs(os.path.join(run_path, "params"), exist_ok=True)
        os.makedirs(os.path.join(run_path, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(run_path, "artifacts"), exist_ok=True)

        initial_meta = {
            "run_id": self._active_run_id,
            "experiment_id": self._active_experiment_id,
            "status": "RUNNING",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
        }
        run_meta_path = os.path.join(run_path, "meta.json")
        with open(run_meta_path, "w") as f:
            json.dump(initial_meta, f, indent=4)

        if log_git_meta:
            self._log_git_metadata()
        if log_env:
            self._log_environment()
        if log_code:
            self._log_source_code()

        try:
            yield self._active_run_id

        finally:
            with open(run_meta_path, "r") as f:
                meta = json.load(f)

            meta["status"] = "FINISHED"
            meta["end_time"] = datetime.now().isoformat()

            with open(os.path.join(run_path, "meta.json"), "w") as f:
                json.dump(meta, f, indent=4)

            # Active state clean-up
            self._active_run_id = None
            self._active_experiment_id = None

    def get_run_details(self, run_id: str) -> Optional[Dict]:
        """Loads all details for a specific run.

        Args:
            run_id (str): The unique ID of the run to retrieve.

        Returns:
            Optional[Dict]: A dictionary containing the run's 'params',
                'metrics', and 'artifacts', or None if the run is not found.
        """
        run_path = None
        for exp_id in os.listdir(self._mlruns_dir):
            path = os.path.join(self._mlruns_dir, exp_id, run_id)
            if os.path.isdir(path):
                run_path = path
                break
        if not run_path:
            return None  # Run not found

        meta = {}
        meta_path = os.path.join(run_path, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)

        params = {}
        params_path = os.path.join(run_path, "params")
        if os.path.exists(params_path):
            for param_file in os.listdir(params_path):
                key = os.path.splitext(param_file)[0]
                with open(os.path.join(params_path, param_file), "r") as f:
                    params[key] = json.load(f)["value"]

        metrics = {}
        metrics_path = os.path.join(run_path, "metrics")
        if os.path.exists(metrics_path):
            for metric_file in os.listdir(metrics_path):
                key = os.path.splitext(metric_file)[0]
                with open(os.path.join(metrics_path, metric_file), "r") as f:
                    metrics[key] = json.load(f)["value"]

        artifacts = []
        artifacts_path = os.path.join(run_path, "artifacts")
        if os.path.exists(artifacts_path):
            artifacts = os.listdir(artifacts_path)

        return {
            "meta": meta,
            "params": params,
            "metrics": metrics,
            "artifacts": artifacts,
        }

    def delete_run(self, run_id: str) -> None:
        """Deletes a run and all of its associated artifacts.

        This is a destructive operation and cannot be undone.

        Args:
            run_id (str): The ID of the run to delete.

        Raises:
            exceptions.RunNotFound: If no run with the given ID is found.
        """
        run_path = self._get_run_path_by_id(run_id)

        if not run_path:
            raise exceptions.RunNotFound(run_id)

        shutil.rmtree(run_path)

    def get_experiment_runs(
        self,
        experiment_id: str,
        sort_by: Optional[str] = "timestamp",
        ascending: bool = True,
    ) -> List[Dict]:
        """Return a list of individual runs for the given experiment.

        Args:
            experiment_id (str): The ID of the experiment to query.
            sort_by (Optional[str], optional): Field to sort runs by (e.g., "timestamp").
                Defaults to "timestamp". Set to None to disable sorting.
            ascending (bool, optional): Sort order. Defaults to True.

        Returns:
            List[Dict]: A list of run dictionaries.
        """
        runs = []
        exp_path = os.path.join(self._mlruns_dir, experiment_id)

        if not os.path.isdir(exp_path):
            return runs

        for run_id in os.listdir(exp_path):
            run_path = os.path.join(exp_path, run_id)
            meta_path = os.path.join(run_path, "meta.json")

            if os.path.isfile(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        run_data = json.load(f)

                    run_data["run_id"] = run_id

                    if run_data.get("timestamp") is None:
                        ts = os.path.getmtime(meta_path)
                        run_data["timestamp"] = datetime.fromtimestamp(ts).isoformat()

                    runs.append(run_data)
                except json.JSONDecodeError:
                    print(
                        f"Warning: Could not parse meta.json for run_id '{run_id}'. Skipping."
                    )
                    continue

        if sort_by:
            fallback_sort_value = (
                "0001-01-01T00:00:00.000000" if sort_by == "timestamp" else ""
            )
            runs.sort(
                key=lambda r: r.get(sort_by, fallback_sort_value), reverse=not ascending
            )

        return runs

    def get_experiment_summaries(
        self, sort_by: Optional[str] = None, ascending: bool = True
    ) -> List[Dict]:
        """Obtain summaries for all experiments, including run statistics:
        number of runs, the timestamp of the most recent run, and the creation time
        of the experiment.

        Args:
            sort_by (Optional[str], optional): Field to sort summaries by
                (e.g., "name", "num_runs", "last_run"). Defaults to None (no sorting).
            ascending (bool, optional): Sort order. Defaults to True.

        Returns:
            List[Dict]: A list of dictionaries, each representing a summarized view
            of an experiment. Each dictionary contains:
                - experiment_id (str): Unique identifier of the experiment.
                - name (str): Human-readable name of the experiment.
                - created_at (str | None): ISO 8601 timestamp of the experiment's creation,
                or None if not available.
                - num_runs (int): Total number of runs recorded for the experiment.
                - last_run (str | None): ISO 8601 timestamp of the most recent run,
                or None if no runs exist.
        """
        summaries = []
        for exp in self.list_experiments():
            exp_id = exp["experiment_id"]
            exp_name = exp.get("name", "—")
            exp_path = os.path.join(self._mlruns_dir, exp_id)

            run_timestamps = []
            for item in os.listdir(exp_path):
                run_path = os.path.join(exp_path, item)
                if os.path.isdir(run_path):
                    meta_path = os.path.join(run_path, "meta.json")
                    if os.path.exists(meta_path):
                        with open(meta_path, "r") as f:
                            meta = json.load(f)
                            ts = meta.get("timestamp")
                            if ts is None:
                                ts = os.path.getmtime(meta_path)
                                ts = datetime.fromtimestamp(ts).isoformat()
                            run_timestamps.append(ts)

            last_run = max(run_timestamps, default=None)
            created_at_path = os.path.join(exp_path, "meta.json")
            created_at = None
            if os.path.exists(created_at_path):
                created_at = datetime.fromtimestamp(
                    os.path.getmtime(created_at_path)
                ).isoformat()

            summaries.append(
                {
                    "experiment_id": exp_id,
                    "name": exp_name,
                    "created_at": created_at,
                    "num_runs": len(run_timestamps),
                    "last_run": last_run,
                }
            )
        if sort_by:
            summaries.sort(key=lambda x: x.get(sort_by) or "", reverse=not ascending)

        return summaries

    # Logging

    def log_param(self, key: str, value):
        """Logs a single parameter for the active run.

        Args:
            key (str): The name of the parameter.
            value (Any): The value of the parameter. Must be JSON-serializable.

        Raises:
            exceptions.NoActiveRun: If called outside of an active run context.
        """
        run_path = self._get_run_path()
        param_path = os.path.join(run_path, "params", f"{key}.json")
        with open(param_path, "w") as f:
            json.dump({"value": value}, f, indent=4)

    def log_metric(self, key: str, value: float):
        """Logs a single metric for the active run.

        Args:
            key (str): The name of the metric.
            value (float): The value of the metric.

        Raises:
            exceptions.NoActiveRun: If called outside of an active run context.
        """
        run_path = self._get_run_path()
        metric_path = os.path.join(run_path, "metrics", f"{key}.json")
        with open(metric_path, "w") as f:
            json.dump({"value": value}, f, indent=4)

    def log_artifact(self, local_path: str):
        """Logs a local file as an artifact of the active run.

        Args:
            local_path (str): The local path to the file to be logged as an
                artifact.

        Raises:
            exceptions.NoActiveRun: If called outside of an active run context.
            exceptions.ArtifactNotFound: If the file at `local_path` does not exist.
        """
        run_path = self._get_run_path()
        artifact_dir = os.path.join(run_path, "artifacts")
        if not os.path.exists(local_path):
            raise exceptions.ArtifactNotFound(local_path)
        shutil.copy(local_path, artifact_dir)

    def log_model(self, model: Any, name: str, compress: int = 3):
        """Logs a trained model as an artifact of the active run.

        Args:
            model (Any): The trained model object to be saved (e.g., a
                scikit-learn model).
            name (str): The filename for the saved model (e.g., "model.pkl").
            compress (int, optional): The level of compression for joblib from 0 to 9.
                Defaults to 3.

        Raises:
            exceptions.NoActiveRun: If called outside of an active run context.
        """
        run_path = self._get_run_path()
        model_path = os.path.join(run_path, "artifacts", name)
        joblib.dump(model, model_path, compress=compress)

    def log_dataset(self, data_path: str, name: str):
        """Logs the hash and metadata of a dataset file.

        This creates a verifiable "fingerprint" of the data used in a run,
        ensuring data lineage and reproducibility.

        Args:
            data_path (str): The local path to the dataset file.
            name (str): A descriptive name for the dataset.
        """
        if not os.path.exists(data_path):
            raise exceptions.ArtifactNotFound(data_path)

        data_meta = {
            "name": name,
            "filename": os.path.basename(data_path),
            "filesize_bytes": os.path.getsize(data_path),
            "hash_sha256": self._hash_file(data_path),
        }

        meta_path = os.path.join(self._get_run_path(), "data_meta.json")
        with open(meta_path, "w") as f:
            json.dump(data_meta, f, indent=4)

    def log_dvc_input(self, data_path: str, name: str):
        """Logs the version hash of a DVC-tracked file.

        This method finds the corresponding .dvc file for the given data path,
        parses it to find the MD5 hash of the data version, and logs this
        information to a dedicated `dvc_inputs.json` file in the run directory.
        This creates a verifiable link between the run and the exact version of the
        input data.

        Args:
            data_path (str): The path to the data file tracked by DVC (e.g.,
                "data/my_data.csv").
            name (str): A descriptive name for this data input.
        """
        dvc_file_path = f"{data_path}.dvc"
        if not os.path.exists(dvc_file_path):
            warnings.warn(
                f"DVC file not found at {dvc_file_path}. Skipping DVC logging."
            )
            return

        with open(dvc_file_path, "r") as f:
            dvc_meta = yaml.safe_load(f)

        # Unique fingerprint for the data version
        data_hash = dvc_meta.get("outs", [{}])[0].get("md5")

        if data_hash:
            dvc_info = {
                "name": name,
                "path": data_path,
                "dvc_file": dvc_file_path,
                "md5_hash": data_hash,
            }

            meta_path = os.path.join(self._get_run_path(), "dvc_inputs.json")
            with open(meta_path, "w") as f:
                json.dump(dvc_info, f, indent=4)

    def log_input_run(
        self, name: str, run_id: str, artifact_name: Optional[str] = None
    ):
        """Logs a dependency on another run, creating a lineage link.

        This is used to create verifiable links to upstream runs and their artifacts,
        for example, to specify that a model training run used the output from a
        specific feature generation run. The information is saved to a `lineage.json` file.

        Args:
            name (str): A logical name for the input.
            run_id (str): The unique ID of the run being used as an input.
            artifact_name (Optional[str], optional): The specific artifact from
            the input run to link and hash. Defaults to None.
        """
        lineage_path = os.path.join(self._get_run_path(), "lineage.json")

        # Read lineage data if exists
        lineage_data = []
        if os.path.exists(lineage_path):
            with open(lineage_path, "r") as f:
                content = f.read()
                if content:
                    lineage_data = json.loads(content)

        parent_run_details = self.get_run_details(run_id)
        if not parent_run_details:
            raise exceptions.RunNotFound(run_id)

        input_info = {
            "name": name,
            "run_id": run_id,
            "experiment_id": parent_run_details.get("meta", {}).get("experiment_id"),
            "artifact_name": None,
            "artifact_hash_sha256": None,
        }

        if artifact_name:
            input_info["artifact_name"] = artifact_name
            artifact_path = self.get_artifact_abspath(run_id, artifact_name)
            input_info["artifact_hash_sha256"] = self._hash_file(artifact_path)

        lineage_data.append(input_info)

        with open(lineage_path, "w") as f:
            json.dump(lineage_data, f, indent=4)

    # Reading

    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """
        Gets the metadata for a single experiment by its ID.

        Args:
            experiment_id (str): The ID of the experiment to retrieve.

        Returns:
            Optional[Dict]: A dictionary containing the experiment's metadata,
                or None if not found.
        """
        meta_path = os.path.join(self._mlruns_dir, experiment_id, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                return json.load(f)
        return None

    def get_run(self, run_id: str) -> Optional[Dict]:
        """Loads the parameters and metrics for a specific run.

        This method provides a summarized view of a run's data, primarily for
        use in creating tabular summaries like in `load_results`. It assumes
        that `run_id` is unique across all experiments.

        Note:
            For a more detailed dictionary that includes artifacts, see the
            `get_run_details()` method.

        Args:
            run_id (str): The unique ID of the run to retrieve.

        Returns:
            Optional[Dict]: A dictionary containing the `run_id` and all
                associated parameters and metrics, or None if the run is not
                found. Parameter keys are prefixed with 'param_'.
        """
        for experiment_id in os.listdir(self._mlruns_dir):
            run_path = os.path.join(self._mlruns_dir, experiment_id, run_id)
            if os.path.isdir(run_path):
                # Load metadata
                meta = {}
                meta_path = os.path.join(run_path, "meta.json")
                if os.path.exists(meta_path):
                    with open(meta_path, "r") as f:
                        try:
                            meta = json.load(f)
                        except json.JSONDecodeError:
                            warnings.warn(f"Warning: Corrupted meta.json for run '{run_id}'. Skipping metadata.")
                            meta = {}

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

                return {"run_id": run_id, **meta, **params, **metrics}
        return None

    def get_run_tags(self) -> Dict:
        """Retrieves the tags for the active run.

        Returns:
            Dict: A dictionary of the run's tags. Returns an empty dict if
                no tags are set.
        """
        run_path = self._get_run_path()
        meta_path = os.path.join(run_path, "meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return meta.get("tags", {})

    def set_run_tags(self, tags: Dict):
        """Sets the entire tag dictionary for the active run.

        This will overwrite any existing tags.

        Args:
            tags (Dict): The dictionary of tags to set for the run.
        """
        run_path = self._get_run_path()
        meta_path = os.path.join(run_path, "meta.json")
        
        with open(meta_path, "r+") as f:
            meta = json.load(f)
            meta["tags"] = tags # Overwrite the entire tags dictionary
            
            f.seek(0)
            json.dump(meta, f, indent=4)
            f.truncate()

    def _resolve_experiment_id(self, name_or_id: str) -> str:
        """
        Finds an experiment's ID from either its name or its ID.

        Args:
            name_or_id (str): The name or ID of the experiment.

        Returns:
            str: The canonical experiment ID.

        Raises:
            exceptions.ExperimentNotFound: If no matching experiment is found.
        """
        path = os.path.join(self._mlruns_dir, name_or_id)
        if os.path.isdir(path):
            return name_or_id, path

        for experiment in self.list_experiments():
            if experiment.get("name") == name_or_id:
                exp_id = experiment["experiment_id"]
                path = os.path.join(self._mlruns_dir, exp_id)
                return exp_id, path

        raise exceptions.ExperimentNotFound(name_or_id)

    def load_results(
        self,
        experiment_name_or_id: str,
        sort_by: Optional[str] = None,
        ascending: bool = True,
    ) -> pd.DataFrame:
        """Loads all run data from an experiment into a pandas DataFrame.

        Args:
            experiment_name_or_id (str): The ID of the experiment to load.
            sort_by (Optional[str], optional): Column to sort the DataFrame by.
                Defaults to None (sort by run_id index).
            ascending (bool, optional): Sort order. Defaults to True.

        Returns:
            pd.DataFrame: A DataFrame containing the parameters and metrics
                for each run in the experiment, indexed by `run_id`. Returns
                an empty DataFrame if the experiment has no runs.

        Raises:
            exceptions.ExperimentNotFound: If no experiment with the given ID
                is found.
        """
        experiment_id, experiment_path = self._resolve_experiment_id(
            experiment_name_or_id
        )
        if not os.path.exists(experiment_path):
            raise exceptions.ExperimentNotFound(experiment_id)

        all_runs_data = []
        for run_id in os.listdir(experiment_path):
            # Skip metadata file, only process run directories
            if os.path.isdir(os.path.join(experiment_path, run_id)):
                run_data = self.get_run(run_id)
                if run_data:
                    all_runs_data.append(run_data)

        if not all_runs_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_runs_data).set_index("run_id")

        if sort_by and sort_by in df.columns:
            df.sort_values(by=sort_by, ascending=ascending, inplace=True)
        else:
            df.sort_index(inplace=True)

        return df

    # Model Registry

    def register_model(
        self, run_id: str, artifact_name: str, model_name: str, tags: dict = None
    ) -> str:
        """Registers a model from a run's artifacts to the model registry.

        Args:
            run_id (str): The ID of the run where the model artifact is stored.
            artifact_name (str): The filename of the model artifact (e.g., "model.pkl").
            model_name (str): The name to register the model under. This can be
                a new or existing model name.
            tags (Optional[Dict], optional): A dictionary of tags to add to the
                new model version. Defaults to None.

        Returns:
            str: The new version number of the registered model as a string.

        Raises:
            exceptions.RunNotFound: If no run with the given ID is found.
            exceptions.ArtifactNotFound: If the specified artifact is not found
                in the run.
        """
        # Find the model artifact
        run_path = None
        for exp_id in os.listdir(self._mlruns_dir):
            path = os.path.join(self._mlruns_dir, exp_id, run_id)
            if os.path.isdir(path):
                run_path = path
                break

        if not run_path:
            raise exceptions.RunNotFound(run_id)

        source_artifact_path = os.path.join(run_path, "artifacts", artifact_name)
        if not os.path.exists(source_artifact_path):
            raise exceptions.ArtifactNotFound(
                artifact_path=artifact_name, run_id=run_id
            )

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
            "tags": tags or {},
        }
        with open(os.path.join(version_path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=4)

        print(
            f"Successfully registered model '{model_name}' with version {new_version}."
        )
        return new_version

    def load_registered_model(self, model_name: str, version: str = "latest") -> Any:
        """Loads a model from the model registry.

        Args:
            model_name (str): The name of the registered model.
            version (str, optional): The version to load. Can be a specific
                version number or "latest". Defaults to "latest".

        Returns:
            Any: The loaded model object.

        Raises:
            exceptions.ModelNotFound: If no model with the given name is found.
            exceptions.NoVersionsFound: If the model exists but has no versions.
            exceptions.ModelVersionNotFound: If the specified version is not
                found for the model.
        """
        model_path = os.path.join(self._registry_dir, model_name)
        if not os.path.exists(model_path):
            raise exceptions.ModelNotFound(model_name)

        if version == "latest":
            versions = [d for d in os.listdir(model_path) if d.isdigit()]
            if not versions:
                raise exceptions.NoVersionsFound(model_name)
            latest_version = str(max([int(v) for v in versions]))
            version_to_load = latest_version
        else:
            version_to_load = version

        final_model_path = os.path.join(model_path, version_to_load, "model.joblib")
        if not os.path.exists(final_model_path):
            raise exceptions.ModelVersionNotFound(
                model_name=model_name, version=version_to_load
            )

        return joblib.load(final_model_path)

    def add_model_tags(self, model_name: str, version: str, tags: dict) -> Dict:
        """Retrieves the tags for a specific registered model version.

        Args:
            model_name (str): The name of the registered model.
            version (str): The version from which to retrieve tags.

        Returns:
            Dict: A dictionary of the model version's tags.

        Raises:
            exceptions.ModelVersionNotFound: If the model or version is not found.
        """
        version_path = os.path.join(self._registry_dir, model_name, version)
        meta_path = os.path.join(version_path, "meta.json")

        if not os.path.exists(meta_path):
            raise exceptions.ModelVersionNotFound(
                model_name=model_name, version=version
            )

        with open(meta_path, "r+") as f:
            meta = json.load(f)
            if "tags" not in meta:
                meta["tags"] = {}
            meta["tags"].update(tags)  # Add or overwrite tags

            f.seek(0)  # Rewind to the beginning of the file
            json.dump(meta, f, indent=4)
            f.truncate()  # Remove any trailing content if the new file is shorter

    def get_model_tags(self, model_name: str, version: str) -> dict:
        """Retrieves the tags for a specific registered model version.

        Args:
            model_name (str): The name of the registered model.
            version (str): The version from which to retrieve tags.

        Returns:
            Dict: A dictionary of the model version's tags.

        Raises:
            exceptions.ModelVersionNotFound: If the model or version is not found.
        """
        version_path = os.path.join(self._registry_dir, model_name, version)
        meta_path = os.path.join(version_path, "meta.json")

        if not os.path.exists(meta_path):
            raise exceptions.ModelVersionNotFound(
                model_name=model_name, version=version
            )

        with open(meta_path, "r") as f:
            meta = json.load(f)
            return meta.get("tags", {})

    def list_registered_models(self, ascending: bool = True) -> List[str]:
        """Lists the names of all models in the registry.

        Returns:
            List[str]: A list of names of all registered models.
        """
        if not os.path.exists(self._registry_dir):
            return []

        # Returns a list of model names (directory names)
        model_names = [
            d
            for d in os.listdir(self._registry_dir)
            if os.path.isdir(os.path.join(self._registry_dir, d))
        ]
        model_names.sort(reverse=not ascending)

        return model_names

    def get_model_versions(
        self,
        model_name: str,
        sort_by: Optional[str] = "version",
        ascending: bool = False,
    ) -> List[Dict]:
        """Gets all versions and their metadata for a registered model.

        The versions are returned sorted from newest to oldest.

        Args:
            model_name (str): The name of the model to retrieve versions for.
            sort_by (Optional[str], optional): Field to sort by (e.g., "version",
                "registration_timestamp"). Sorted by 'version' by default.
            ascending (bool, optional): Sort order. Defaults to True (newest first).

        Returns:
            List[Dict]: A list of metadata dictionaries, where each dictionary
                represents a single version of the model. Returns an empty
                list if the model is not found.
        """
        model_path = os.path.join(self._registry_dir, model_name)
        if not os.path.exists(model_path):
            return []

        versions_data = []
        versions = [d for d in os.listdir(model_path) if d.isdigit()]

        for version in versions:
            meta_path = os.path.join(model_path, version, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    versions_data.append(json.load(f))

        if sort_by:
            if sort_by == "version":
                versions_data.sort(
                    key=lambda x: int(x.get("version", 0)), reverse=not ascending
                )
            else:
                versions_data.sort(
                    key=lambda x: str(x.get(sort_by, "")), reverse=not ascending
                )

        return versions_data

    def get_artifact_abspath(self, run_id: str, artifact_name: str) -> str:
        """
        Gets the absolute path of a specific artifact from a given run.

        Args:
            run_id (str): The ID of the run containing the artifact.
            artifact_name (str): The filename of the artifact.

        Returns:
            str: The absolute, local path to the artifact file.

        Raises:
            exceptions.RunNotFound: If the run ID does not exist.
            exceptions.ArtifactNotFound: If the artifact name does not exist in the run.
        """
        run_details = self.get_run_details(run_id)
        if not run_details:
            raise exceptions.RunNotFound(run_id)

        # Reconstruct the full path
        exp_id = None
        for eid in os.listdir(self._mlruns_dir):
            if os.path.isdir(os.path.join(self._mlruns_dir, eid, run_id)):
                exp_id = eid
                break

        artifact_path = os.path.join(
            self._mlruns_dir, exp_id, run_id, "artifacts", artifact_name
        )

        if artifact_name not in run_details["artifacts"] or not os.path.exists(
            artifact_path
        ):
            raise exceptions.ArtifactNotFound(artifact_name, run_id)

        return artifact_path

    def download_artifact(
        self, run_id: str, artifact_name: str, destination_path: str = "."
    ) -> str:
        """Downloads an artifact from a specific run to a local path.

        Args:
            run_id (str): The ID of the run containing the artifact.
            artifact_name (str): The filename of the artifact to download.
            destination_path (str, optional): The local directory to save the
                artifact in. Defaults to the current directory.

        Returns:
            str: The absolute path to the downloaded artifact.
        """
        source_path = self.get_artifact_abspath(run_id, artifact_name)

        os.makedirs(destination_path, exist_ok=True)
        final_destination = os.path.join(destination_path, artifact_name)

        shutil.copy(source_path, final_destination)

        return os.path.abspath(final_destination)
