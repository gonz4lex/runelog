import os
import json
import uuid
import joblib
import shutil
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
    def start_run(self, experiment_id: str = "0") -> Generator[str, None, None]:
        """Starts a new run within an experiment as a context manager.

        Upon entering the 'with' block, a new run is created and marked as
        'RUNNING'. When the block is exited, the run is marked as 'FINISHED'.

        Args:
            experiment_id (str, optional): The ID of the experiment to create
                the run in. Defaults to "0" (the default experiment).

        Yields:
            str: The unique ID of the newly created run.
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
            "timestamp": datetime.now().isoformat(),
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

        return {"params": params, "metrics": metrics, "artifacts": artifacts}

 
    def get_experiment_runs(
        self, experiment_id: str, sort_by: Optional[str] = None, ascending: bool = True
    ) -> List[Dict]:
        """Return a list of individual runs for the given experiment.

        Args:
            experiment_id (str): The ID of the experiment to query.
            sort_by (Optional[str], optional): Field to sort runs by (e.g., "timestamp").
                Defaults to "timestamp". Set to None to disable sorting.
            ascending (bool, optional): Sort order. Defaults to True.

        Returns:
            List[Dict]: A list of run dictionaries, each containing:
                - run_id (str): Folder name or unique ID of the run.
                - timestamp (str | None): ISO 8601 timestamp of the run, or None.
                - status (str | None): Run status if available in meta.json.
                - other metadata keys as present in meta.json.
        """
        runs = []
        exp_path = os.path.join(self._mlruns_dir, experiment_id)

        if not os.path.isdir(exp_path):
            return runs

        for item in os.listdir(exp_path):
            run_path = os.path.join(exp_path, item)
            if os.path.isdir(run_path):
                meta_path = os.path.join(run_path, "meta.json")
                if os.path.exists(meta_path):
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    run_id = item
                    timestamp = meta.get("timestamp")
                    if timestamp is None:
                        # Fallback: use file modification time
                        ts = os.path.getmtime(meta_path)
                        timestamp = datetime.fromtimestamp(ts).isoformat()
                    run_data = {
                        "run_id": run_id,
                        "timestamp": timestamp,
                        "status": meta.get("status"),
                        **{
                            k: v
                            for k, v in meta.items()
                            if k not in {"timestamp", "status"}
                        },
                    }
                    runs.append(run_data)

        if sort_by:
            runs.sort(key=lambda x: x.get(sort_by, ""), reverse=not ascending)

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
