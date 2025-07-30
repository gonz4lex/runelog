import yaml
import importlib

import pandas as pd

from runelog import get_tracker

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import make_scorer


def _silent_handler(message):
    """
    Simple "no-op" function to use as a default handler.
    """
    pass


def _load_data(config: dict) -> tuple:
    """Loads a dataset based on the configuration."""
    source = config["source"]

    # Handle loading from a local CSV file
    if source.endswith(".csv"):
        df = pd.read_csv(source)
        target_col = config["target_column"]
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return X, y

    # Handle loading from scikit-learn's built-in datasets
    else:
        module_path, func_name = source.rsplit(".", 1)
        module = importlib.import_module(module_path)
        loader_func = getattr(module, func_name)
        return loader_func(return_X_y=True)


def _run_single_experiment(
    model, X, y, validation_config: dict, evaluation_config: dict
) -> dict:
    """Runs training and evaluation based on the validation and evaluation configs."""
    metrics_to_compute = evaluation_config["metrics"]
    metric_params = evaluation_config.get("metric_params", {})

    # Cross-Validation
    if validation_config["strategy"] == "cross_validation":
        cv_params = validation_config.get("params", {})
        scoring = {}
        for metric_name in metrics_to_compute:
            metric_func = getattr(
                importlib.import_module("sklearn.metrics"), metric_name
            )
            params = metric_params.get(metric_name, {})
            scoring[metric_name] = make_scorer(metric_func, **params)

        # Run cross-validation with the custom scorers
        scores = cross_validate(
            model, X, y, cv=cv_params.get("n_splits", 5), scoring=scoring
        )

        # Average the scores from all folds
        avg_scores = {
            key.replace("test_", ""): value.mean()
            for key, value in scores.items()
            if key.startswith("test_")
        }
        return avg_scores

    # Split Strategy
    elif validation_config["strategy"] == "train_test_split":
        split_params = validation_config.get("params", {})
        X_train, X_test, y_train, y_test = train_test_split(X, y, **split_params)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        computed_metrics = {}
        for metric_name in metrics_to_compute:
            metric_func = getattr(
                importlib.import_module("sklearn.metrics"), metric_name
            )
            params = metric_params.get(metric_name, {})
            score = metric_func(y_test, y_pred, **params)
            computed_metrics[metric_name] = score
        return computed_metrics

    else:
        raise ValueError(
            f"Unknown validation strategy: {validation_config['strategy']}. Choose either 'test_train_split' or 'cross_validation'."
        )


def run_sweep(config_path: str, progress_handler=None):
    """
    Parses a config file and runs a series of ML experiments.

    Args:
        config_path (str): The local path to the sweep's YAML config file.
        progress_handler (callable, optional): A function to call with
            progress messages. Defaults to a silent handler.
    """
    # Use the provided handler or the silent default
    handler = progress_handler or _silent_handler
    tracker = get_tracker()

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    input_dataset = config["dataset"]

    X, y = _load_data(input_dataset)
    handler(
        f"Dataset '{input_dataset['source']}' loaded successfully. Shape: {X.shape}"
    )

    experiment_name = config.get("experiment_name", "default-sweep")
    exp_id = tracker.get_or_create_experiment(experiment_name)
    handler(f"Running sweep for experiment: '{experiment_name}'")

    # Apply each run config
    for run_config in config.get("runs", []):
        run_id_str = run_config.get("id", "unnamed-run")
        handler(f"\nrun in progress: '{run_id_str}'")

        with tracker.start_run(experiment_id=exp_id):
            run_params = {
                "run_config_id": run_id_str,
                "model_class": run_config.get("model_class"),
                "validation_strategy": config.get("validation", {}).get("strategy"),
            }
            model_params = run_config.get("model_params", {})
            all_params = {**run_params, **model_params}

            for key, value in all_params.items():
                tracker.log_param(key, value)

            handler(f"Logged Parameters: {all_params}")

            # Dynamically instantiate the model
            module_path, class_name = run_config["model_class"].rsplit(".", 1)
            ModelClass = getattr(importlib.import_module(module_path), class_name)
            model = ModelClass(**model_params)

            # Run and get metrics
            metrics = _run_single_experiment(
                model, X, y, config["validation"], config["evaluation"]
            )

            for key, value in metrics.items():
                tracker.log_metric(key, value)

            logged_metrics = {k: round(v, 4) for k, v in metrics.items()}
            handler(f"Logged Metrics: {logged_metrics}")
