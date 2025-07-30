import pytest
import yaml
import pandas as pd
from unittest.mock import MagicMock

from runelog.runner import _load_data, _run_single_experiment, run_sweep
from runelog import get_tracker


def test_load_data_from_sklearn():
    """Tests that data can be loaded from a scikit-learn dataset."""
    config = {"source": "sklearn.datasets.load_iris"}
    X, y = _load_data(config)
    assert X.shape == (150, 4)
    assert y.shape == (150,)


def test_load_data_from_csv(tmp_path):
    """Tests that data can be loaded from a local CSV file."""
    csv_path = tmp_path / "test_data.csv"
    dummy_df = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4], "target": [0, 1]})
    dummy_df.to_csv(csv_path, index=False)

    config = {"source": str(csv_path), "target_column": "target"}
    X, y = _load_data(config)

    assert X.shape == (2, 2)
    assert y.shape == (2,)
    assert "target" not in X.columns


@pytest.fixture
def mock_data():
    """Provides simple mock data for testing."""
    return pd.DataFrame({"feature": range(10)}), pd.Series(range(10))


def test_run_single_experiment_split(mock_data):
    """Tests the train_test_split validation strategy."""
    from sklearn.linear_model import LinearRegression

    X, y = mock_data
    model = LinearRegression()

    validation_config = {"strategy": "train_test_split", "params": {"test_size": 0.2}}
    evaluation_config = {"metrics": ["mean_squared_error"]}

    metrics = _run_single_experiment(model, X, y, validation_config, evaluation_config)
    assert "mean_squared_error" in metrics
    assert isinstance(metrics["mean_squared_error"], float)


def test_run_single_experiment_cv(mock_data):
    """Tests the cross_validation strategy."""
    from sklearn.linear_model import LinearRegression

    X, y = mock_data
    model = LinearRegression()

    validation_config = {"strategy": "cross_validation", "params": {"n_splits": 3}}
    evaluation_config = {"metrics": ["mean_squared_error"]}

    metrics = _run_single_experiment(model, X, y, validation_config, evaluation_config)
    assert "mean_squared_error" in metrics
    assert isinstance(metrics["mean_squared_error"], float)


def test_run_single_experiment_invalid_strategy(mock_data):
    """Tests that an invalid validation strategy raises a ValueError."""
    from sklearn.linear_model import LinearRegression

    X, y = mock_data
    model = LinearRegression()

    validation_config = {"strategy": "invalid_strategy"}
    evaluation_config = {"metrics": []}

    with pytest.raises(ValueError, match="Unknown validation strategy"):
        _run_single_experiment(model, X, y, validation_config, evaluation_config)


@pytest.fixture
def mock_tracker(tmp_path):
    """A fixture to create a fresh, isolated RuneLog instance for testing."""
    return get_tracker(path=str(tmp_path))


@pytest.fixture
def sweep_config_file(tmp_path):
    """A fixture to create a temporary YAML config file for a sweep."""
    config_data = {
        "experiment_name": "My Sweep Test",
        "dataset": {"source": "sklearn.datasets.load_iris"},
        "validation": {"strategy": "train_test_split", "params": {"test_size": 0.2}},
        "evaluation": {"metrics": ["accuracy_score"]},
        "runs": [
            {
                "id": "logistic_regression",
                "model_class": "sklearn.linear_model.LogisticRegression",
                "model_params": {"C": 1.0},
            }
        ],
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return str(config_path)


def test_run_sweep_full_workflow(monkeypatch, mock_tracker, sweep_config_file):
    """
    Tests the entire run_sweep workflow, ensuring it calls the tracker
    and logs data correctly.
    """
    monkeypatch.setattr("runelog.runner.get_tracker", lambda: mock_tracker)

    mock_handler = MagicMock()

    run_sweep(sweep_config_file, progress_handler=mock_handler)

    experiments = mock_tracker.list_experiments()
    assert len(experiments) == 1
    assert experiments[0]["name"] == "My Sweep Test"

    results = mock_tracker.load_results(experiments[0]["experiment_id"])
    assert len(results) == 1
    assert "param_C" in results.columns
    assert "accuracy_score" in results.columns

    assert mock_handler.call_count > 0
