import pytest

import os
import sys
import json
import subprocess

import pandas as pd

from runelog.runelog import RuneLog
from runelog import exceptions


class MockModel:
    def __init__(self, val=1):
        self.val = val


@pytest.fixture
def tracker(tmp_path):
    """
    A pytest fixture that creates a RuneLog instance in a temporary directory
    for each test function, ensuring tests are isolated.
    """
    # The tmp_path fixture provides a unique temporary directory managed by pytest
    return RuneLog(path=str(tmp_path))


def test_initialization(tracker):
    """Tests that the RuneLog class initializes its directories correctly."""
    assert os.path.exists(tracker._mlruns_dir)
    assert os.path.exists(tracker._registry_dir)


def test_get_or_create_experiment(tracker):
    """Tests experiment creation and retrieval."""
    exp_name = "test-get-or-create"

    # First call should create the experiment
    exp_id_1 = tracker.get_or_create_experiment(exp_name)
    assert exp_id_1 == "0"

    # Second call with the same name should return the same ID
    exp_id_2 = tracker.get_or_create_experiment(exp_name)
    assert exp_id_2 == "0"

    # Check metadata was created correctly
    meta_path = os.path.join(tracker._mlruns_dir, exp_id_1, "meta.json")
    assert os.path.exists(meta_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
        assert meta["name"] == exp_name


def test_start_run_context(tracker):
    """Tests the start_run context manager."""
    exp_id = tracker.get_or_create_experiment("test-context")

    run_id = None
    with tracker.start_run(experiment_id=exp_id) as active_run_id:
        run_id = active_run_id
        assert tracker._active_run_id is not None

        # Check that the status is "RUNNING" inside the context
        run_meta_path = os.path.join(tracker._get_run_path(), "meta.json")
        with open(run_meta_path, "r") as f:
            meta = json.load(f)
            assert meta["status"] == "RUNNING"

    # Check that the context was cleaned up
    assert tracker._active_run_id is None

    # Check that the status is "FINISHED" after the context
    run_path = os.path.join(tracker._mlruns_dir, exp_id, run_id)
    final_meta_path = os.path.join(run_path, "meta.json")
    with open(final_meta_path, "r") as f:
        meta = json.load(f)
        assert meta["status"] == "FINISHED"


def test_logging_functions(tracker):
    """Tests that logging methods create the correct files."""
    exp_id = tracker.get_or_create_experiment("test-logging")

    with tracker.start_run(experiment_id=exp_id) as run_id:
        # Test param logging
        tracker.log_param("learning_rate", 0.01)
        param_path = os.path.join(
            tracker._get_run_path(), "params", "learning_rate.json"
        )
        assert os.path.exists(param_path)
        with open(param_path, "r") as f:
            assert json.load(f)["value"] == 0.01

        # Test metric logging
        tracker.log_metric("accuracy", 0.95)
        metric_path = os.path.join(tracker._get_run_path(), "metrics", "accuracy.json")
        assert os.path.exists(metric_path)
        with open(metric_path, "r") as f:
            assert json.load(f)["value"] == 0.95


def test_log_artifact_and_model(tracker):
    """Tests artifact and model logging."""
    exp_id = tracker.get_or_create_experiment("test-artifacts")

    # Create a dummy artifact file
    dummy_artifact_path = os.path.join(tracker.root_path, "dummy_artifact.txt")
    with open(dummy_artifact_path, "w") as f:
        f.write("hello world")

    with tracker.start_run(experiment_id=exp_id) as run_id:
        # Test artifact logging
        tracker.log_artifact(dummy_artifact_path)
        logged_artifact_path = os.path.join(
            tracker._get_run_path(), "artifacts", "dummy_artifact.txt"
        )
        assert os.path.exists(logged_artifact_path)

        # Test model logging
        model = MockModel()
        tracker.log_model(model, "test_model.pkl")
        logged_model_path = os.path.join(
            tracker._get_run_path(), "artifacts", "test_model.pkl"
        )
        assert os.path.exists(logged_model_path)


def test_model_registry_workflow(tracker):
    """Tests the full model registry workflow."""
    exp_id = tracker.get_or_create_experiment("test-registry-workflow")
    model = MockModel(val=0)
    model_name = "test-registry-model"

    with tracker.start_run(experiment_id=exp_id) as run_id:
        tracker.log_model(model, "model.pkl")

    # Register the model
    version_1 = tracker.register_model(run_id, "model.pkl", model_name)
    assert version_1 == "1"

    # Register a new version
    version_2 = tracker.register_model(run_id, "model.pkl", model_name)
    assert version_2 == "2"

    # Load the latest version
    loaded_model = tracker.load_registered_model(model_name, version="latest")
    assert loaded_model.val == 0

    # Load a specific version
    loaded_model_v1 = tracker.load_registered_model(model_name, version="1")
    assert loaded_model_v1.val == 0

    # Add and get tags
    tracker.add_model_tags(model_name, "2", {"status": "production"})
    tags = tracker.get_model_tags(model_name, "2")
    assert tags["status"] == "production"


def test_logging_outside_active_run_raises_error(tracker):
    """
    Tests that calling a logging function outside of a 'start_run'
    context raises the appropriate exception.
    """
    with pytest.raises(exceptions.NoActiveRun):
        tracker.log_param("should_fail", 123)

    with pytest.raises(exceptions.NoActiveRun):
        tracker.log_metric("should_also_fail", 0.5)


def test_load_results_on_empty_experiment(tracker):
    """
    Tests that loading results from an experiment with no runs
    returns an empty DataFrame.
    """
    exp_id = tracker.get_or_create_experiment("test-empty")
    results = tracker.load_results(exp_id)

    assert isinstance(results, pd.DataFrame)
    assert results.empty


def test_log_nonexistent_artifact_raises_error(tracker):
    """
    Tests that trying to log an artifact from a path that does not
    exist raises a FileNotFoundError.
    """
    exp_id = tracker.get_or_create_experiment("test-artifact-edge-cases")

    with tracker.start_run(experiment_id=exp_id):
        with pytest.raises(exceptions.ArtifactNotFound):
            tracker.log_artifact("path/to/nonexistent/file.txt")


def test_registry_loading_edge_cases(tracker):
    """
    Tests edge cases for loading models from the registry, such as when
    a model or version is not found.
    """
    # Test loading a model that has not been registered
    with pytest.raises(exceptions.ModelNotFound):
        tracker.load_registered_model("nonexistent-model")

    # Register a model but then try to load a version that doesn't exist
    exp_id = tracker.get_or_create_experiment("registry-edge-case")
    model_name = "my-test-model"
    model = MockModel()

    with tracker.start_run(experiment_id=exp_id) as run_id:
        tracker.log_model(model, "model.pkl")

    tracker.register_model(run_id, "model.pkl", model_name)  # Registers version "1"

    with pytest.raises(exceptions.ModelVersionNotFound):
        tracker.load_registered_model(model_name, version="99")


def test_get_run_details_for_nonexistent_run(tracker):
    """
    Tests that get_run_details returns None for a run ID that doesn't exist.
    """
    # This assumes you might later change this to raise a RunNotFound exception
    assert tracker.get_run_details("nonexistent_run_id") is None


def test_delete_experiment_success(tracker):
    """Tests that a specific experiment can be successfully deleted."""
    exp_name = "experiment-to-delete"
    exp_id = tracker.get_or_create_experiment(exp_name)
    exp_path = os.path.join(tracker._mlruns_dir, exp_id)
    assert os.path.exists(exp_path)

    tracker.delete_experiment(exp_name)

    assert not os.path.exists(exp_path)


def test_delete_nonexistent_experiment_raises_error(tracker):
    """Tests that deleting a non-existent experiment raises an error."""
    with pytest.raises(exceptions.ExperimentNotFound):
        tracker.delete_experiment("nonexistent-experiment")


def test_delete_run_success(tracker):
    """Tests that a specific run can be successfully deleted."""
    experiment_id = tracker.get_or_create_experiment("test-delete-run")
    with tracker.start_run(experiment_id=experiment_id) as run_id:
        pass

    run_path = tracker._get_run_path_by_id(run_id)
    assert os.path.exists(run_path)

    tracker.delete_run(run_id)

    assert not os.path.exists(run_path)


def test_delete_nonexistent_run_raises_error(tracker):
    """Tests that deleting a non-existent run raises an error."""
    with pytest.raises(exceptions.RunNotFound):
        tracker.delete_run("nonexistent-run-id")


def test_log_git_metadata_success(tracker, monkeypatch):
    """
    Tests that Git metadata is correctly logged for a clean repository.
    """

    def mock_subprocess(args):
        if "rev-parse" in args and "--abbrev-ref" in args:
            return b"mock_branch\n"
        if "rev-parse" in args and "HEAD" in args:
            return b"mock_commit_hash\n"
        if "status" in args:
            return b""  # empty bytes means a clean repo
        return b""

    monkeypatch.setattr(subprocess, "check_output", mock_subprocess)

    with tracker.start_run(experiment_name="git-test"):
        tracker._log_git_metadata()

        meta_path = os.path.join(tracker._get_run_path(), "source_control.json")
        assert os.path.exists(meta_path)
        with open(meta_path, "r") as f:
            data = json.load(f)
            assert data["commit_hash"] == "mock_commit_hash"
            assert data["branch"] == "mock_branch"
            assert data["is_dirty"] is False


def test_log_git_metadata_dirty_repo(tracker, monkeypatch):
    """
    Tests that the 'is_dirty' flag is set correctly for a dirty repository.
    """

    def mock_subprocess(args):
        if "status" in args:
            return b"M modified_file.py\n"  # non-empty bytes means a dirty repo
        return b"mock_data\n"

    monkeypatch.setattr(subprocess, "check_output", mock_subprocess)

    with tracker.start_run(experiment_name="git-dirty-test"):
        tracker._log_git_metadata()
        meta_path = os.path.join(tracker._get_run_path(), "source_control.json")
        with open(meta_path, "r") as f:
            assert json.load(f)["is_dirty"] is True


def test_log_environment_success(tracker, monkeypatch):
    """
    Tests that environment and package info is correctly logged.
    """
    mock_pip_output = b"pandas==1.5.0\nscikit-learn==1.3.0\n"

    def mock_subprocess(args):
        if "pip" in args and "freeze" in args:
            return mock_pip_output
        return b""

    monkeypatch.setattr(subprocess, "check_output", mock_subprocess)

    with tracker.start_run(experiment_name="env-test"):
        tracker._log_environment()

        json_path = os.path.join(tracker._get_run_path(), "environment.json")
        assert os.path.exists(json_path)
        with open(json_path, "r") as f:
            data = json.load(f)
            assert "python_version" in data
            assert data["packages"]["pandas"] == "1.5.0"

        artifact_path = os.path.join(
            tracker._get_run_path(), "artifacts", "requirements.txt"
        )
        assert os.path.exists(artifact_path)
        with open(artifact_path, "r") as f:
            assert f.read() == mock_pip_output.decode().strip()


def test_log_environment_failure(tracker, monkeypatch):
    """
    Tests that the method fails silently if 'pip' command fails.
    """

    def mock_subprocess_fail(args):
        raise subprocess.CalledProcessError(1, "pip freeze")

    monkeypatch.setattr(subprocess, "check_output", mock_subprocess_fail)

    with tracker.start_run(experiment_name="env-fail-test"):
        tracker._log_environment()  # should NOT raise an exception

        json_path = os.path.join(tracker._get_run_path(), "environment.json")
        artifact_path = os.path.join(
            tracker._get_run_path(), "artifacts", "requirements.txt"
        )
        assert not os.path.exists(json_path)
        assert not os.path.exists(artifact_path)


def test_log_source_code_script(tracker, monkeypatch, tmp_path):
    """Tests that the executing script is automatically logged as an artifact."""
    dummy_script_path = tmp_path / "my_test_script.py"
    dummy_script_path.write_text("print('This is a test script.')")
    monkeypatch.setattr(sys, "argv", [str(dummy_script_path)])

    with tracker.start_run(
        experiment_name="code-log-test", log_code=True
    ) as run_id:
        pass

    run_details = tracker.get_run_details(run_id)
    assert "my_test_script.py" in run_details["artifacts"]
