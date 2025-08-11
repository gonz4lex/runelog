import pytest

import os
import sys
import json
import yaml
import hashlib
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

    with tracker.start_run(experiment_name="code-log-test", log_code=True) as run_id:
        pass

    run_details = tracker.get_run_details(run_id)
    assert "my_test_script.py" in run_details["artifacts"]


def test_log_dataset_success(tracker, tmp_path):
    """
    Tests that dataset metadata is correctly logged to a data.json file.
    """
    dummy_data_path = tmp_path / "training_data.csv"
    file_content = "feature1,feature2\n1,2\n3,4"
    dummy_data_path.write_text(file_content)

    expected_hash = hashlib.sha256(file_content.encode()).hexdigest()
    expected_size = os.path.getsize(dummy_data_path)

    with tracker.start_run(experiment_name="dataset-test") as run_id:
        tracker.log_dataset(str(dummy_data_path), name="primary_dataset")

        data_json_path = os.path.join(tracker._get_run_path(), "data_meta.json")
        assert os.path.exists(data_json_path)

        with open(data_json_path, "r") as f:
            data = json.load(f)
            assert data["name"] == "primary_dataset"
            assert data["filename"] == "training_data.csv"
            assert data["filesize_bytes"] == expected_size
            assert data["hash_sha256"] == expected_hash


def test_log_dataset_file_not_found(tracker):
    """
    Tests that log_dataset raises ArtifactNotFound for a non-existent file.
    """
    with tracker.start_run(experiment_name="dataset-fail-test"):
        with pytest.raises(exceptions.ArtifactNotFound):
            tracker.log_dataset("path/to/nonexistent/file.csv", name="bad_data")


def test_log_dataset_hash_changes_on_modification(tracker, tmp_path):
    """
    Tests that the logged data hash changes if the source file is modified.
    """
    dummy_data_path = tmp_path / "mutable_data.csv"
    dummy_data_path.write_text("initial content")

    with tracker.start_run(experiment_name="hash-change-test") as run_id_1:
        tracker.log_dataset(str(dummy_data_path), name="test_data")

    dummy_data_path.write_text("modified content")

    with tracker.start_run(experiment_name="hash-change-test") as run_id_2:
        tracker.log_dataset(str(dummy_data_path), name="test_data")

    run_1_path = tracker._get_run_path_by_id(run_id_1)
    with open(os.path.join(run_1_path, "data_meta.json"), "r") as f:
        data_1 = json.load(f)
    hash_1 = data_1["hash_sha256"]

    run_2_path = tracker._get_run_path_by_id(run_id_2)
    with open(os.path.join(run_2_path, "data_meta.json"), "r") as f:
        data_2 = json.load(f)
    hash_2 = data_2["hash_sha256"]

    assert hash_1 is not None
    assert hash_2 is not None
    assert hash_1 != hash_2


def test_log_dvc_input_success(tracker, tmp_path):
    """Tests that DVC metadata is correctly logged from a .dvc file."""
    data_path = tmp_path / "data.csv"
    dvc_file = tmp_path / "data.csv.dvc"
    mock_hash = "abc123def456"
    dvc_content = {"outs": [{"md5": mock_hash}]}
    with open(dvc_file, "w") as f:
        yaml.dump(dvc_content, f)

    with tracker.start_run(experiment_name="dvc-test") as run_id:
        tracker.log_dvc_input(str(data_path), name="training_data")

    dvc_json_path = os.path.join(tracker._get_run_path_by_id(run_id), "dvc_inputs.json")
    assert os.path.exists(dvc_json_path)
    with open(dvc_json_path, "r") as f:
        data = json.load(f)
        assert data["name"] == "training_data"
        assert data["md5_hash"] == mock_hash


def test_log_dvc_input_file_not_found(tracker):
    """Tests that the method warns but does not fail if no .dvc file is found."""
    with tracker.start_run(experiment_name="dvc-fail-test"):
        tracker.log_dvc_input("path/to/nonexistent_data.csv", name="bad_data")
        dvc_json_path = os.path.join(tracker._get_run_path(), "dvc_inputs.json")
        assert not os.path.exists(dvc_json_path)


def test_log_input_run(tracker):
    """Tests that run lineage dependencies are correctly logged."""
    with tracker.start_run(experiment_name="lineage-test") as parent_run_id:
        pass

    with tracker.start_run(experiment_name="lineage-test") as child_run_id:
        tracker.log_input_run(name="feature_source", run_id=parent_run_id)

    lineage_path = os.path.join(
        tracker._get_run_path_by_id(child_run_id), "lineage.json"
    )
    assert os.path.exists(lineage_path)

    with open(lineage_path, "r") as f:
        data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 1

        assert data[0]["name"] == "feature_source"
        assert data[0]["run_id"] == parent_run_id


def test_log_input_run_multiple_distinct_inputs(tracker):
    """
    Tests that logging multiple, different inputs results in a lineage file
    with all inputs present.
    """
    with tracker.start_run(experiment_name="multi-input-test") as parent_run_1:
        pass
    with tracker.start_run(experiment_name="multi-input-test") as parent_run_2:
        pass

    with tracker.start_run(experiment_name="multi-input-test") as child_run_id:
        tracker.log_input_run(name="feature_source", run_id=parent_run_1)
        tracker.log_input_run(name="config_source", run_id=parent_run_2)

    lineage_path = os.path.join(
        tracker._get_run_path_by_id(child_run_id), "lineage.json"
    )
    assert os.path.exists(lineage_path)

    with open(lineage_path, "r") as f:
        data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 2

        assert data[0]["name"] == "feature_source"
        assert data[0]["run_id"] == parent_run_1

        assert data[1]["name"] == "config_source"
        assert data[1]["run_id"] == parent_run_2


def test_log_input_run_does_not_overwrite(tracker):
    """
    Tests that calling log_input_run multiple times appends entries
    to the lineage list, rather than overwriting.
    """
    with tracker.start_run(experiment_name="test") as parent_1:
        pass
    with tracker.start_run(experiment_name="test") as parent_2:
        pass

    with tracker.start_run(experiment_name="test") as child_run_id:
        tracker.log_input_run(name="dataset", run_id=parent_1)
        tracker.log_input_run(name="dataset", run_id=parent_2)  # Append, not overwrite

    lineage_path = os.path.join(
        tracker._get_run_path_by_id(child_run_id), "lineage.json"
    )
    with open(lineage_path, "r") as f:
        data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 2  # Should now have two entries
        assert data[0]["run_id"] == parent_1
        assert data[1]["run_id"] == parent_2


def test_log_input_run_appends_on_duplicate_name(tracker):
    """
    Tests that calling log_input_run multiple times with the same name
    correctly appends new entries to the list.
    """
    with tracker.start_run(experiment_name="append-test") as old_parent_id:
        pass
    with tracker.start_run(experiment_name="append-test") as new_parent_id:
        pass

    with tracker.start_run(experiment_name="append-test") as child_run_id:
        tracker.log_input_run(name="dataset", run_id=old_parent_id)
        tracker.log_input_run(name="dataset", run_id=new_parent_id)  # Appends

    lineage_path = os.path.join(
        tracker._get_run_path_by_id(child_run_id), "lineage.json"
    )
    with open(lineage_path, "r") as f:
        data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 2

        assert data[0]["run_id"] == old_parent_id
        assert data[1]["run_id"] == new_parent_id


def test_log_input_run_with_artifact_hashing(tracker, tmp_path):
    """
    Tests that logging an input run with a specific artifact also logs
    the artifact's name, hash, and parent experiment ID.
    """
    parent_exp_id = tracker.get_or_create_experiment("feature-engineering")
    dummy_artifact_path = tmp_path / "features.csv"
    file_content = "feature,target\n1,0"
    dummy_artifact_path.write_text(file_content)
    expected_hash = tracker._hash_file(dummy_artifact_path)

    with tracker.start_run(experiment_id=parent_exp_id) as parent_run_id:
        tracker.log_artifact(str(dummy_artifact_path))

    with tracker.start_run(experiment_name="model-training") as child_run_id:
        tracker.log_input_run(
            name="training_features", run_id=parent_run_id, artifact_name="features.csv"
        )

    lineage_path = os.path.join(
        tracker._get_run_path_by_id(child_run_id), "lineage.json"
    )
    with open(lineage_path, "r") as f:
        data = json.load(f)
        assert isinstance(data, list) and len(data) == 1

        input_data = data[0]
        assert input_data["name"] == "training_features"
        assert input_data["run_id"] == parent_run_id
        assert input_data["experiment_id"] == parent_exp_id
        assert input_data["artifact_name"] == "features.csv"
        assert input_data["artifact_hash_sha256"] == expected_hash


def test_download_artifact(tracker, tmp_path):
    """Tests that an artifact can be successfully downloaded from a run."""
    dummy_artifact_path = tmp_path / "test.txt"
    dummy_content = "hello world"
    dummy_artifact_path.write_text(dummy_content)

    with tracker.start_run(experiment_name="download-test") as run_id:
        tracker.log_artifact(str(dummy_artifact_path))

    download_dir = tmp_path / "downloads"
    downloaded_path = tracker.download_artifact(
        run_id, "test.txt", destination_path=str(download_dir)
    )

    assert os.path.exists(downloaded_path)
    assert downloaded_path == str(download_dir / "test.txt")
    with open(downloaded_path, "r") as f:
        assert f.read() == dummy_content


def test_run_tags_workflow(tracker):
    """Tests the full workflow for getting and setting run tags."""
    with tracker.start_run(experiment_name="tags-workflow-test") as run_id:
        initial_tags = {"status": "running", "validated": False}
        tracker.set_run_tags(initial_tags)

        tags = tracker.get_run_tags()
        assert tags == initial_tags

        tags["status"] = "complete"  # Update a value
        del tags["validated"]  # Delete a key
        tags["new_tag"] = "value"  # Add a new value

        tracker.set_run_tags(tags)

    run_details = tracker.get_run_details(run_id)
    final_tags = run_details["meta"]["tags"]

    assert final_tags["status"] == "complete"
    assert "validated" not in final_tags
    assert final_tags["new_tag"] == "value"
