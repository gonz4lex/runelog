import pytest
from runelog import get_tracker, exceptions


class MockModel:
    def __init__(self, val=1):
        self.val = val


def test_core_api_workflow(tmp_path):
    """
    Tests the full end-to-end workflow of the core RuneLog Python API.
    """
    # 1. Perform full user workflow

    # Initialize the tracker in an isolated directory
    tracker = get_tracker(path=str(tmp_path))

    exp_name = "Core API Test"
    model_name = "core-test-model"

    # Create an experiment and run to log data
    exp_id = tracker.get_or_create_experiment(exp_name)
    exp_id_again = tracker.get_or_create_experiment(exp_name)
    assert exp_id == exp_id_again  # assert creation idempotency

    with tracker.start_run(experiment_id=exp_id) as run_id:
        tracker.log_param("alpha", 0.5)
        tracker.log_metric("accuracy", 0.95)
        tracker.log_model(MockModel(val=123), "model.pkl")

    # Register the model from the completed run and add a tag
    version = tracker.register_model(run_id, "model.pkl", model_name)
    tracker.add_model_tags(model_name, version, {"status": "validated"})

    # 2. Verify the state using the library's read methods

    # Verify experiment
    experiments = tracker.list_experiments()
    assert len(experiments) == 1
    assert experiments[0]["name"] == exp_name

    # Verify run details
    run_details = tracker.get_run_details(run_id)
    assert run_details is not None
    assert run_details["params"]["alpha"] == 0.5
    assert run_details["metrics"]["accuracy"] == 0.95
    assert "model.pkl" in run_details["artifacts"]

    # Verify results DataFrame
    results_df = tracker.load_results(exp_id)
    assert not results_df.empty
    assert results_df.index[0] == run_id
    assert results_df.loc[run_id]["param_alpha"] == 0.5
    assert results_df.loc[run_id]["accuracy"] == 0.95

    # Verify registry state
    registered_models = tracker.list_registered_models()
    assert model_name in registered_models

    model_versions = tracker.get_model_versions(model_name)
    assert len(model_versions) == 1
    assert model_versions[0]["version"] == "1"
    assert model_versions[0]["source_run_id"] == run_id

    tags = tracker.get_model_tags(model_name, "1")
    assert tags["status"] == "validated"

    # Verify tag update
    tracker.add_model_tags(model_name, "1", {"status": "production"})
    tags = tracker.get_model_tags(model_name, "1")
    assert tags["status"] == "production"

    # Verify model loading
    loaded_model = tracker.load_registered_model(model_name, version="latest")
    assert isinstance(loaded_model, MockModel)
    assert loaded_model.val == 123

    # Verify latest version logic
    with tracker.start_run(experiment_id=exp_id) as run_id_2:
        tracker.log_model(MockModel(val=456), "model.pkl")
    tracker.register_model(run_id_2, "model.pkl", model_name)  # This creates version 2

    model_versions = tracker.get_model_versions(model_name)
    assert len(model_versions) == 2
    assert model_versions[0]["version"] == "2"

    # Verify latest model is actually the new one
    latest_model = tracker.load_registered_model(model_name, version="latest")
    assert latest_model.val == 456
