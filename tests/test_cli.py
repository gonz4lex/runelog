import os
import pytest
from typer.testing import CliRunner

from runelog.cli import app
from runelog import get_tracker, exceptions

runner = CliRunner()


class MockModel:
    def __init__(self, val=1):
        self.val = val


@pytest.fixture
def test_tracker(tmp_path):
    """
    Creates a RuneLog tracker instance that points to a unique, temporary directory
    provided by pytest's tmp_path fixture.

    This ensures every test runs in an isolated environment.
    """
    return get_tracker(path=str(tmp_path))


class TestExperimentsCommands:
    def test_list_success(self, test_tracker):
        """Tests that 'experiments list' prints experiments when they exist."""
        test_tracker.get_or_create_experiment("test-list-success")
        result = runner.invoke(app, ["experiments", "list"], obj=test_tracker)
        assert result.exit_code == 0
        assert "test-list-success" in result.stdout

    def test_list_empty(self, test_tracker):
        """Tests 'experiments list' shows a message when no experiments exist."""
        result = runner.invoke(app, ["experiments", "list"], obj=test_tracker)

        assert result.exit_code == 0
        assert "No experiments found" in result.stdout

    def test_get_success(self, test_tracker):
        """Tests 'experiments get' with a valid ID."""
        exp_id = test_tracker.get_or_create_experiment("test-get-success")
        result = runner.invoke(app, ["experiments", "get", exp_id], obj=test_tracker)
        assert result.exit_code == 0
        assert "Experiment Details" in result.stdout
        assert "test-get-success" in result.stdout

    def test_get_not_found(self, test_tracker):
        """Tests 'experiments get' with a non-existent ID."""
        result = runner.invoke(
            app, ["experiments", "get", "nonexistent-id"], obj=test_tracker
        )
        assert result.exit_code == 1
        assert "Error: Experiment with ID 'nonexistent-id' not found" in result.stdout

    def test_export_success(self, test_tracker, tmp_path):
        """Tests 'experiments export' successfully creates a CSV."""
        exp_id = test_tracker.get_or_create_experiment("test-export")

        with test_tracker.start_run(experiment_id=exp_id):
            test_tracker.log_metric("accuracy", 0.99)

        output_file = tmp_path / "export.csv"
        result = runner.invoke(
            app,
            ["experiments", "export", exp_id, "--output", str(output_file)],
            obj=test_tracker,
        )

        assert result.exit_code == 0
        assert "Successfully exported" in result.stdout
        assert os.path.exists(output_file)

    def test_export_empty(self, test_tracker):
        """Tests 'experiments export' on an experiment with no runs."""
        exp_id = test_tracker.get_or_create_experiment("test-export-empty")
        result = runner.invoke(app, ["experiments", "export", exp_id], obj=test_tracker)
        assert result.exit_code == 0
        assert "has no runs to export" in result.stdout


class TestRunsCommands:
    def test_list_success(self, test_tracker):
        """Tests 'runs list' for an experiment with runs."""
        exp_id = test_tracker.get_or_create_experiment("test-list-run-success")

        with test_tracker.start_run(experiment_id=exp_id) as run_id:
            test_tracker.log_param("alpha", 0.5)

        result = runner.invoke(app, ["runs", "list", exp_id], obj=test_tracker)

        assert result.exit_code == 0
        assert run_id in result.stdout

    def test_list_empty(self, test_tracker):
        """Tests 'runs list' for an experiment with no runs."""
        exp_id = test_tracker.get_or_create_experiment("test-list-run-empty")
        result = runner.invoke(app, ["runs", "list", exp_id], obj=test_tracker)
        assert result.exit_code == 0
        assert "No runs found" in result.stdout

    def test_get_success(self, test_tracker):
        """Tests 'runs get' with a valid run ID."""
        exp_id = test_tracker.get_or_create_experiment("test-get-run-success")
        with test_tracker.start_run(experiment_id=exp_id) as run_id:
            test_tracker.log_param("alpha", 0.5)

        result = runner.invoke(app, ["runs", "get", run_id], obj=test_tracker)
        assert result.exit_code == 0
        assert "Run Details" in result.stdout
        assert "Parameters" in result.stdout
        assert "alpha" in result.stdout

    def test_get_not_found(self, test_tracker):
        """Tests 'runs get' with a non-existent run ID."""
        result = runner.invoke(
            app, ["runs", "get", "nonexistent-run"], obj=test_tracker
        )
        assert result.exit_code == 0
        assert "Error: Run with ID 'nonexistent-run' not found" in result.stdout


class TestRegistryCommands:
    def test_register_success(self, test_tracker):
        """Tests that 'registry register' successfully registers a model."""
        exp_id = test_tracker.get_or_create_experiment("test-register-command")

        with test_tracker.start_run(experiment_id=exp_id) as run_id:
            test_tracker.log_model(MockModel(), "model.pkl")

        model_name = "cli-registered-model"

        result = runner.invoke(
            app,
            ["registry", "register", run_id, "model.pkl", model_name],
            obj=test_tracker,
        )

        assert result.exit_code == 0
        assert "Successfully registered model" in result.stdout
        assert model_name in result.stdout

        registered_models = test_tracker.list_registered_models()
        assert model_name in registered_models

    def test_list_success(self, test_tracker):
        """Tests 'registry list' when models exist."""
        exp_id = test_tracker.get_or_create_experiment("test-list-registry-success")

        with test_tracker.start_run(experiment_id=exp_id) as run_id:
            test_tracker.log_model(MockModel(), "model.pkl")
        test_tracker.register_model(run_id, "model.pkl", "my-model")

        result = runner.invoke(app, ["registry", "list"], obj=test_tracker)
        assert result.exit_code == 0
        assert "my-model" in result.stdout

    def test_list_empty(self, test_tracker):
        """Tests 'registry list' when no models exist."""
        result = runner.invoke(app, ["registry", "list"], obj=test_tracker)
        assert result.exit_code == 0
        assert "No models found" in result.stdout

    def test_tag_success(self, test_tracker):
        """Tests 'registry tag' can add a tag successfully."""
        exp_id = test_tracker.get_or_create_experiment("test-tags")

        with test_tracker.start_run(experiment_id=exp_id) as run_id:
            test_tracker.log_model(MockModel(), "model.pkl")
        test_tracker.register_model(run_id, "model.pkl", "tag-model")

        result = runner.invoke(
            app,
            ["registry", "tag", "tag-model", "1", "--add", "status=production"],
            obj=test_tracker,
        )
        assert result.exit_code == 0
        assert "Updated tags successfully" in result.stdout
        assert "status" in result.stdout
        assert "production" in result.stdout

    def test_tag_not_found(self, test_tracker):
        """Tests 'registry tag' on a non-existent model version."""
        result = runner.invoke(
            app, ["registry", "tag", "no-model", "1", "--add", "a=b"], obj=test_tracker
        )
        assert result.exit_code == 1
        assert "Error: Model 'no-model' version '1' not found" in result.stdout

    def test_register_run_not_found(self):
        """Tests that 'registry register' fails gracefully for a non-existent run."""
        result = runner.invoke(
            app,
            ["registry", "register", "nonexistent-run-id", "model.pkl", "test-model"],
        )

        assert result.exit_code != 0
        assert "Error: Run with ID 'nonexistent-run-id' not found" in result.stdout
