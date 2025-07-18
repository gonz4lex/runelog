import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from runelog import get_tracker
from . import exceptions

import os
import shutil
from datetime import datetime
from typing import Optional, List

# Main app and sub-commands
app = typer.Typer(help="RuneLog CLI: Lightweight ML experiment tracker.")
experiments_app = typer.Typer()
runs_app = typer.Typer()
registry_app = typer.Typer()

app.add_typer(experiments_app, name="experiments", help="Manage experiments.")
app.add_typer(runs_app, name="runs", help="Manage runs.")
app.add_typer(registry_app, name="registry", help="Manage the model registry.")

# Console object for rich printing
console = Console()
tracker = get_tracker()

## Experiments


@experiments_app.command("list")
def list_experiments():
    """List all available experiments."""
    experiments = tracker.list_experiments()
    if not experiments:
        console.print("No experiments found.", style="yellow")
        return

    table = Table("ID", "Name", title="Experiments")
    for exp in experiments:
        table.add_row(exp["experiment_id"], exp["name"])

    console.print(table)


@experiments_app.command("get")
def get_experiment_details(
    experiment_id: str = typer.Argument(
        ..., help="The ID of the experiment to retrieve."
    )
):
    """Get details for a specific experiment."""
    console.print(
        f"Fetching details for experiment [bold cyan]{experiment_id}[/bold cyan]..."
    )
    console.print(
        "[yellow]Info:[/yellow] This feature requires enhancing `tracker.py` to fetch a single experiment's details."
    )


@experiments_app.command("delete")
def delete_experiment(
    experiment_id: str = typer.Argument(..., help="The ID of the experiment to delete.")
):
    """Delete an experiment and all of its associated runs."""
    console.print(
        f"You are about to delete experiment '{experiment_id}'.",
        style="bold yellow",
    )

    if typer.confirm("This action cannot be undone. Are you sure?"):
        try:
            # TODO: add this method
            # tracker.delete_experiment(experiment_id)
            console.print(
                "`delete_experiment()` is not yet implemented.", style="bold red"
            )
        except exceptions.ExperimentNotFound as e:
            console.print(f"Error: {e}", style="bold red")
    else:
        console.print("Operation cancelled.")


## Runs

# TODO:
# rich table comparisons: runelog runs compare <run_id_1> <run_id_2>
# export data for other tools: runelog runs export <experiment_id> --output results.csv


@runs_app.command("list")
def list_runs(
    experiment_id: str = typer.Argument(
        ..., help="The ID of the experiment whose runs you want to list."
    )
):
    """List all runs for a given experiment."""
    # TODO
    console.print(
        f"Listing runs for experiment [bold cyan]{experiment_id}[/bold cyan]..."
    )


@runs_app.command("get")
def get_run_details(
    run_id: str = typer.Argument(..., help="The ID of the run to inspect.")
):
    """Display the detailed parameters, metrics, and artifacts for a specific run."""
    details = tracker.get_run_details(run_id)

    if not details:
        console.print(f"Error: Run with ID '{run_id}' not found.", style="bold red")
        return

    panel_content = f"[bold]Run ID[/bold]: {run_id}\n\n"

    param_table = Table(title="Parameters")
    param_table.add_column("Parameter", style="cyan")
    param_table.add_column("Value", style="magenta")
    for key, value in details.get("params", {}).items():
        param_table.add_row(key, str(value))

    metric_table = Table(title="Metrics")
    metric_table.add_column("Metric", style="cyan")
    metric_table.add_column("Value", style="magenta")
    for key, value in details.get("metrics", {}).items():
        metric_table.add_row(key, f"{value:.4f}")

    console.print(Panel(panel_content, title="Run Details", border_style="green"))
    console.print(param_table)
    console.print(metric_table)


@runs_app.command("download-artifact")
def download_artifact(
    run_id: str = typer.Argument(..., help="The ID of the run."),
    artifact_name: str = typer.Argument(..., help="The filename of the artifact to download."),
    output_path: Optional[str] = typer.Option(
        None, "-o", "--output-path",
        help="Optional directory to save the artifact. Defaults to the current directory."
    )
):
    """Download an artifact from a specific run."""
    try:
        source_path = tracker.get_artifact_path(run_id, artifact_name)

        if output_path:
            # If an output path is provided, create it if it doesn't exist
            os.makedirs(output_path, exist_ok=True)
            destination_path = os.path.join(output_path, artifact_name)
        else:
            # Default to the current working directory
            destination_path = artifact_name

        shutil.copy(source_path, destination_path)

        console.print(
            f"Artifact '[bold cyan]{artifact_name}[/bold cyan]' downloaded successfully to '[bold green]{os.path.abspath(destination_path)}[/bold green]'."
        )

    except (exceptions.RunNotFound, exceptions.ArtifactNotFound) as e:
        console.print(f"Error: {e}", style="bold red")
        raise typer.Exit(1)

## Registry


@registry_app.command("list")
def list_registered_models():
    """List all models in the registry."""
    console.print("Listing registered models...")
    try:
        model_names = tracker.list_registered_models()

        if not model_names:
            console.print("No models found in the registry.", style="yellow")
            return

        table = Table(
            "Model Name",
            "Latest Version",
            "Registered On",
            "Tags",
            title="Models in Registry",
        )

        for name in model_names:
            versions = tracker.get_model_versions(name)
            if not versions:
                # Unlikely but just in case
                table.add_row(name, "N/A", "N/A", "No versions found")
                continue

            latest_version = versions[0]

            # Format timestamp and tags
            timestamp = datetime.fromisoformat(
                latest_version.get("registration_timestamp", "")
            ).strftime("%Y-%m-%d %H:%M")
            tags = latest_version.get("tags", {})
            tag_str = (
                ", ".join([f"{k}={v}" for k, v in tags.items()]) if tags else "none"
            )

            table.add_row(
                name, latest_version.get("version", "N/A"), timestamp, tag_str
            )

        console.print(table)

    except Exception as e:
        console.print(f"An error occurred: {e}", style="bold red")
        raise typer.Exit(1)


@registry_app.command("get-versions")
def list_registered_model_versions(
    model_name: str = typer.Argument(
        ...,
        help="The name of the registered model. Use '[bold cyan]runelog registry list[/bold cyan]' to see options.",
    )
):
    """List all versions of a model in the registry."""
    try:
        versions = tracker.get_model_versions(model_name=model_name)

        if not versions:
            console.print(
                f"No versions found for model '[bold cyan]{model_name}[/bold cyan]'.",
                style="yellow",
            )
            return

        table = Table(
            "Version",
            "Registered On",
            "Source Run ID",
            "Tags",
            title=f"Versions for [bold cyan]{model_name}[/bold cyan]",
        )

        for version_info in versions:
            # Format the timestamp for readability
            timestamp = datetime.fromisoformat(
                version_info.get("registration_timestamp", "")
            ).strftime("%Y-%m-%d %H:%M")

            # Format the tags dictionary
            tags = version_info.get("tags", {})
            tag_str = (
                ", ".join([f"{k}={v}" for k, v in tags.items()]) if tags else "none"
            )

            table.add_row(
                version_info.get("version", "N/A"),
                timestamp,
                version_info.get("source_run_id", "N/A"),
                tag_str,
            )

            console.print(table)

    except Exception as e:
        console.print(f"An error occurred: {e}", style="bold red")
        raise typer.Exit(1)


@registry_app.command("tag")
def manage_tags(
    model_name: str = typer.Argument(..., help="The name of the registered model."),
    version: str = typer.Argument(..., help="The version to modify."),
    add_tags: Optional[List[str]] = typer.Option(
        None,
        "--add",
        "-a",
        help="Tags to add or update in 'key=value' format. Can be used multiple times.",
    ),
    remove_tags: Optional[List[str]] = typer.Option(
        None, "--remove", "-r", help="Tag keys to remove. Can be used multiple times."
    ),
):
    """Add or remove tags for a specific model version in the registry."""
    try:
        # Get the existing tags
        current_tags = tracker.get_model_tags(model_name, version)
        console.print(f"Current tags: {current_tags}")

        # Remove
        if remove_tags:
            for key in remove_tags:
                if key in current_tags:
                    del current_tags[key]
                    console.print(f"Tag '{key}' removed.", style="yellow")

        # Add/update
        if add_tags:
            for tag_pair in add_tags:
                if "=" not in tag_pair:
                    console.print(
                        f"Error: Invalid tag format '{tag_pair}'. Use 'key=value'.",
                        style="bold red",
                    )
                    continue
                key, value = tag_pair.split("=", 1)
                current_tags[key] = value
                console.print(f"Tag '{key}' set to '{value}'.", style="green")

        # Save updated tags
        tracker.add_model_tags(model_name, version, current_tags)
        console.print("\nUpdated tags successfully!", style="bold green")
        console.print(f"Final tags: {current_tags}")

    except exceptions.ModelVersionNotFound as e:
        console.print(f"Error: {e}", style="bold red")
        raise typer.Exit(1)

# TODO
@registry_app.command("serve")
def serve_model(
    model_name: str = typer.Argument(..., help="The name of the model to serve."),
    version: str = "latest",
    port: int = 8000,
    # ...
):
    # runelog registry serve my-model --version latest --port 8000
    console.print(
            "Model serving is not yet supported in RuneLog 0.1.0", style="bold red"
        )
    

def main():
    app()
