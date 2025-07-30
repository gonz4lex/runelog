import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from runelog import get_tracker, exceptions
from runelog.runner import run_sweep

import os
import sys
import shutil
import subprocess
from datetime import datetime
from typing import Optional, List

ASCII_ART = """
 ██▀███   █    ██  ███▄    █ ▓█████  ██▓     ▒█████    ▄████ 
▓██ ▒ ██▒ ██  ▓██▒ ██ ▀█   █ ▓█   ▀ ▓██▒    ▒██▒  ██▒ ██▒ ▀█▒
▓██ ░▄█ ▒▓██  ▒██░▓██  ▀█ ██▒▒███   ▒██░    ▒██░  ██▒▒██░▄▄▄░
▒██▀▀█▄  ▓▓█  ░██░▓██▒  ▐▌██▒▒▓█  ▄ ▒██░    ▒██   ██░░▓█  ██▓
░██▓ ▒██▒▒▒█████▓ ▒██░   ▓██░░▒████▒░██████▒░ ████▓▒░░▒▓███▀▒
░ ▒▓ ░▒▓░░▒▓▒ ▒ ▒ ░ ▒░   ▒ ▒ ░░ ▒░ ░░ ▒░▓  ░░ ▒░▒░▒░  ░▒   ▒ 
  ░▒ ░ ▒░░░▒░ ░ ░ ░ ░░   ░ ▒░ ░ ░  ░░ ░ ▒  ░  ░ ▒ ▒░   ░   ░ 
  ░░   ░  ░░░ ░ ░    ░   ░ ░    ░     ░ ░   ░ ░ ░ ▒  ░ ░   ░ 
   ░        ░              ░    ░  ░    ░  ░    ░ ░        ░                                                         

   RuneLog CLI: Lightweight ML experiment tracker.

   """

HEADER = Panel(
    ASCII_ART, style="bold cyan", border_style="dim", title="[dim]v0.1.0[/dim]"
)

# Main app and subcommands
app = typer.Typer(rich_markup_mode="rich")
experiments_app = typer.Typer()
runs_app = typer.Typer()
registry_app = typer.Typer()
examples_app = typer.Typer()

app.add_typer(experiments_app, name="experiments", help="Manage experiments.")
app.add_typer(runs_app, name="runs", help="Manage runs.")
app.add_typer(registry_app, name="registry", help="Manage the model registry.")
app.add_typer(examples_app, name="examples", help="Run example scripts.")

# Console object for rich printing
console = Console()


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", help="Show the application's version and exit."
    ),
    help: bool = typer.Option(
        False, "--help", "-h", is_eager=True, help="Show this message and exit."
    ),
):
    """
    Initialize the tracker and handle top-level commands.
    """
    if version:
        console.print("Runelog CLI v0.1.0")  # TODO: runelog.__version__
        raise typer.Exit()

    if help or (ctx.invoked_subcommand is None and not version):
        console.print(HEADER)
        typer.echo(ctx.get_help())  # Display Typer help
        raise typer.Exit()

    if ctx.obj is None:
        ctx.obj = get_tracker()


## Experiments


@experiments_app.command("list")
def list_experiments(
    ctx: typer.Context,
    sort_by: Optional[str] = typer.Option(
        None,
        "--sort-by",
        "-s",
        help="Field to sort by (e.g., 'name', 'num_runs', 'last_run').",
    ),
    descending: bool = typer.Option(
        False, "--desc", "-d", help="Sort in descending order."
    ),
):
    """List all available experiments."""
    tracker = ctx.obj
    summaries = tracker.get_experiment_summaries(
        sort_by=sort_by, ascending=not descending
    )

    if not summaries:
        console.print("No experiments found.", style="yellow")
        return

    table = Table(
        "ID", "Name", "Runs", "Last Run", "Created At", title="Experiments", expand=True
    )

    for summary in summaries:
        exp_id = summary["experiment_id"]
        exp_name = summary.get("name", "—")
        created_at = _fmt_timestamp(summary.get("created_at"))
        num_runs = summary.get("num_runs", 0)
        last_run = (
            _fmt_timestamp(summary.get("last_run")) if summary.get("last_run") else "—"
        )

        table.add_row(exp_id, exp_name, str(num_runs), last_run, created_at or "—")

    console.print(table)


@experiments_app.command("get")
def get_experiment_details(
    ctx: typer.Context,
    experiment_name_or_id: str = typer.Argument(
        ..., help="The ID or name of the experiment to retrieve."
    ),
    sort_by: Optional[str] = typer.Option(
        None,
        "--sort-by",
        "-s",
        help="Metric or param to sort runs by (e.g., 'accuracy', 'param_lr').",
    ),
    descending: bool = typer.Option(
        False, "--desc", "-d", help="Sort runs in descending order."
    ),
):
    """Get details for a specific experiment."""
    tracker = ctx.obj

    try:
        experiment_id, _ = tracker._resolve_experiment_id(experiment_name_or_id)
        experiment = tracker.get_experiment(experiment_id)

        if not experiment:
            raise exceptions.ExperimentNotFound(experiment_name_or_id)

        experiment_name = experiment.get("name", "n/a")

        results_df = tracker.load_results(
            experiment_id, sort_by=sort_by, ascending=not descending
        )

        summary_panel = Panel(
            f"[bold]Name[/bold]: {experiment_name}\n"
            f"[bold]ID[/bold]: {experiment_id}\n"
            f"[bold]Number of Runs[/bold]: {len(results_df)}",
            title="Experiment Details",
            border_style="green",
        )
        console.print(summary_panel)

        if not results_df.empty:
            metric_columns = [
                col for col in results_df.columns if not col.startswith("param_")
            ]
            table = Table(
                "Run ID",
                *metric_columns,
                title=f"Runs for Experiment {experiment_id}",
                expand=True,
            )

            for run_id, row_data in results_df.iterrows():
                row_values = [run_id]
                for metric in metric_columns:
                    value = row_data.get(metric)
                    if isinstance(value, float):
                        row_values.append(f"{value:.4f}")
                    else:
                        row_values.append(str(value) if value is not None else "N/A")

                table.add_row(*row_values)

            console.print(table)
        else:
            console.print("This experiment has no runs.", style="yellow")

    except exceptions.ExperimentNotFound as e:
        console.print(f"Error: {e}", style="bold red")
        raise typer.Exit(1)


@experiments_app.command("delete")
def delete_experiment(
    ctx: typer.Context,
    experiment_id: str = typer.Argument(
        ..., help="The ID of the experiment to delete."
    ),
):
    """Delete an experiment and all of its associated runs."""
    tracker = ctx.obj

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


@experiments_app.command("export")
def export_experiment(
    ctx: typer.Context,
    experiment_name_or_id: str = typer.Argument(
        ..., help="The ID of the experiment to export."
    ),
    output_path: Optional[str] = typer.Option(
        None,
        "-o",
        "--output",
        help="Path to save the CSV file. Defaults to '<experiment_name>_export.csv'.",
    ),
):
    """Export all runs from an experiment to a CSV file."""
    tracker = ctx.obj

    try:
        experiment_id, _ = tracker._resolve_experiment_id(experiment_name_or_id)
        results_df = tracker.load_results(experiment_id)

        if results_df.empty:
            console.print(
                f"Experiment '{experiment_name_or_id}' has no runs to export.",
                style="yellow",
            )
            return

        if not output_path:
            experiment = tracker.get_or_create_experiment(experiment_name_or_id)
            exp_name = experiment.get(
                "name", f"experiment_{experiment_name_or_id}"
            ).replace(" ", "_")
            output_path = f"{exp_name}_export.csv"

        results_df.to_csv(output_path)

        console.print(
            f"Successfully exported {len(results_df)} runs to '[bold green]{output_path}[/bold green]'."
        )

    except exceptions.ExperimentNotFound as e:
        console.print(f"Error: {e}", style="bold red")
        raise typer.Exit(1)


@experiments_app.command("delete")
def delete_experiment(
    ctx: typer.Context,
    experiment_name_or_id: str = typer.Argument(
        ..., help="The name or ID of the experiment to delete."
    ),
):
    """Delete an experiment and all of its associated runs."""
    tracker = ctx.obj

    try:
        if typer.confirm(
            f"Are you sure you want to delete the experiment '{experiment_name_or_id}'? "
            "This action cannot be undone."
        ):
            tracker.delete_experiment(experiment_name_or_id)
            console.print(
                f"Experiment '{experiment_name_or_id}' has been deleted.",
                style="bold red",
            )
        else:
            console.print("Operation cancelled.")

    except exceptions.ExperimentNotFound as e:
        console.print(f"Error: {e}", style="bold red")
        raise typer.Exit(1)


## Runs


@runs_app.command("list")
def list_runs(
    ctx: typer.Context,
    experiment_name_or_id: str = typer.Argument(
        ..., help="The ID or name of the experiment whose runs you want to list."
    ),
    sort_by: Optional[str] = typer.Option(
        None,
        "--sort-by",
        "-s",
        help="Metric or param to sort by (e.g., 'accuracy', 'param_lr').",
    ),
    descending: bool = typer.Option(
        False, "--desc", "-d", help="Sort in descending order."
    ),
):
    """List all runs and their metrics for a given experiment."""
    tracker = ctx.obj
    try:
        results_df = tracker.load_results(
            experiment_name_or_id, sort_by=sort_by, ascending=not descending
        )

        if results_df.empty:
            console.print(
                f"No runs found for experiment '{experiment_name_or_id}'.",
                style="yellow",
            )
            return

        metric_columns = [
            col for col in results_df.columns if col.startswith("param_") is False
        ]
        table = Table(
            "Run ID",
            *metric_columns,
            title=f"Runs for Experiment {experiment_name_or_id}",
            expand=True,
        )

        for run_id, row_data in results_df.iterrows():
            row_values = [run_id]
            for metric in metric_columns:
                value = row_data.get(metric)
                if isinstance(value, float):
                    row_values.append(f"{value:.4f}")
                else:
                    row_values.append(str(value) if value is not None else "N/A")

            table.add_row(*row_values)

        console.print(table)

    except exceptions.ExperimentNotFound as e:
        console.print(f"Error: {e}", style="bold red")
        raise typer.Exit(1)


@runs_app.command("get")
def get_run_details(
    ctx: typer.Context,
    run_id: str = typer.Argument(..., help="The ID of the run to inspect."),
):
    """Display the detailed parameters, metrics, and artifacts for a specific run."""
    tracker = ctx.obj
    details = tracker.get_run_details(run_id)

    if not details:
        console.print(f"Error: Run with ID '{run_id}' not found.", style="bold red")
        return

    panel_content = f"[bold]Run ID[/bold]: {run_id}\n\n"

    param_table = Table(title="Parameters", expand=True)
    param_table.add_column("Parameter", style="cyan")
    param_table.add_column("Value", style="magenta")
    for key, value in details.get("params", {}).items():
        param_table.add_row(key, str(value))

    metric_table = Table(title="Metrics", expand=True)
    metric_table.add_column("Metric", style="cyan")
    metric_table.add_column("Value", style="magenta")
    for key, value in details.get("metrics", {}).items():
        metric_table.add_row(key, f"{value:.4f}")

    console.print(Panel(panel_content, title="Run Details", border_style="green"))
    console.print(param_table)
    console.print(metric_table)


@runs_app.command("download-artifact")
def download_artifact(
    ctx: typer.Context,
    run_id: str = typer.Argument(..., help="The ID of the run."),
    artifact_name: str = typer.Argument(
        ..., help="The filename of the artifact to download."
    ),
    output_path: Optional[str] = typer.Option(
        None,
        "-o",
        "--output-path",
        help="Optional directory to save the artifact. Defaults to the current directory.",
    ),
):
    """Download an artifact from a specific run."""
    tracker = ctx.obj

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


@runs_app.command("compare")
def compare_runs(
    ctx: typer.Context,
    run_ids: List[str] = typer.Argument(..., help="Two or more run IDs to compare."),
):
    """Compare the parameters and metrics of multiple runs side-by-side."""
    tracker = ctx.obj

    if len(run_ids) < 2:
        console.print(
            "Error: You must provide at least two run IDs to compare.", style="bold red"
        )
        raise typer.Exit(1)

    run_details = {}
    for run_id in run_ids:
        details = tracker.get_run_details(run_id)
        if not details:
            console.print(
                f"Warning: Run with ID '{run_id}' not found. Skipping.", style="yellow"
            )
            continue
        run_details[run_id] = details

    if len(run_details) < 2:
        console.print(
            "Error: Could not find at least two valid runs to compare.",
            style="bold red",
        )
        raise typer.Exit(1)

    # Params
    param_table = Table(title="Parameter Comparison", expand=True)
    param_table.add_column("Parameter", style="cyan")
    all_param_keys = sorted(
        set(key for data in run_details.values() for key in data["params"])
    )

    for run_id in run_details:
        param_table.add_column(run_id)

    for key in all_param_keys:
        values = [run_details[run_id]["params"].get(key) for run_id in run_details]
        # Highlight parameters that are different across runs
        if len(set(map(str, values))) > 1:
            row_values = [f"[yellow]{str(v)}[/yellow]" for v in values]
        else:
            row_values = [str(values[0])] * len(values)
        param_table.add_row(key, *row_values)

    # Metrics
    metric_table = Table(title="Metric Comparison", expand=True)
    metric_table.add_column("Metric", style="cyan")
    all_metric_keys = sorted(
        set(key for data in run_details.values() for key in data["metrics"])
    )

    for run_id in run_details:
        metric_table.add_column(run_id)

    for key in all_metric_keys:
        values = [run_details[run_id]["metrics"].get(key) for run_id in run_details]
        # Highlight max values
        valid_values = [v for v in values if v is not None]
        max_value = max(valid_values) if valid_values else None

        row_values = []
        for v in values:
            if v == max_value:
                row_values.append(f"[bold green]{v:.4f}[/bold green]")
            elif v is not None:
                row_values.append(f"{v:.4f}")
            else:
                row_values.append("N/A")
        metric_table.add_row(key, *row_values)

    console.print(param_table)
    console.print(metric_table)


## Registry


@registry_app.command("register")
def register_model(
    ctx: typer.Context,
    run_id: str = typer.Argument(..., help="The ID of the run containing the model."),
    artifact_name: str = typer.Argument(
        ..., help="The filename of the model artifact (e.g., 'model.pkl')."
    ),
    model_name: str = typer.Argument(..., help="The name to register the model under."),
):
    """Register a model from a run's artifact to the model registry."""
    tracker = ctx.obj
    try:
        console.print(
            f"Registering artifact '[bold cyan]{artifact_name}[/bold cyan]' from run '[bold yellow]{run_id}[/bold yellow]'..."
        )
        version = tracker.register_model(run_id, artifact_name, model_name)
        console.print(
            f"Successfully registered model '[bold green]{model_name}[/bold green]' as version [bold blue]{version}[/bold blue]."
        )
    except (exceptions.RunNotFound, exceptions.ArtifactNotFound) as e:
        console.print(f"Error: {e}", style="bold red")
        raise typer.Exit(1)


@registry_app.command("list")
def list_registered_models(
    ctx: typer.Context,
    sort_by: Optional[str] = typer.Option(
        "timestamp", "--sort-by", "-s", help="Sort by 'name', 'version', or 'timestamp'."
    ),
    descending: bool = typer.Option(
        True, "--desc", "-d", help="Sort in descending order."
    ),
):
    """List all models in the registry."""
    tracker = ctx.obj

    try:
        model_names = tracker.list_registered_models()

        if not model_names:
            console.print("No models found in the registry.", style="yellow")
            return
        
        all_models_data = []
        for name in model_names:
            versions = tracker.get_model_versions(name)
            if versions:
                all_models_data.append(versions[0])

        sort_key_map = {
            "name": "model_name",
            "version": "version",
            "timestamp": "registration_timestamp",
        }
        actual_sort_key = sort_key_map.get(sort_by, "model_name")

        if actual_sort_key == "version":
            all_models_data.sort(
                key=lambda x: int(x.get(actual_sort_key, 0)), reverse=not descending
            )
        else:
            all_models_data.sort(
                key=lambda x: str(x.get(actual_sort_key, "")).lower(),
                reverse=not descending,
            )

        table = Table(
            "Model Name",
            "Latest Version",
            "Registered On",
            "Tags",
            title="Models in Registry",
            expand=True,
        )

        for data in all_models_data:
            timestamp = _fmt_timestamp(data.get("registration_timestamp"))
            tags = data.get("tags", {})
            tag_str = (
                ", ".join([f"{k}={v}" for k, v in tags.items()]) if tags else "none"
            )
            table.add_row(
                data.get("model_name", "N/A"),
                data.get("version", "N/A"),
                timestamp,
                tag_str,
            )

        console.print(table)

    except Exception as e:
        console.print(f"An error occurred: {e}", style="bold red")
        raise typer.Exit(1)


@registry_app.command("get-versions")
def list_registered_model_versions(
    ctx: typer.Context,
    model_name: str = typer.Argument(
        ...,
        help="The name of the registered model. Use '[bold cyan]runelog registry list[/bold cyan]' to see options.",
    ),
    sort_by: Optional[str] = typer.Option(
        "version",
        "--sort-by",
        "-s",
        help="Field to sort by (e.g., 'version', 'registration_timestamp').",
    ),
    ascending: bool = typer.Option(
        True,
        "--desc",
        "-d",
        help="Sort in ascending order. Defaults to true (newest first).",
    ),
):
    """List all versions of a model in the registry."""
    tracker = ctx.obj

    try:
        versions = tracker.get_model_versions(
            model_name=model_name, sort_by=sort_by, ascending=ascending
        )

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
            expand=True,
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
    ctx: typer.Context,
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
    tracker = ctx.obj

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
    ctx: typer.Context,
    model_name: str = typer.Argument(..., help="The name of the model to serve."),
    version: str = "latest",
    port: int = 8000,
    # ...
):
    # runelog registry serve my-model --version latest --port 8000
    console.print(
        "Model serving is not yet supported in RuneLog 0.1.0", style="bold red"
    )


# UI


@app.command()
def ui():
    """Launch the Runelog Streamlit web UI."""
    console.print("🚀 Launching the Streamlit UI...", style="bold green")

    cli_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(cli_dir, "..", ".."))
    app_path = os.path.join(project_root, "app", "main.py")

    if not os.path.exists(app_path):
        console.print(
            f"Error: Could not find app entrypoint at {app_path}", style="bold red"
        )
        raise typer.Exit(1)

    command = [sys.executable, "-m", "streamlit", "run", app_path]

    try:
        subprocess.run(command)
    except Exception as e:
        console.print(f"Error launching Streamlit UI: {e}", style="bold red")
        raise typer.Exit(1)


# Examples


def _run_example(script_name: str):
    """Helper function to find and execute an example script."""
    console.print(f"▶ Running example: [bold green]{script_name}[/bold green]\n")

    cli_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(cli_dir, "..", ".."))
    script_path = os.path.join(project_root, "examples", script_name)

    if not os.path.exists(script_path):
        console.print(
            f"Error: Example script not found at {script_path}", style="bold red"
        )
        raise typer.Exit(1)

    command = [sys.executable, "-u", script_path]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"\nError running script '{script_name}': {e}", style="bold red")
        raise typer.Exit(1)


@examples_app.command("minimal")
def run_minimal_example():
    """Run the minimal_tracking.py example script."""
    _run_example("minimal_tracking.py")


@examples_app.command("train")
def run_train_example():
    """Run the train_model.py example script."""
    _run_example("train_model.py")


@examples_app.command("sweep")
def run_sweep_example():
    """Run the sweep.py example script."""
    _run_example("sweep/sweep.py")


# Sweep


@app.command()
def sweep(
    config_path: str = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to the sweep config YAML file.",
        exists=True,
    )
):
    """Run a series of experiments from a configuration file."""
    try:
        console.print(
            f"🚀 Loading configuration from: [bold green]{config_path}[/bold green]"
        )

        run_sweep(config_path, progress_handler=console.print)

        console.print("\nSweep finished successfully! ✨", style="bold green")
    except Exception as e:
        console.print(f"\n❌ An error occurred during the sweep: {e}", style="bold red")
        raise typer.Exit(code=1)


# Utils


def _fmt_timestamp(ts):
    """Helper function to format timestamps."""
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except ValueError:
            return ts
    if isinstance(ts, datetime):
        return ts.strftime("%Y-%m-%d %H:%M")
    return None
