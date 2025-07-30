# RuneLog Command-Line Interface (CLI)

The RuneLog CLI provides a powerful, terminal-based interface for interacting with your experiments, runs, and model registry. It's designed to be a fast and efficient alternative to the web UI for common tasks.

## Installation

The CLI is automatically installed when you install the `runelog` package:

```bash
pip install runelog
```

## General Usage

All commands follow a standard structure. You can get help for any command or sub-command by adding the `--help` flag.

```bash
runelog [SUBCOMMAND] [COMMAND] [ARGUMENTS] [OPTIONS]
runelog experiments list --help
```

-----

## Experiment Commands (`runelog experiments`)

Manage and inspect your experiments.

### `runelog experiments list`

Lists all available experiments in your project.

```bash
runelog experiments list
```

### `runelog experiments get`

Get details for a specific experiment.

> 🚧 **Note:** This command is not yet fully implemented.

```bash
runelog experiments get <EXPERIMENT_NAME_OR_ID>
```

### `runelog experiments delete`

Deletes an experiment and all of its associated runs. This action is irreversible.

```bash
runelog experiments delete <EXPERIMENT_NAME_OR_ID>
```

### `runelog experiments export`

Exports all run data (parameters and metrics) from a specific experiment to a CSV file.

```bash
runelog experiments export <EXPERIMENT_NAME_OR_ID> [OPTIONS]
```

**Options:**

  - `-o, --output TEXT`: Path to save the CSV file. Defaults to `<experiment_name>_export.csv`.

**Example:**

```bash
runelog experiments export 0 -o my_experiment_results.csv
```

-----

## Run Commands (`runelog runs`)

Manage and inspect individual runs.

### `runelog runs list`

Lists all runs for a given experiment.


```bash
runelog runs list <EXPERIMENT_NAME_OR_ID>
```

### `runelog runs get`

Displays the detailed parameters, metrics, and artifacts for a specific run.

```bash
runelog runs get <RUN_ID>
```

### `runelog runs download-artifact`

Downloads an artifact file from a specific run to your local machine.

```bash
runelog runs download-artifact <RUN_ID> <ARTIFACT_NAME> [OPTIONS]
```

**Options:**

  - `-o, --output-path TEXT`: Directory to save the artifact. Defaults to the current directory.

**Example:**

```bash
runelog runs download-artifact 8b2604f4 model.pkl -o ./downloaded_models/
```

### `runelog runs compare`

Displays a side-by-side comparison of the parameters and metrics for two or more runs.

```bash
runelog runs compare <RUN_ID_1> <RUN_ID_2> ...
```

-----

## Registry Commands (`runelog registry`)

Manage the model registry.

### `runelog registry list`

Lists all models in the registry and shows their latest version information.

```bash
runelog registry list
```

### `runelog registry get-versions`

Lists all available versions for a specific registered model.

```bash
runelog registry get-versions <MODEL_NAME>
```

### `runelog registry tag`

Adds or removes tags for a specific model version.

```bash
runelog registry tag <MODEL_NAME> <VERSION> [OPTIONS]
```

**Options:**

  - `-a, --add TEXT`: Tag to add/update in `key=value` format.
  - `-r, --remove TEXT`: Tag key to remove.

**Example:**

```bash
runelog registry tag my-model 2 --add status=production --remove status=staging
```

-----

## Interface Commands

### `runelog ui`

Launches the Runelog Streamlit web UI in your browser.

**Usage:**

```bash
runelog ui
```

## Example Commands

### `runelog examples`

Run the example scripts included with the library.

**Usage:**

```bash
runelog examples [COMMAND]
```

**Commands:**

  - `minimal`: Runs the `minimal_tracking.py` example.
  - `train`: Runs the `train_model.py` example.