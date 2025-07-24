"""
Example script to demonstrate how to run an experiment sweep from a config file.

This script uses the `run_sweep` function from the core runelog library to
execute the experiments defined in the accompanying `sweep_conf.yaml` file.
"""
import os
from runelog.runner import run_sweep
from rich.console import Console

def main():
    """Finds the config file and starts the sweep."""
    console = Console()
    console.print("Starting experiment sweep from configuration file...")

    # Construct the path to the config file relative to this script's location
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, "sweep_conf.yaml")
    
    if not os.path.exists(config_path):
        console.print(f"Error: Could not find config file at {config_path}")
        return

    run_sweep(config_path, progress_handler=console.print)


if __name__ == "__main__":
    main()