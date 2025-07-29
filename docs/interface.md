# Runelog UI User Guide

Welcome to the user guide for the RuneLog web interface, which will walk you through the main features of the application.

## Launching the App

To start the user interface, navigate to your project's root directory in your terminal and run:

```bash
streamlit run app/main.py
```

You can also use the CLI:

```bash
streamlit ui
```

This will open the application in a new browser tab.

-----

## The Experiment Explorer 🔬

This is the main view for analyzing your experiment runs. After selecting an experiment from the top dropdown, you can either inspect a single run or compare multiple runs.

### Selecting an Experiment

Use the dropdown menu at the top of the page to choose which experiment you want to inspect. The table will automatically update to show all the runs for that experiment.

### Inspecting a Single Run

The main table gives you a high-level overview of all your runs. To see the details for a specific run:

1.  Find the `run_id` of the run you are interested in from the table.
2.  Use the searchable dropdown labeled **"Select a run to view its details"** to find and select that `run_id`.
3.  A detailed view will appear below the table, showing the specific **Parameters**, **Metrics**, and **Artifacts** for that run.

### Comparing Multiple Runs

The explorer also allows you to visually compare the performance of multiple runs.

1. Use the "Select runs to view details or compare" dropdown and select two or more runs from the list.
2. A "Compare Selected Runs" section will appear below the table.
3. Choose a metric from the new dropdown to plot the comparison. A bar chart will instantly show you the results, making it easy to see which run performed best.

-----

## The Model Registry 📚

Navigate to the Model Registry using the sidebar link. This page allows you to view all your "blessed" or production-ready models that have been versioned for easy access.

### Viewing Model Versions

1.  On the left, select a registered model from the list.
2.  The right side of the screen will update to show all available versions for that model, with the latest version at the top.
3.  Click on any version to expand it and see its details.

### Understanding Version Details

Inside each version's expander, you will find:

  - **Tags**: Any tags assigned to this version (e.g., `status: production`).
  - **Source Run ID**: A reference to the original experiment run that produced this model.
  - **Metrics & Parameters**: The exact metrics and parameters from the source run, giving you full traceability.
  - **Source Artifacts**: A list of all artifacts that were created during the source run.