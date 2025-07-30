"""Streamlit page for the Experiment Explorer.

This module renders the primary user interface for Browse and analyzing
experiment runs. It allows users to select an experiment from a dropdown,
view all associated runs in a summary table, and then select a specific
run to inspect its detailed parameters, metrics, and artifacts.
"""

import os

import re
import pandas as pd
import streamlit as st
import plotly.express as px

from runelog import get_tracker
from app.components import render_sidebar

st.set_page_config(page_title="🔬 Explorer | Runelog", layout="wide")

render_sidebar()

st.title("🔬 Experiment Explorer")
tracker = get_tracker()


def show_actions(run_id: str, details):
    model_artifact = next(
        (art for art in details.get("artifacts", []) if art.endswith(".pkl")), None
    )

    if model_artifact:
        # Sanitize the run_id to suggest a default model name
        suggested_name = re.sub(r"[^a-zA-Z0-9-]", "-", f"model-{run_id}")

        c1, c2 = st.columns([2, 1])

        with c1:
            model_name = st.text_input(
                label="Model Name",
                value=suggested_name,
                placeholder="Enter a name for the registered model",
                label_visibility="collapsed",
            )

        with c2:
            if st.button("Add to Registry"):
                if model_name:
                    try:
                        version = tracker.register_model(
                            run_id, model_artifact, model_name
                        )
                        st.success(
                            f"Successfully registered model '{model_name}' as version {version}!"
                        )
                        st.balloons()
                    except Exception as e:
                        st.error(f"Failed to register model: {e}")
                else:
                    st.warning("Please enter a name for the model.")
    else:
        st.info("No model artifact found in this run to register.")


def show_run_details(run_id: str):
    details = tracker.get_run_details(run_id)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"Details for Run: `{run_id}`")
    with c2:
        show_actions(run_id=run_id, details=details)

    if not details:
        st.error("Could not load details for this run.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Metrics")
        metrics = details.get("metrics", {})
        if metrics:
            for key, value in metrics.items():
                st.metric(label=key, value=round(value, 4))
        else:
            st.info("No metrics logged.")

    with col2:
        st.markdown("#### Parameters")
        params = details.get("params", {})
        if params:
            params_df = pd.DataFrame(params.items(), columns=["Parameter", "Value"])
            params_df["Value"] = params_df["Value"].astype(str)
            st.table(params_df)
        else:
            st.info("No parameters logged.")

    st.markdown("#### Artifacts")
    artifacts = details.get("artifacts", [])
    if artifacts:
        for artifact_name in artifacts:
            with st.expander(f"📄 {artifact_name}"):
                try:
                    full_path = tracker.get_artifact_abspath(run_id, artifact_name)
                    extension = os.path.splitext(artifact_name)[1].lower()

                    if extension in [".png", ".jpg", ".jpeg", ".gif"]:
                        st.image(full_path, caption=artifact_name)

                    elif extension in [".txt", ".log", ".json", ".yaml", ".md", ".csv"]:
                        with open(full_path, "r") as f:
                            content = f.read()
                        st.code(content, language="text")

                    else:
                        st.info(
                            f"Preview is not available for this file type ({extension})."
                        )

                except Exception as e:
                    st.error(f"Could not load artifact '{artifact_name}': {e}")
    else:
        st.info("No artifacts logged.")


experiments = tracker.list_experiments()
if not experiments:
    st.info("No experiments found. Run a training script to create one.")
    st.stop()

experiment_map = {exp["experiment_id"]: exp["name"] for exp in experiments}

selected_experiment_id = st.selectbox(
    "Select an Experiment",
    options=list(experiment_map.keys()),
    format_func=lambda exp_id: f"{experiment_map[exp_id]} (id: {exp_id})",
)

if selected_experiment_id:
    results_df = tracker.load_results(selected_experiment_id)

    st.markdown("### Runs")
    if not results_df.empty:
        selected_run_ids = st.multiselect(
            "Select two or more runs to compare:",
            options=results_df.index.tolist(),
        )

        st.dataframe(results_df, use_container_width=True)

        if len(selected_run_ids) >= 2:
            st.markdown("### Compare Selected Runs")

            selected_rows = results_df.loc[selected_run_ids]
            metric_columns = [
                col for col in selected_rows.columns if not col.startswith("param_")
            ]

            if not metric_columns:
                st.warning("No metrics found to compare for the selected runs.")
            else:
                selected_metric = st.selectbox(
                    "Select a metric to compare:", options=metric_columns
                )
                chart_data = (
                    selected_rows[[selected_metric]]
                    .reset_index()
                    .rename(columns={"index": "run_id"})
                )

                fig = px.bar(
                    chart_data,
                    x="run_id",
                    y=selected_metric,
                    title=f"Comparison of '{selected_metric}'",
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

        elif len(selected_run_ids) == 1:
            show_run_details(selected_run_ids[0])
    else:
        st.info("This experiment has no runs yet.")
