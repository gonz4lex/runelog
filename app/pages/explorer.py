import pandas as pd
import streamlit as st

from src.runelog import get_tracker
from app.components import render_sidebar

st.set_page_config(
    page_title="🔬 Explorer | Runelog",
    layout="wide"
)

render_sidebar()
st.title("🔬 Experiment Explorer")


tracker = get_tracker()

def show_run_details(run_id):
    st.subheader(f"Details for Run: `{run_id}`")
    details = tracker.get_run_details(run_id)

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
            params_df['Value'] = params_df['Value'].astype(str)
            st.table(params_df)
        else:
            st.info("No parameters logged.")
    
    st.markdown("#### Artifacts")
    artifacts = details.get("artifacts", [])
    if artifacts:
        st.table(artifacts)
    else:
        st.info("No artifacts logged.")


experiments = tracker.list_experiments()
if not experiments:
    st.info("No experiments found. Run a training script to create one.")
    st.stop()

experiment_map = {exp['experiment_id']: exp['name'] for exp in experiments}

selected_experiment_id = st.selectbox(
    "Select an Experiment",
    options=list(experiment_map.keys()),
    format_func=lambda exp_id: f"{experiment_map[exp_id]} (id: {exp_id})"
)

if selected_experiment_id:
    results_df = tracker.load_results(selected_experiment_id)

    st.markdown("### Runs")
    if not results_df.empty:
        selected_run_id = st.selectbox(
            "Select a run to view its details (you can type to search):",
            options=[None] + results_df.index.tolist(),
        )
        st.dataframe(results_df, use_container_width=True)
        
        if selected_run_id:
            st.divider()
            show_run_details(selected_run_id)
    else:
        st.info("This experiment has no runs yet.")