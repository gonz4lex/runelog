import streamlit as st
from tracker import Tracker

tracker = Tracker()


def show_run_details(run_id):
    """
    Helper function to display run details.
    """
    st.subheader(f"Details for Run: `{run_id}`")
    details = tracker.get_run_details(run_id)

    if not details:
        st.error("Could not load details for this run.")
        return

    # Create two columns for params and metrics
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Parameters")
        if details["params"]:
            st.json(details["params"])
        else:
            st.info("No parameters logged.")

    with col2:
        st.markdown("#### Metrics")
        if details["metrics"]:
            # Display metrics in a more visually appealing way
            for key, value in details["metrics"].items():
                st.metric(label=key, value=round(value, 4))
        else:
            st.info("No metrics logged.")

    st.markdown("#### Artifacts")
    if details["artifacts"]:
        st.table(details["artifacts"])
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
    selected_experiment_name = experiment_map[selected_experiment_id]
    st.header(f"🔬 Experiment: {selected_experiment_name}")

    results_df = tracker.load_results(selected_experiment_id)

    st.markdown("### Runs")
    if not results_df.empty:
        run_ids = results_df.index.tolist()
        run_ids.insert(0, None)

        selected_run_id = st.selectbox(
            "Select a run to view its details (you can type to search):",
            options=run_ids,
        )

        st.dataframe(results_df, use_container_width=True)
        
        if selected_run_id:
            st.divider()
            show_run_details(selected_run_id)

    else:
        st.info("This experiment has no runs yet.")
