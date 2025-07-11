import pandas as pd
import streamlit as st

def render_sidebar_footer():
    """
    A reusable component to create a footer in the sidebar,
    anchored to the bottom.
    """
    st.markdown(
        """
        <style>
            .sidebar-footer {
                position: fixed;
                bottom: 10px;
                width: 280px; /* TODO: Adjust this width to match sidebar */
            }
        </style>

        <div class="sidebar-footer">
            <details>
                <summary>About</summary>
                <p>
                Runelog allows you to explore and compare your ML experiments. 
                Data is read directly from the local .mlruns directory.
                <br><br>
                </p>
            </details>
            <a href="https://github.com/gonz4lex/runelog" target="_blank">View on GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """
    A reusable function to create the sidebar content for all pages.
    """

    with st.sidebar:
        st.title("📜 Runelog")
        st.markdown("A lightweight ML experiment tracker.")
        st.divider()

        st.page_link("pages/explorer.py", label="Experiment Explorer", icon="🔬")
        st.page_link("pages/registry.py", label="Model Registry", icon="📚")

        render_sidebar_footer()


def display_version_details(tracker, version_info):
    """
    A reusable component to display the details for a model version inside an expander.
    """
    tags = version_info.get("tags", {})
    if tags:
        # Custom CSS for tags
        tag_style = """
            <style>
                .tag-badge {
                    display: inline-block;
                    padding: 4px 8px;
                    margin: 0 5px 5px 0;
                    background-color: #262730; /* Secondary background color */
                    color: #FAFAFA;
                    border-radius: 10px;
                    font-size: 13px;
                }
            </style>
        """
        tags_html = "".join(
            [
                f'<span class="tag-badge">{key}: {value}</span>'
                for key, value in tags.items()
            ]
        )
        st.markdown(tag_style + tags_html, unsafe_allow_html=True)

    # Display Source Run Details
    source_run_id = version_info.get("source_run_id")
    st.markdown(f"Source Run ID: `{source_run_id}`")

    if source_run_id:
        run_details = tracker.get_run_details(source_run_id)

        if run_details:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Metrics**")
                metrics = run_details.get("metrics", {})
                if metrics:
                    for key, value in metrics.items():
                        st.metric(label=key, value=f"{value:.4f}")
                else:
                    st.info("No metrics logged.")

            with col2:
                st.markdown("**Parameters**")
                params = run_details.get("params", {})
                if params:
                    params_df = pd.DataFrame(
                        params.items(), columns=["Parameter", "Value"]
                    )
                    params_df["Value"] = params_df["Value"].astype(str)
                    st.table(params_df)
                else:
                    st.info("No parameters logged.")

            st.markdown("**Source Artifacts**")
            artifacts = run_details.get("artifacts", [])
            if artifacts:
                st.table(artifacts)
            else:
                st.info("No artifacts logged.")

    timestamp_str = version_info.get("registration_timestamp")
    if timestamp_str:
        dt_object = __import__("datetime").datetime.fromisoformat(timestamp_str)
        formatted_timestamp = dt_object.strftime("%B %d, %Y, %I:%M %p")
        st.text(f"Registered on: {formatted_timestamp}")
    else:
        st.text("Registered on: N/A")
