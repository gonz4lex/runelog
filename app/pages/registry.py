"""
Streamlit page for viewing and exploring the Model Registry.

This page provides a two-column layout to select a registered model and view
all of its version details, including tags, source run info, metrics, and
parameters.
"""

import streamlit as st

from runelog import get_tracker
from app.components import display_version_details, render_sidebar

st.set_page_config(
    page_title="📚 Registry | Runelog",
    layout="wide"
)

render_sidebar()

tracker = get_tracker()

registered_models = tracker.list_registered_models()

if not registered_models:
    st.info("No models have been registered yet.")
    st.stop()

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("#### Registered Models")
    selected_model_name = st.radio(
        "Select a model to view its versions:",
        options=registered_models,
        label_visibility="collapsed"
    )

with col2:
    if selected_model_name:
        st.markdown(f"#### Versions for `{selected_model_name}`")
        versions = tracker.get_model_versions(selected_model_name)

        if not versions:
            st.warning("No versions found for this model.")
        else:
            # Display each version in an expander
            for version_info in versions:
                version_number = version_info.get("version", "N/A")
                with st.expander(f"**Version {version_number}**"):
                    display_version_details(tracker, version_info)
    else:
        st.info("Select a model from the list to see its details.")