import streamlit as st
import sys
import os
st.cache_data.clear()
st.cache_resource.clear()
# 1. Setup System Paths
# This ensures Python can find your 'core' and 'algorithms' modules
# regardless of how the script is executed.
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'algorithms'))

# 2. Import the GUI Layout
from gui.layout import run_app

# 3. Application Entry Point
if __name__ == "__main__":
    # Global page configuration
    st.set_page_config(
        page_title="PJE - Algorithm & Dataset Abstraction",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Launch the App
    run_app()