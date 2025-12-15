import streamlit as st

def render(manager):
    st.header("✍️ Annotation Studio (Human Baseline)")
    st.info("Goal: Manually label tweets to create a 'Ground Truth' for evaluation.")

    st.markdown("""
    ### 🚧 TODO:
    1. **Input:** Load tweets from the 'Test Set' (or uploaded unlabeled file).
    2. **Interface:** specific view to show **one tweet at a time**.
    3. **Controls:** Add buttons for Pos/Neut/Neg (or 0/2/4).
    4. **Progress:** Add a progress bar (e.g., '15/100 annotated').
    5. **Output:** Save results to `manual_annotations.csv`.
    6. **Dev Tool:** Add 'Random Fill' button for testing.
    """)
    st.write("---")
    st.write("*(Build the annotation interface here)*")