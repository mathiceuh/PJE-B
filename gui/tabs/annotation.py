import streamlit as st
import pandas as pd
import random


def render(manager):
    st.header("🏷️ Human Annotation (Ground Truth)")
    st.caption("Create a 'Gold Standard' dataset by manually labeling a random sample.")

    # ==============================================================================
    # 1. SETUP & SAMPLING
    # ==============================================================================

    # Check dependencies
    if 'test_df' not in st.session_state:
        st.warning("⚠️ No Test Data found. Please go to the 'Data Cleaning' tab first.")
        return

    # Initialize State Variables
    if 'annotation_subset' not in st.session_state:
        st.session_state['annotation_subset'] = None  # The dataframe to label
        st.session_state['current_index'] = 0  # Which row are we on?
        st.session_state['human_labels'] = []  # List to store results

    # --- A. Start Session Button ---
    if st.session_state['annotation_subset'] is None:
        st.subheader("1. Start Session")

        full_df = st.session_state['test_df']
        total_rows = len(full_df)

        c1, c2 = st.columns([2, 1])
        with c1:
            st.info(f"Available Data: **{total_rows}** rows (Test Set)")
            sample_size = st.number_input("Sample Size", min_value=5, max_value=1000, value=100)

        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🎲 Generate Random Sample", type="primary"):
                # Logic: Grab random sample
                subset = full_df.sample(n=min(sample_size, total_rows), random_state=42).reset_index(drop=True)

                # Reset State
                st.session_state['annotation_subset'] = subset
                st.session_state['current_index'] = 0
                st.session_state['human_labels'] = [None] * len(subset)  # Empty placeholders
                st.rerun()
        return

    # ==============================================================================
    # 2. ANNOTATION STATION
    # ==============================================================================

    subset = st.session_state['annotation_subset']
    curr_idx = st.session_state['current_index']
    total = len(subset)

    # Get Column Names
    text_col_idx = st.session_state['cleaned_text_col_idx']
    text_col = subset.columns[text_col_idx]

    # --- Progress Bar ---
    progress = curr_idx / total
    st.progress(progress, text=f"Progress: {curr_idx} / {total}")

    # --- Check if Done ---
    if curr_idx >= total:
        st.success("🎉 Annotation Complete!")

        # Save results to the subset dataframe
        subset['human_label'] = st.session_state['human_labels']

        # Show Preview (Fixed Console Warning)
        st.dataframe(subset[[text_col, 'human_label']].head(50), width="stretch")

        # Download
        # FIX: header=False prevents the first tweet (which is the column name) from being printed as the first line.
        csv = subset.to_csv(index=False, header=False).encode('utf-8')

        st.download_button(
            "⬇️ Download Labeled Dataset",
            csv,
            "human_ground_truth.csv",
            "text/csv",
            type="primary"
        )

        if st.button("🔄 Reset & Start Over"):
            del st.session_state['annotation_subset']
            st.rerun()
        return  # STOPS HERE if done

    # ==============================================================================
    # THE INTERFACE (Only runs if NOT done)
    # ==============================================================================

    current_row = subset.iloc[curr_idx]
    tweet_text = str(current_row[text_col])

    # --- Display Tweet (Card Style) ---
    st.markdown(
        f"""
        <div style="
            padding: 30px; 
            background-color: #ffffff; 
            color: #111111; 
            border-radius: 12px; 
            font-size: 1.3rem; 
            line-height: 1.6;
            font-family: sans-serif;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border: 1px solid #e0e0e0;
            margin-bottom: 25px;
            height: auto;
        ">
            {tweet_text}
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Action Buttons ---
    def submit_label(label_val):
        st.session_state['human_labels'][curr_idx] = label_val
        st.session_state['current_index'] += 1

    # Centered Layout: using empty columns on sides to squeeze buttons to center
    _, c_neg, c_neu, c_pos, c_skip, _ = st.columns([0.5, 2, 2, 2, 1, 0.5], gap="small")

    with c_neg:
        st.button("🔴 Negative", on_click=submit_label, args=(0,), use_container_width=True)
    with c_neu:
        st.button("⚪ Neutral", on_click=submit_label, args=(2,), use_container_width=True)
    with c_pos:
        st.button("🟢 Positive", on_click=submit_label, args=(4,), use_container_width=True)
    with c_skip:
        st.button("⏭️", help="Skip", on_click=submit_label, args=(None,), use_container_width=True)

    st.caption(f"Reviewing tweet {curr_idx + 1} of {total}")

    # ==============================================================================
    # 3. DEBUG / HEURISTICS
    # ==============================================================================
    st.divider()
    with st.expander("🛠️ Debug / Quick Fill"):
        st.caption("Automatically fill remaining rows to test export.")

        c_fill1, c_fill2 = st.columns(2)

        with c_fill1:
            if st.button("🤖 Randomly Label Remaining"):
                for i in range(curr_idx, total):
                    st.session_state['human_labels'][i] = random.choice([0, 2, 4])
                st.session_state['current_index'] = total
                st.rerun()

        with c_fill2:
            if st.button("🟢 Label All 'Positive'"):
                for i in range(curr_idx, total):
                    st.session_state['human_labels'][i] = 4
                st.session_state['current_index'] = total
                st.rerun()