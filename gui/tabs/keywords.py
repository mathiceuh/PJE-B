import streamlit as st
import pandas as pd
import json
import os
from algorithms.keywords.annotation import AnnotationParams, annotate_tweet


def highlight_text(text, model):
    """
    Reconstructs the tweet with :green[positive] and :red[negative] highlights.
    """
    words = text.split()
    highlighted_words = []

    for word in words:
        # Check logic (handle case sensitivity based on model params)
        check_word = word.lower() if model.lowercase else word

        if check_word in model.positive_words:
            highlighted_words.append(f":green[**{word}**]")  # Bold and Green
        elif check_word in model.negative_words:
            highlighted_words.append(f":red[**{word}**]")  # Bold and Red
        else:
            highlighted_words.append(word)

    return " ".join(highlighted_words)

def render(manager):
    st.header("🔑 Keyword-Based Algorithm")
    # 1. CONFIGURATION ("Hyperparameters")
    # ==============================================================================
    st.subheader("1. Configuration")

    # --- Load Default Logic (Hidden from UI) ---
    if 'default_keywords' not in st.session_state:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        json_path = os.path.join(project_root, "algorithms", "keywords", "keywords.json")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                st.session_state['default_keywords'] = json.load(f)
        except FileNotFoundError:
            st.session_state['default_keywords'] = {"positive": ["good", "great"], "negative": ["bad"]}

    # --- A. Dictionary Source (Side-by-Side Layout) ---
    col_upload, col_hint = st.columns([1, 1], gap="medium")

    with col_upload:
        st.markdown("##### 📤 Import Dictionary")
        st.caption("Upload a JSON file to override the default words.")
        uploaded_file = st.file_uploader(
            "Upload JSON",
            type=["json"],
            label_visibility="collapsed"  # Hides the label to save space
        )

    with col_hint:
        st.markdown("##### ℹ️ Expected JSON Format")
        # Directly show the code (No expander)
        st.code("""
    {
      "positive": ["good", "great"],
      "negative": ["bad", "terrible"]
    }
            """, language="json")

    # --- Logic: Load Data (Upload overrides Default) ---
    if uploaded_file is not None:
        try:
            uploaded_data = json.load(uploaded_file)
            if "positive" in uploaded_data and "negative" in uploaded_data:
                current_dict = uploaded_data
                st.toast("✅ Custom dictionary loaded!", icon="📂")
            else:
                st.error("Invalid Format! JSON must contain 'positive' and 'negative' keys.")
                current_dict = st.session_state['default_keywords']
        except Exception as e:
            st.error(f"Error reading file: {e}")
            current_dict = st.session_state['default_keywords']
    else:
        current_dict = st.session_state['default_keywords']

    st.markdown("---")
    # --- B. The Lists (Fixed-Height Tables) ---
    col_pos, col_neg, col_rules = st.columns([1, 1, 1])

    # Helper to config clean columns
    list_config = st.column_config.TextColumn(
        "Word",
        help="Add or remove words here",
        width="large",
        required=True
    )

    with col_pos:
        st.markdown("#### Positive Words")
        df_pos = pd.DataFrame({"word": current_dict.get("positive", [])})

        # KEY FIX: height=250 forces a scrollbar, keeping the UI small
        edited_pos_df = st.data_editor(
            df_pos,
            num_rows="dynamic",
            use_container_width=True,
            height=250,
            key="editor_pos",
            column_config={"word": list_config},
            hide_index=True  # Hides the 0,1,2 index numbers for a cleaner look
        )

    with col_neg:
        st.markdown("#### Negative Words")
        df_neg = pd.DataFrame({"word": current_dict.get("negative", [])})

        edited_neg_df = st.data_editor(
            df_neg,
            num_rows="dynamic",
            use_container_width=True,
            height=250,
            key="editor_neg",
            column_config={"word": list_config},
            hide_index=True
        )

    # --- C. Rules & Training ---
    with col_rules:
        st.markdown("#### Rules")
        st.caption("Normalization settings")
        use_lowercase = st.checkbox("Force Lowercase", value=True)
        use_stemming = st.checkbox("Use Stemming", value=False)

        st.markdown("<br>", unsafe_allow_html=True)  # Spacer

        # TRAIN ACTION
        if st.button("💾 Load & Train", type="primary", use_container_width=True):
            # 1. Extract lists back from the edited DataFrames
            # We filter out any empty rows just in case
            pos_list = [str(x).strip() for x in edited_pos_df["word"].tolist() if str(x).strip()]
            neg_list = [str(x).strip() for x in edited_neg_df["word"].tolist() if str(x).strip()]

            # 2. Create Params
            model = AnnotationParams(
                positive_words=pos_list,
                negative_words=neg_list,
                lowercase=use_lowercase,
                use_stemming=use_stemming
            )

            # 3. Save to Session State
            st.session_state['keyword_model'] = model
            st.success(f"Model Ready! \nVocab: {len(pos_list)} pos, {len(neg_list)} neg.")
    # ==============================================================================
    # 2. TEST ("Inference")
    # ==============================================================================
    st.divider()
    st.subheader("2. Test a Tweet")

    # Check if model exists
    if 'keyword_model' not in st.session_state:
        st.warning("⚠️ Please click 'Load & Train' above to initialize the algorithm.")
    else:
        # Input for single prediction
        user_tweet = st.text_input("Type a sentence to test the current configuration:",
                                   placeholder="Example: This app is absolutely amazing!")

        if user_tweet:
            # Retrieve the model
            model = st.session_state['keyword_model']

            # Run the Algorithm
            prediction_score = annotate_tweet(user_tweet, model)

            # Interpret Result (Mapping your 0/2/4 logic)
            if prediction_score == 4:
                result_text = "POSITIVE"
                color = "green"
            elif prediction_score == 0:
                result_text = "NEGATIVE"
                color = "red"
            else:
                result_text = "NEUTRAL"
                color = "gray"

            # Display Result
            st.markdown(f"### Prediction: :{color}[{result_text}]")
            st.caption(f"Algorithm Output Code: {prediction_score}")

            # --- DIAGNOSTIC ADDITION ---
            # This appears below your result to explain "Why"
            with st.expander("🔍 See which words triggered this result"):
                highlighted = highlight_text(user_tweet, model)
                st.markdown(f"> {highlighted}")

    # ==============================================================================
    # 3. BATCH EXECUTION (Apply to Dataset)
    # ==============================================================================
    st.divider()
    st.subheader("3. Apply to Dataset")
    st.caption("Run the configured rules on the Test dataset.")

    # 1. Checks
    if 'keyword_model' not in st.session_state:
        st.warning("⚠️ You must Train the model (Section 1) first.")
    elif 'test_df' not in st.session_state:
        st.warning("⚠️ No dataset found. Please go to 'Data Cleaning' first.")
    else:
        # 2. LABEL MAPPING SETTINGS
        # The user tells us what the algorithm's numeric codes should look like in the final CSV
        with st.expander("⚙️ Output Label Formatting", expanded=True):
            st.write("Map the algorithm's internal codes (0, 2, 4) to your preferred output format:")

            c_neg, c_neu, c_pos = st.columns(3)

            # Defaults are "Negative", "Neutral", "Positive"
            # But user can change them to -1, 0, 1 or "BAD", "OK", "GOOD"
            with c_neg:
                map_0 = st.text_input("Negative (0) →", value="Negative")
            with c_neu:
                map_2 = st.text_input("Neutral (2) →", value="Neutral")
            with c_pos:
                map_4 = st.text_input("Positive (4) →", value="Positive")

        # 3. RUN BUTTON
        if st.button("🚀 Run Batch Annotation", type="primary", use_container_width=True):

            # Prepare Data
            df_batch = st.session_state['test_df'].copy()
            model = st.session_state['keyword_model']

            # Get Text Column
            text_col_idx = st.session_state['cleaned_text_col_idx']
            text_col = df_batch.columns[text_col_idx]

            # Run Logic (with progress bar)
            progress_bar = st.progress(0, text="Annotating rows...")

            # A. Calculate Raw Scores (0, 2, 4)
            # We use a lambda to apply the function row-by-row
            raw_scores = df_batch[text_col].astype(str).apply(lambda x: annotate_tweet(x, model))

            progress_bar.progress(50, text="Formatting labels...")

            # B. Apply User Mapping
            def map_score(score):
                if score == 4:
                    return map_4
                elif score == 0:
                    return map_0
                return map_2

            # Create the new column 'predicted_label'
            df_batch['predicted_label'] = raw_scores.apply(map_score)

            progress_bar.progress(100, text="Done!")

            # 4. DISPLAY RESULTS (No Evaluation)
            st.success(f"Processed {len(df_batch)} rows successfully!")

            st.markdown("### Preview Result")
            # Show Text + Predicted Label
            st.dataframe(
                df_batch[[text_col, 'predicted_label']].head(100),
                use_container_width=True
            )

            # 5. EXPORT
            csv_data = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Annotated CSV",
                data=csv_data,
                file_name="keyword_annotated_results.csv",
                mime="text/csv",
                use_container_width=True
            )