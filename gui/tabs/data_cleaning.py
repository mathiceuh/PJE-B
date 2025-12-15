import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import your backend logic
from cleaning.column_detection.tweet_column_detector import FinalTweetDetector
from cleaning.column_detection.label_column_detector import FinalLabelDetector
from cleaning.tweet_cleaning.tweet_cleaner import TweetCleaner
from cleaning.tweet_cleaning import rules


def render(manager):
    st.header("🛠️ Data Import & Cleaning")

    # ==============================================================================
    # 1. DATA IMPORT
    # ==============================================================================
    st.subheader("1. Import Data")
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

    if uploaded_file:
        # Load data into session state if not already present or if file changed
        if 'raw_df' not in st.session_state or st.session_state.get('uploaded_filename') != uploaded_file.name:
            try:
                # Try reading with default utf-8
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                # Fallback for excel encoded files
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin-1')

            st.session_state['raw_df'] = df
            st.session_state['uploaded_filename'] = uploaded_file.name
            st.session_state['cleaning_done'] = False  # Reset cleaning status on new file
            st.success(f"Loaded {uploaded_file.name} with {df.shape[0]} rows and {df.shape[1]} columns.")

    # Stop here if no data
    if 'raw_df' not in st.session_state:
        st.info("Please upload a CSV file to proceed.")
        return

    df = st.session_state['raw_df']

    # ==============================================================================
    # 2. COLUMN MAPPING (AUTO-DETECTION)
    # ==============================================================================
    st.markdown("---")
    st.subheader("2. Column Mapping")

    col1, col2 = st.columns(2)

    # --- Text Column Detection ---
    # We run detection once and store it to avoid re-running on every click
    if 'detected_text_idx' not in st.session_state:
        detector = FinalTweetDetector()
        detected_idx = detector.detect(df)
        st.session_state['detected_text_idx'] = detected_idx if detected_idx is not None else 0

    with col1:
        text_col_name = st.selectbox(
            "Select Text Column (Feature)",
            options=df.columns,
            index=st.session_state['detected_text_idx'],
            help="The column containing the tweets or text to analyze."
        )

    # --- Label Column Detection ---
    if 'detected_label_idx' not in st.session_state:
        detector = FinalLabelDetector()
        detected_idx = detector.detect(df)
        # If None found, we default to -1 (Special case for 'No Label')
        st.session_state['detected_label_idx'] = detected_idx

    with col2:
        # Add "Unsupervised" option to the list of columns
        label_options = ["⛔ None (Unsupervised)"] + list(df.columns)

        # Calculate default index for selectbox
        default_label_index = 0  # Default to "None"
        if st.session_state['detected_label_idx'] is not None:
            # Shift index by 1 because of the "None" option added at start
            default_label_index = st.session_state['detected_label_idx'] + 1

        label_selection = st.selectbox(
            "Select Label Column (Target)",
            options=label_options,
            index=default_label_index,
            help="The column containing sentiments or categories. Choose 'None' if you only have text."
        )

    # Determine actual column names/indices based on selection
    text_col_idx = df.columns.get_loc(text_col_name)

    if label_selection == "⛔ None (Unsupervised)":
        label_col_idx = None
    else:
        label_col_idx = df.columns.get_loc(label_selection)

    # ==============================================================================
    # 3. CLEANING CONFIGURATION
    # ==============================================================================
    st.markdown("---")
    st.subheader("3. Cleaning Rules")

    col_rules, col_preview = st.columns([1, 2])

    with col_rules:
        st.markdown("**Select rules to apply:**")

        # UI Toggles for Rules
        use_lowercase = st.checkbox("Convert to Lowercase", value=True)
        use_remove_urls = st.checkbox("Remove URLs", value=True)
        use_remove_mentions = st.checkbox("Remove Mentions (@user)", value=True)
        use_remove_hashtags = st.checkbox("Remove Hashtags (#tag)", value=False)
        use_remove_rt = st.checkbox("Remove 'RT' markers", value=True)
        use_remove_punct = st.checkbox("Remove Punctuation", value=True)
        use_normalize = st.checkbox("Normalize Whitespace", value=True)
        use_emojis = st.checkbox("Remove Emojis", value=False)

    # Construct the list of rule objects based on UI
    active_rules = []
    if use_lowercase: active_rules.append(rules.ToLowercase())
    if use_remove_urls: active_rules.append(rules.RemoveURLs())
    if use_remove_mentions: active_rules.append(rules.RemoveMentions())
    if use_remove_hashtags: active_rules.append(rules.RemoveHashtags())
    if use_remove_rt: active_rules.append(rules.RemoveRetweetMarker())
    if use_remove_punct: active_rules.append(rules.RemovePunctuation())
    if use_emojis: active_rules.append(rules.RemoveEmojis())
    if use_normalize: active_rules.append(rules.NormalizeWhitespace())

    # --- LIVE PREVIEW ---
    with col_preview:
        st.markdown("**🔍 Live Preview (First 5 rows)**")

        # Create a small subset for preview
        preview_df = df.iloc[:5].copy()

        # Initialize a temp cleaner just for preview
        preview_cleaner = TweetCleaner(text_rules=active_rules, row_filters=[], dataset_filters=[])

        # We manually apply text rules here for the preview (skipping row/dataset filters for speed)
        cleaned_preview = preview_cleaner._apply_text_rules(preview_df, text_col_idx)

        # Prepare comparison dataframe
        comparison = pd.DataFrame({
            "Original": preview_df.iloc[:, text_col_idx],
            "Cleaned": cleaned_preview.iloc[:, text_col_idx]
        })
        st.dataframe(comparison, use_container_width=True)

    # ==============================================================================
    # 4. EXECUTE CLEANING
    # ==============================================================================
    st.markdown("---")

    # Advanced Options (Row Filters)
    with st.expander("Advanced Row Filters"):
        filter_emoji = st.checkbox("Remove rows with mixed emojis (Pos & Neg)", value=False)
        filter_dupes = st.checkbox("Remove Duplicate Tweets", value=True)
        filter_lang = st.checkbox("Keep only dominant language (slow)", value=False)

    if st.button("🚀 Clean Dataset", type="primary"):
        with st.spinner("Cleaning data... This might take a moment."):
            # Prepare Filters
            row_filters = []
            dataset_filters = []

            from cleaning.tweet_cleaning.tweet_cleaner import RemoveMixedEmojiRow, RemoveDuplicates, \
                RemoveDifferentLanguageTweets

            if filter_emoji: row_filters.append(RemoveMixedEmojiRow())
            if filter_dupes: dataset_filters.append(RemoveDuplicates(subset="tweet"))
            if filter_lang: dataset_filters.append(RemoveDifferentLanguageTweets())

            # Instantiate Main Cleaner
            cleaner = TweetCleaner(
                text_rules=active_rules,
                row_filters=row_filters,
                dataset_filters=dataset_filters
            )

            # Run Cleaning
            cleaned_df = cleaner.clean_dataframe(df, tweet_idx=text_col_idx, label_idx=label_col_idx)

            # Save to Session State
            st.session_state['cleaned_df'] = cleaned_df
            st.session_state['cleaned_text_col_idx'] = text_col_idx
            st.session_state['cleaned_label_col_idx'] = label_col_idx
            st.session_state['cleaning_done'] = True

            st.success(f"Cleaning Complete! Rows: {len(df)} ➝ {len(cleaned_df)}")

    # ==============================================================================
    # 5. SPLIT TRAIN / TEST
    # ==============================================================================
    if st.session_state.get('cleaning_done', False):
        st.markdown("---")
        st.subheader("4. Train / Test Split")

        c_split1, c_split2, c_split3 = st.columns([2, 1, 1])

        with c_split1:
            test_size = st.slider("Test Set Size (%)", min_value=10, max_value=50, value=20, step=5)

        with c_split2:
            do_shuffle = st.checkbox("Shuffle Data", value=True)
            stratify_opt = st.checkbox("Stratify (keep class balance)", value=True, disabled=(label_col_idx is None))

        with c_split3:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacer
            split_btn = st.button("✂️ Split Data")

        if split_btn:
            clean_df = st.session_state['cleaned_df']

            # Prepare arguments for train_test_split
            split_args = {
                "test_size": test_size / 100.0,
                "shuffle": do_shuffle,
                "random_state": 42
            }

            # Handle Stratification (only if label exists and not None)
            if stratify_opt and label_col_idx is not None:
                # We need to pass the label column for stratification
                # But wait, our 'clean_df' might have dropped rows, so we use the column by index
                split_args["stratify"] = clean_df.iloc[:, label_col_idx]

            try:
                train_df, test_df = train_test_split(clean_df, **split_args)

                # Store in session state
                st.session_state['train_df'] = train_df
                st.session_state['test_df'] = test_df

                # Show Stats
                st.success("Data successfully split!")
                col_res1, col_res2 = st.columns(2)
                col_res1.metric("Training Set", f"{len(train_df)} rows")
                col_res2.metric("Test Set", f"{len(test_df)} rows")

                with st.expander("View Split Data", expanded=True):
                    # --- DISPLAY LOGIC START ---
                    st.markdown("**Training Sample (First 5 rows)**")
                    # 1. Create a copy so we don't touch the real data
                    train_display = train_df.head().copy()
                    # 2. Overwrite columns with simple indices (0, 1, 2...)
                    train_display.columns = range(len(train_display.columns))
                    # 3. Show the display copy
                    st.dataframe(train_display, use_container_width=True)

                    st.markdown("**Test Sample (First 5 rows)**")
                    test_display = test_df.head().copy()
                    test_display.columns = range(len(test_display.columns))
                    st.dataframe(test_display, use_container_width=True)
                    # --- DISPLAY LOGIC END ---

            except ValueError as e:
                st.error(
                    f"Split failed. Usually happens if a class has too few examples for stratification. Try unchecking 'Stratify'. Error: {e}")