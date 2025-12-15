import streamlit as st
import pandas as pd
import json
import os
import time

# --- Import Backend Logic ---
from algorithms.knn.knn_classifier import KNNClassifier
from algorithms.knn.distance import (
    JaccardDistance,
    CosineDistance,
    LevenshteinDistance,
    EuclideanDistance,
    ManhattanDistance
)
from algorithms.knn.voting import MajorityVote, WeightedVote
from algorithms.knn.synonym_mapper import SynonymMapper

# Map friendly names to backend classes
DISTANCE_MAP = {
    "Jaccard (Set Overlap)": JaccardDistance,
    "Cosine (Vector Angle)": CosineDistance,
    "Euclidean (L2 Norm)": EuclideanDistance,
    "Manhattan (L1 Norm)": ManhattanDistance,
    "Levenshtein (Edit Distance)": LevenshteinDistance
}

VOTING_MAP = {
    "Majority Vote": MajorityVote,
    "Weighted Vote (Inverse Distance)": WeightedVote
}


def render(manager):
    st.header("🤖 K-Nearest Neighbors (KNN)")
    st.caption("Classify tweets based on their similarity to labeled examples.")

    # ==============================================================================
    # 1. CONFIGURATION
    # ==============================================================================
    st.subheader("1. Configuration")

    # --- A. Synonyms Loader (Default vs Custom) ---
    if 'knn_synonyms' not in st.session_state:
        # Load default from file
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            json_path = os.path.join(project_root, "algorithms", "knn", "synonyms.json")

            with open(json_path, "r", encoding="utf-8") as f:
                st.session_state['knn_synonyms'] = json.load(f)
        except Exception:
            # Fallback if file missing
            st.session_state['knn_synonyms'] = {"0": ["love", "like"], "1": ["hate", "dislike"]}

    with st.expander("📚 Synonym Dictionary Configuration", expanded=False):
        c1, c2 = st.columns([1, 1])
        with c1:
            uploaded_syn = st.file_uploader("Upload Synonyms JSON", type=["json"])
        with c2:
            st.info("Format: Key = ID, Value = List of words. First word is canonical.")

        if uploaded_syn:
            try:
                st.session_state['knn_synonyms'] = json.load(uploaded_syn)
                st.toast("Synonyms loaded!", icon="✅")
            except:
                st.error("Invalid JSON")

        # Edit Synonyms Table
        # Convert dict to list of dicts for data_editor
        syn_data = []
        for gid, words in st.session_state['knn_synonyms'].items():
            syn_data.append({"ID": gid, "Words": ", ".join(words)})

        df_syn = pd.DataFrame(syn_data)
        edited_syn = st.data_editor(
            df_syn,
            num_rows="dynamic",
            width="stretch",
            key="knn_syn_editor",
            column_config={
                "Words": st.column_config.TextColumn(help="Comma-separated synonyms. First one is the replacement.")
            }
        )

    st.markdown("---")

    # --- B. Hyperparameters ---
    col_k, col_dist, col_vote = st.columns(3)

    with col_k:
        k_val = st.slider("K (Number of Neighbors)", 1, 21, 3, step=2, help="Odd numbers are better to avoid ties.")

    with col_dist:
        dist_name = st.selectbox("Distance Metric", list(DISTANCE_MAP.keys()), index=0)
        use_synonyms = st.checkbox("Use Synonym Normalization", value=True)

    with col_vote:
        vote_name = st.selectbox("Voting Strategy", list(VOTING_MAP.keys()), index=0)

    # --- C. Train Button ---
    st.markdown("<br>", unsafe_allow_html=True)

    # Validation check
    if 'train_df' not in st.session_state:
        st.warning("⚠️ No Training Data found. Please go to 'Data Cleaning' and split your data.")
        train_ready = False
    else:
        train_ready = True
        train_size = len(st.session_state['train_df'])
        st.caption(f"Ready to train on **{train_size}** rows.")

    if st.button("💾 Train KNN Model", type="primary", disabled=not train_ready, use_container_width=True):
        with st.spinner("Training (Indexing data)..."):
            # 1. Parse Synonyms from Editor
            final_synonyms = {}
            for index, row in edited_syn.iterrows():
                # clean and split
                w_list = [w.strip() for w in str(row["Words"]).split(",") if w.strip()]
                if w_list:
                    final_synonyms[str(row["ID"])] = w_list

            # Save temporary json for the backend class (it expects a file path or we modify it to accept dict)
            # Since your class takes a path, let's dump a temp file or modify the class.
            # *Better approach:* We will mock the synonym behavior or dump a temp file.
            # Let's dump to a temp path for compatibility with your existing code.
            temp_syn_path = "temp_synonyms.json"
            with open(temp_syn_path, "w", encoding="utf-8") as f:
                json.dump(final_synonyms, f)

            # 2. Initialize Distance Class
            DistClass = DISTANCE_MAP[dist_name]
            # Instantiate with synonym logic
            distance_instance = DistClass(use_synonyms=use_synonyms,
                                          synonym_json=temp_syn_path if use_synonyms else None)

            # 3. Initialize Voter
            VoteClass = VOTING_MAP[vote_name]
            voter_instance = VoteClass()

            # 4. Initialize KNN
            knn = KNNClassifier(k=k_val, distance=distance_instance, voter=voter_instance)

            # 5. Fit Data
            # Format: list of (label, tweet) tuples
            train_df = st.session_state['train_df']

            # Identify columns
            txt_col = train_df.columns[st.session_state['cleaned_text_col_idx']]
            lbl_col = train_df.columns[st.session_state['cleaned_label_col_idx']]

            # Convert to list of tuples
            training_data = list(zip(train_df[lbl_col], train_df[txt_col]))
            knn.fit(training_data)

            # 6. Save to Session
            st.session_state['knn_model'] = knn
            st.success(f"KNN Trained successfully! (K={k_val}, Dist={dist_name})")

    # ==============================================================================
    # 2. TEST ("Inference")
    # ==============================================================================
    st.divider()
    st.subheader("2. Test a Tweet")

    if 'knn_model' not in st.session_state:
        st.info("Please train the model above.")
    else:
        user_tweet = st.text_input("Type a tweet to classify:", placeholder="I love this movie so much!")

        if user_tweet:
            knn = st.session_state['knn_model']

            # We want to see the neighbors, so we need to peek inside predict_one logic
            # OR we modify predict_one to return neighbors.
            # Since we can't easily modify backend files here, we will replicate the logic for visualization.

            # 1. Run Prediction
            start_time = time.time()
            prediction = knn.predict_one(user_tweet)
            elapsed = time.time() - start_time

            st.markdown(f"### Prediction: **{prediction}**")
            st.caption(f"Inference time: {elapsed:.4f}s")

            # 2. Explain (Find neighbors again to show them)
            with st.expander("🔍 See K-Nearest Neighbors (Explanation)", expanded=True):
                # Calculate distances manually to display
                neighbors = []
                for label, tweet in knn.base:
                    d = knn.distance.compute(user_tweet, tweet)
                    neighbors.append({"Distance": d, "Label": label, "Tweet": tweet})

                # Sort and take K
                neighbors = sorted(neighbors, key=lambda x: x["Distance"])[:knn.k]

                # Display
                st.table(pd.DataFrame(neighbors))

    # ==============================================================================
    # 3. BATCH EXECUTION
    # ==============================================================================
    st.divider()
    st.subheader("3. Apply to Dataset")

    if 'knn_model' not in st.session_state:
        st.warning("⚠️ Train the model first.")
    elif 'test_df' not in st.session_state:
        st.warning("⚠️ No Test Data.")
    else:
        # User Output Mapping
        with st.expander("⚙️ Output Label Formatting"):
            st.write("Current labels in training data are likely: 0, 2, 4. Map them here if needed.")
            c_neg, c_neu, c_pos = st.columns(3)
            map_0 = st.text_input("Map 0 to:", value="Negative")
            map_2 = st.text_input("Map 2 to:", value="Neutral")
            map_4 = st.text_input("Map 4 to:", value="Positive")

        if st.button("🚀 Run Batch KNN", type="primary", use_container_width=True):
            test_df = st.session_state['test_df'].copy()
            # Safety check: KNN on 300k rows is SLOW. Warn user or sample.
            if len(test_df) > 2000:
                st.warning("⚠️ KNN is slow on large datasets. Processing first 2000 rows only for demo.")
                test_df = test_df.head(2000)

            knn = st.session_state['knn_model']
            txt_col = test_df.columns[st.session_state['cleaned_text_col_idx']]

            # Progress bar
            bar = st.progress(0, "Classifying...")

            predictions = []
            total = len(test_df)

            # Loop with progress
            for i, tweet in enumerate(test_df[txt_col]):
                pred = knn.predict_one(str(tweet))

                # Apply Mapping
                if pred == 0 or str(pred) == "0":
                    pred_mapped = map_0
                elif pred == 4 or str(pred) == "4":
                    pred_mapped = map_4
                else:
                    pred_mapped = map_2

                predictions.append(pred_mapped)

                if i % 50 == 0:
                    bar.progress(min(i / total, 1.0))

            bar.progress(100, "Done!")
            test_df['knn_prediction'] = predictions

            st.success(f"Processed {len(test_df)} rows.")
            st.dataframe(test_df[[txt_col, 'knn_prediction']].head(50), width="stretch")

            # Download
            csv = test_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download KNN Results", csv, "knn_results.csv", "text/csv")