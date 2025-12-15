import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.feature_extraction.text import TfidfVectorizer

# --- IMPORT YOUR MODULES ---
from core.algorithm_wrapper import (
    KNNWrapper, KeywordWrapper, NaiveBayesWrapper,
    DummyWrapper, DummyAlgo2
)
from core.algorithm_manager import AlgorithmManager

# Import your Data/Cleaning modules
from cleaning.tweet_cleaning.tweet_cleaner import TweetCleaner
from cleaning.column_detection.tweet_column_detector import FinalTweetDetector
from cleaning.column_detection.label_column_detector import FinalLabelDetector

# ADD THESE IMPORTS for the rules
from cleaning.tweet_cleaning.rules import (
    ToLowercase, RemoveURLs, RemoveMentions, RemoveHashtags,
    RemoveRetweetMarker, RemovePunctuation, NormalizeWhitespace, RemoveEmojis
)
from cleaning.tweet_cleaning.tweet_cleaner import (
    RemoveDifferentLanguageTweets, RemoveDuplicates, RemoveMixedEmojiRow
)

# =========================================================
# 🛠️ HELPER: CLUSTERING VISUALIZATION
# =========================================================
def plot_dendrogram(texts, method='ward'):
    """
    Generates a dendrogram for the Unsupervised tab.
    """
    # 1. Vectorize (TF-IDF)
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X = vectorizer.fit_transform(texts).toarray()

    # 2. Linkage
    Z = linkage(X, method=method)

    # 3. Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, ax=ax, truncate_mode='lastp', p=20)
    plt.title(f"Hierarchical Clustering Dendrogram ({method})")
    plt.xlabel("Cluster Size / Sample Index")
    plt.ylabel("Distance")
    return fig


# =========================================================
# 🎨 MAIN STREAMLIT APP
# =========================================================
def main():
    st.set_page_config(page_title="PJE Sentiment Analysis", layout="wide", page_icon="🐦")

    # --- SESSION STATE SETUP ---
    if 'manager' not in st.session_state:
        st.session_state['manager'] = AlgorithmManager([
            ("KNN", KNNWrapper()),
            ("Naive Bayes", NaiveBayesWrapper()),
            ("Keyword Rule-Based", KeywordWrapper()),
            ("Dummy Random", DummyWrapper()),
            ("Dummy Fixed (4)", DummyAlgo2()),
        ])

    if 'raw_df' not in st.session_state: st.session_state['raw_df'] = None
    if 'clean_df' not in st.session_state: st.session_state['clean_df'] = None
    if 'train_set' not in st.session_state: st.session_state['train_set'] = None
    if 'test_set' not in st.session_state: st.session_state['test_set'] = None
    if 'col_map' not in st.session_state: st.session_state['col_map'] = {'text': None, 'label': None}
    if 'human_annotations' not in st.session_state: st.session_state['human_annotations'] = []

    # --- SIDEBAR ---
    st.sidebar.title("🐦 PJE Workflow")
    tabs = ["1. Data & Cleaning", "2. Annotation Studio", "3. Algorithm Studio", "4. Evaluation"]
    page = st.sidebar.radio("Navigate", tabs)

    # =====================================================
    # 📂 TAB 1: DATA & CLEANING
    # =====================================================
    if page == "1. Data & Cleaning":
        st.header("📂 Data Ingestion & Preprocessing")

        uploaded_file = st.file_uploader("Upload Tweet CSV", type=["csv"])

        if uploaded_file:
            # 1. Load Data
            try:
                if st.session_state['raw_df'] is None:
                    df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
                    st.session_state['raw_df'] = df
                    st.success(f"Loaded {len(df)} rows.")
                else:
                    df = st.session_state['raw_df']
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                return

            st.divider()

            # 2. Column Mapping
            st.subheader("🕵️ Column Mapping")
            text_detector = FinalTweetDetector()
            label_detector = FinalLabelDetector()

            suggested_text_idx = text_detector.detect(df)
            suggested_label_idx = label_detector.detect(df)

            c1, c2 = st.columns(2)
            with c1:
                options = list(df.columns)
                default_text = options[suggested_text_idx] if suggested_text_idx is not None else options[0]
                text_col_name = st.selectbox("Text Column", options, index=options.index(default_text))

            with c2:
                label_options = ["(None / Unsupervised)"] + options
                default_label_idx = suggested_label_idx + 1 if suggested_label_idx is not None else 0
                label_col_name = st.selectbox("Label Column", label_options, index=default_label_idx)

            actual_label = None if label_col_name == "(None / Unsupervised)" else label_col_name
            st.session_state['col_map'] = {'text': text_col_name, 'label': actual_label}

            # --- Cleaning Configuration ---
            st.divider()
            st.subheader("🧹 Cleaning Configuration")

            c_conf1, c_conf2 = st.columns(2)

            with c_conf1:
                st.markdown("#### ⚡ Fast Rules (Regex)")
                rule_map = {
                    "Lowercase": ToLowercase(),
                    "Remove URLs": RemoveURLs(),
                    "Remove Mentions (@)": RemoveMentions(),
                    "Remove RT Marker": RemoveRetweetMarker(),
                    "Remove Punctuation": RemovePunctuation(),
                    "Normalize Whitespace": NormalizeWhitespace(),
                    "Remove Emojis": RemoveEmojis(),
                    "Remove Hashtags": RemoveHashtags(keep_word=False)
                }
                selected_rule_names = st.multiselect(
                    "Select Text Normalization Rules",
                    options=list(rule_map.keys()),
                    default=["Lowercase", "Remove URLs", "Remove Mentions (@)", "Remove RT Marker", "Normalize Whitespace"]
                )

            with c_conf2:
                st.markdown("#### 🐢 Slow / Heavy Filters")
                use_lang = st.checkbox("🌍 Remove Non-English (Very Slow)", value=False,
                                       help="Uses langdetect. Can take minutes for >5k rows.")
                use_dupes = st.checkbox("♻️ Remove Duplicates", value=True)
                use_mixed_emoji = st.checkbox("🎭 Remove Mixed Emojis (Pos+Neg)", value=True)

            # --- Execution ---
            if st.button("Run Custom Cleaning"):
                # 1. Build Rules
                active_text_rules = [rule_map[name] for name in selected_rule_names]

                active_row_filters = []
                if use_mixed_emoji: active_row_filters.append(RemoveMixedEmojiRow())
                if use_lang: active_row_filters.append(RemoveDifferentLanguageTweets())

                active_dataset_filters = []
                if use_dupes: active_dataset_filters.append(RemoveDuplicates())

                # 2. Instantiate Cleaner
                cleaner = TweetCleaner(
                    text_rules=active_text_rules,
                    row_filters=active_row_filters,
                    dataset_filters=active_dataset_filters
                )

                # 3. Get Data from Session State (Fixes Unresolved Reference)
                target_df = st.session_state['raw_df']
                txt_col = st.session_state['col_map']['text']
                lbl_col = st.session_state['col_map']['label']

                t_idx = target_df.columns.get_loc(txt_col)
                l_idx = target_df.columns.get_loc(lbl_col) if lbl_col else None

                with st.spinner("Applying custom cleaning pipeline..."):
                    cleaned = cleaner.clean_dataframe(target_df, tweet_idx=t_idx, label_idx=l_idx)

                st.session_state['clean_df'] = cleaned
                st.success(f"Cleaning Done! {len(cleaned)} rows remaining.")
                st.dataframe(cleaned.head())

            # --- Splitting ---
            if st.session_state['clean_df'] is not None:
                st.divider()
                st.subheader("✂️ Train / Test Split")
                split_ratio = st.slider("Test Set Size", 0.1, 0.5, 0.2)

                if st.button("Create Split"):
                    data = st.session_state['clean_df']
                    train, test = train_test_split(data, test_size=split_ratio, random_state=42)
                    st.session_state['train_set'] = train
                    st.session_state['test_set'] = test
                    st.success(f"Split created: {len(train)} Training / {len(test)} Test")

    # =====================================================
    # ✍️ TAB 2: ANNOTATION STUDIO
    # =====================================================
    elif page == "2. Annotation Studio":
        st.header("✍️ Manual Annotation (Gold Standard)")

        if st.session_state['test_set'] is None:
            st.warning("Please split the data in Tab 1 first.")
            return

        test_df = st.session_state['test_set']
        text_col = st.session_state['col_map']['text']
        annotated_count = len(st.session_state['human_annotations'])
        target_count = 100

        st.progress(min(annotated_count / target_count, 1.0))
        st.caption(f"Progress: {annotated_count} / {target_count} tweets annotated")

        if annotated_count < len(test_df):
            current_row = test_df.iloc[annotated_count]
            tweet_text = current_row[text_col]

            st.markdown(f"### Tweet:")
            st.info(tweet_text)

            st.markdown("#### Assign Label:")
            c1, c2, c3 = st.columns(3)

            def save_label(lbl):
                st.session_state['human_annotations'].append({
                    'text': tweet_text,
                    'label': lbl,
                    'original_row': current_row.to_dict()
                })
                st.rerun()

            with c1:
                if st.button("😡 Negative (0)", use_container_width=True): save_label(0)
            with c2:
                if st.button("😐 Neutral (2)", use_container_width=True): save_label(2)
            with c3:
                if st.button("😃 Positive (4)", use_container_width=True): save_label(4)
        else:
            st.success("🎉 You have annotated all available tweets!")

        if annotated_count > 0:
            with st.expander("View Annotations"):
                st.dataframe(pd.DataFrame(st.session_state['human_annotations']))

    # =====================================================
    # 🧪 TAB 3: ALGORITHM STUDIO (UNIFIED)
    # =====================================================
    elif page == "3. Algorithm Studio":
        st.header("⚙️ Model Configuration & Execution")
        manager = st.session_state['manager']

        algo_name = st.selectbox("Select Strategy", manager.get_available_algos() + ["Hierarchical Clustering (Unsupervised)"])

        if algo_name == "Hierarchical Clustering (Unsupervised)":
            MODE = "unsupervised"
        else:
            manager.select(algo_name)
            current_algo = manager.get_current()
            MODE = current_algo.mode

        st.divider()

        if MODE == "supervised":
            st.subheader(f"🟢 Supervised: {algo_name}")
            st.write("**Hyperparameters:**")
            params = {}
            if hasattr(current_algo, 'param_schema'):
                cols = st.columns(3)
                i = 0
                for pname, config in current_algo.param_schema.items():
                    with cols[i % 3]:
                        if config['type'] == 'int':
                            params[pname] = st.slider(pname, config['min'], config['max'], config['default'])
                        elif config['type'] == 'float':
                            params[pname] = st.number_input(pname, value=config['default'], step=config.get('step', 0.1))
                        elif config['type'] == 'select':
                            params[pname] = st.selectbox(pname, config['options'])
                    i += 1
                current_algo.set_params(**params)

            c1, c2 = st.columns([1, 2])
            with c1:
                if st.button("🚀 Train Model"):
                    if st.session_state['train_set'] is None:
                        st.error("No training data!")
                    else:
                        t_col = st.session_state['col_map']['text']
                        l_col = st.session_state['col_map']['label']
                        train_data = list(zip(
                            st.session_state['train_set'][l_col],
                            st.session_state['train_set'][t_col]
                        ))
                        with st.spinner("Training..."):
                            current_algo.fit(train_data)
                        st.success("Model Trained Successfully")

            with c2:
                user_input = st.text_input("Test the model with a phrase:")
                if user_input:
                    try:
                        pred = manager.predict_one(user_input)
                        label_map = {0: "Negative", 2: "Neutral", 4: "Positive"}
                        st.metric("Prediction", f"{pred} ({label_map.get(pred, 'Unknown')})")
                    except Exception as e:
                        st.warning("Model needs training first.")

        elif MODE == "unsupervised":
            st.subheader(f"🔵 Unsupervised: {algo_name}")
            st.info("Explores the structure of the text without using labels.")
            linkage_method = st.selectbox("Linkage Method", ["ward", "average", "complete"])

            if st.button("🌳 Generate Dendrogram"):
                if st.session_state['train_set'] is None:
                    st.error("No data loaded.")
                else:
                    t_col = st.session_state['col_map']['text']
                    texts = st.session_state['train_set'][t_col].tolist()
                    if len(texts) > 500:
                        st.warning("Data too large for live demo, sampling first 500 tweets.")
                        texts = texts[:500]
                    with st.spinner("Calculating distances and linkage..."):
                        fig = plot_dendrogram(texts, method=linkage_method)
                        st.pyplot(fig)

    # =====================================================
    # 📊 TAB 4: EVALUATION
    # =====================================================
    elif page == "4. Evaluation":
        st.header("📊 Comparative Evaluation")
        if not st.session_state['human_annotations']:
            st.warning("⚠️ You haven't annotated any tweets in Tab 2 yet. We need a 'Gold Standard' to compare against.")
            st.stop()

        gold_df = pd.DataFrame(st.session_state['human_annotations'])
        st.write(f"Testing against **{len(gold_df)} manually labeled tweets**.")

        if st.button("🏁 Run Benchmark"):
            manager = st.session_state['manager']
            results = []
            for name in manager.get_available_algos():
                manager.select(name)
                algo = manager.get_current()
                if algo.mode == "supervised":
                    try:
                        texts = gold_df['text'].tolist()
                        y_true = gold_df['label'].tolist()
                        start_time = pd.Timestamp.now()
                        y_pred = algo.predict_batch(texts)
                        duration = (pd.Timestamp.now() - start_time).total_seconds()
                        acc = accuracy_score(y_true, y_pred)
                        results.append({"Algorithm": name, "Accuracy": acc, "Time (s)": duration})

                        with st.expander(f"Confusion Matrix: {name}"):
                            cm = confusion_matrix(y_true, y_pred, labels=[0, 2, 4])
                            fig, ax = plt.subplots()
                            sns.heatmap(cm, annot=True, fmt='d', xticklabels=[0, 2, 4], yticklabels=[0, 2, 4], cmap='Blues')
                            st.pyplot(fig)
                            # Image tag inserted for visual reference of a generic confusion matrix heatmap
                            st.caption("[Image of a Confusion Matrix Heatmap]")
                    except Exception as e:
                        st.error(f"Could not eval {name}: {e} (Did you train it?)")

            if results:
                st.subheader("🏆 Leaderboard")
                res_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
                st.table(res_df)
                fig, ax = plt.subplots()
                sns.barplot(data=res_df, x="Algorithm", y="Accuracy", ax=ax)
                st.pyplot(fig)

if __name__ == "__main__":
    main()