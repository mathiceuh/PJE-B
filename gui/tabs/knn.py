import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix


def render(manager):
    st.header("📍 Algorithme : K-Nearest Neighbors (KNN)")
    st.markdown("Classification basée sur la similarité (Distance de Jaccard).")

    # 1. Vérification des données
    if 'train_df' not in st.session_state or 'test_df' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord charger et diviser les données dans l'onglet '1. Data & Cleaning'.")
        return

    train_df = st.session_state['train_df']
    test_df = st.session_state['test_df']

    col_text_idx = st.session_state.get('cleaned_text_col_idx', 1)
    col_label_idx = st.session_state.get('cleaned_label_col_idx', 0)

    # Préparation des données pour le wrapper [(label, text)]
    # On le fait à la volée pour être sûr d'avoir les dernières données
    def prepare_data(df):
        data = []
        labels = []
        for i in range(len(df)):
            txt = str(df.iloc[i, col_text_idx])
            lbl = int(df.iloc[i, col_label_idx])
            data.append((lbl, txt))
            labels.append(lbl)
        return data, labels

    train_data, train_labels = prepare_data(train_df)
    test_data, test_labels_reels = prepare_data(test_df)

    # 2. Configuration
    st.subheader("1. Configuration & Entraînement")

    c1, c2 = st.columns(2)
    with c1:
        k_value = st.slider("Nombre de voisins (k)", 1, 21, 5, step=2,
                            help="Choisissez un nombre impair de préférence.")
    with c2:
        dist_metric = st.selectbox("Distance", ["jaccard"], disabled=True, help="Jaccard est imposé pour ce TP.")

    # Bouton d'entraînement
    if st.button("🧠 Entraîner le modèle KNN", type="primary"):
        with st.spinner(f"Entraînement sur {len(train_data)} tweets..."):
            manager.select("KNN")
            algo = manager.get_current()

            # Configuration
            algo.set_params(k=k_value, distance="jaccard")

            # Fit
            algo.fit(train_data)

            # Marquer comme entraîné
            st.session_state['knn_trained'] = True
            st.success(f"Modèle entraîné avec succès (k={k_value}) !")

    # 3. Test & Évaluation
    st.divider()
    st.subheader("2. Test & Évaluation")

    if not st.session_state.get('knn_trained'):
        st.info("Entraînez le modèle pour accéder aux tests.")
    else:
        # A. Test Manuel
        col_test, col_res = st.columns([3, 1])
        with col_test:
            user_tweet = st.text_input("Tester un tweet :", placeholder="Ce film est vraiment génial !")

        if user_tweet:
            algo = manager.get_current()  # Récupérer l'instance entraînée
            pred = algo.predict_one(user_tweet)

            # Mapping pour affichage sympa
            map_res = {0: "😡 Négatif", 2: "😐 Neutre", 4: "🥰 Positif"}
            res_str = map_res.get(pred, f"Classe {pred}")

            with col_res:
                st.markdown(f"### {res_str}")

        # B. Évaluation Globale (Batch)
        st.markdown("---")
        st.write("📊 **Performance sur le Test Set**")

        if st.button("Lancer l'évaluation complète"):
            algo = manager.get_current()

            progress_bar = st.progress(0, text="Prédiction en cours...")

            # Prédiction en batch (on extrait juste les textes)
            test_texts = [t for _, t in test_data]
            predictions = algo.predict_batch(test_texts)

            progress_bar.progress(100, text="Calcul des métriques...")

            # Métriques
            acc = accuracy_score(test_labels_reels, predictions)
            cm = confusion_matrix(test_labels_reels, predictions)

            # Affichage
            c_metric, c_mat = st.columns([1, 2])

            with c_metric:
                st.metric("Accuracy", f"{acc * 100:.2f}%")
                st.caption(f"Corrects : {int(acc * len(test_labels_reels))}/{len(test_labels_reels)}")

            with c_mat:
                st.write("**Matrice de Confusion**")
                # DataFrame pour joli affichage
                labels_uniques = sorted(list(set(test_labels_reels + predictions)))
                df_cm = pd.DataFrame(cm, index=[f"Vrai {l}" for l in labels_uniques],
                                     columns=[f"Pred {l}" for l in labels_uniques])
                st.dataframe(df_cm, use_container_width=True)