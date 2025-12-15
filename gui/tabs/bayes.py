import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix


def render(manager):
    st.header("🎲 Algorithme : Naive Bayes")

    # Force la sélection pour éviter le bug du "manager perdu"
    manager.select("Naive Bayes")

    # 1. Vérification des données
    if 'train_df' not in st.session_state or 'test_df' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord charger les données dans l'onglet '1. Data & Cleaning'.")
        return

    train_df = st.session_state['train_df']
    test_df = st.session_state['test_df']

    col_text_idx = st.session_state.get('cleaned_text_col_idx', 1)
    col_label_idx = st.session_state.get('cleaned_label_col_idx', 0)

    # Préparation des données (Filtre 0, 2, 4 uniquement)
    def preparer_donnees(df):
        data = []
        labels = []
        for i in range(len(df)):
            try:
                txt = str(df.iloc[i, col_text_idx])
                val = df.iloc[i, col_label_idx]
                lbl = int(float(val))
                if lbl in [0, 2, 4]:
                    data.append((lbl, txt))
                    labels.append(lbl)
            except:
                continue
        return data, labels

    train_data, _ = preparer_donnees(train_df)
    test_data, test_labels = preparer_donnees(test_df)

    # 2. Configuration & Entraînement
    st.subheader("1. Configuration & Entraînement")

    c1, c2 = st.columns(2)
    with c1:
        smoothing = st.slider("Lissage (Alpha)", 0.1, 5.0, 1.0, 0.1, help="Gère les mots inconnus.")
    with c2:
        st.info(f"📊 **{len(train_data)}** tweets prêts pour l'entraînement.")

    # Bouton d'entraînement
    if st.button("🧠 Entraîner le modèle", type="primary"):
        with st.spinner("Calcul des probabilités..."):
            algo = manager.get_current()
            algo.set_params(smoothing=smoothing)
            algo.fit(train_data)

            st.session_state['bayes_trained'] = True
            st.success("Modèle entraîné avec succès !")

    # 3. Test & Évaluation
    st.divider()
    st.subheader("2. Test & Évaluation")

    if not st.session_state.get('bayes_trained'):
        st.info("Veuillez entraîner le modèle pour accéder aux tests.")
    else:
        # A. Test Manuel
        col_test, col_res = st.columns([3, 1])
        with col_test:
            user_tweet = st.text_input("Tester un tweet :", placeholder="Ex: Ce cours est génial !")

        if user_tweet:
            algo = manager.get_current()
            pred = algo.predict_one(user_tweet)

            # Affichage du résultat
            map_res = {0: "😡 Négatif", 2: "😐 Neutre", 4: "🥰 Positif"}
            res_str = map_res.get(pred, f"Classe {pred}")

            with col_res:
                st.markdown(f"### {res_str}")

            # Debug : Mots reconnus
            if hasattr(algo, 'model') and algo.model:
                tokens = algo.model._get_ngrams(user_tweet)
                connus = [w for w in tokens if w in algo.model.vocab]
                with st.expander("🔍 Détails de la prédiction"):
                    if not connus:
                        st.warning("Aucun mot connu -> Neutre par défaut.")
                    else:
                        st.write(f"Mots reconnus : {connus}")

        # B. Évaluation Globale
        st.markdown("---")
        st.write("📊 **Performance sur le Test Set**")

        # Utilisation d'une clé unique pour éviter le bug StreamlitDuplicateElementId
        if st.button("Lancer l'évaluation complète", key="bayes_eval_btn"):
            algo = manager.get_current()

            progress = st.progress(0, text="Prédiction en cours...")

            textes_test = [t for _, t in test_data]
            predictions = algo.predict_batch(textes_test)

            progress.progress(100, text="Calcul des métriques...")

            acc = accuracy_score(test_labels, predictions)
            cm = confusion_matrix(test_labels, predictions)

            # 1. Affichage Métriques
            c_metric, c_mat = st.columns([1, 2])
            with c_metric:
                st.metric("Accuracy", f"{acc * 100:.2f}%")

            with c_mat:
                st.write("**Matrice de Confusion**")
                labels_classes = sorted(list(set(test_labels + predictions)))
                df_cm = pd.DataFrame(cm,
                                     index=[f"Vrai {c}" for c in labels_classes],
                                     columns=[f"Pred {c}" for c in labels_classes])
                st.dataframe(df_cm, use_container_width=True)

            # 2. Préparation du CSV pour téléchargement (Stockage dans session_state)
            df_export = pd.DataFrame({
                "tweet": textes_test,
                "label_reel": test_labels,
                "label_predit": predictions
            })
            # On ajoute une colonne lisible
            map_label = {0: "Negative", 2: "Neutral", 4: "Positive"}
            df_export["sentiment_predit"] = df_export["label_predit"].map(map_label)

            st.session_state['bayes_result_df'] = df_export
            st.success("Évaluation terminée ! Résultats prêts au téléchargement.")

        # C. Zone de Téléchargement (Affichée si les résultats existent)
        if 'bayes_result_df' in st.session_state:
            st.markdown("### 📥 Télécharger les résultats")

            csv = st.session_state['bayes_result_df'].to_csv(index=False).encode('utf-8')

            st.download_button(
                label="⬇️ Télécharger le CSV classifié",
                data=csv,
                file_name="resultats_naive_bayes.csv",
                mime="text/csv",
                key="download_bayes_btn"
            )