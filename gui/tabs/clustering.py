import streamlit as st
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score


def render(manager):
    st.header("🧶 Clustering Hiérarchique (TP)")
    st.markdown(
        "Implémentation des pipelines du cours : **Average/Complete** (Distance Mots Communs) et **Ward** (TF-IDF).")

    # 1. Chargement & Échantillonnage
    if 'train_df' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord charger les données dans l'onglet '1. Data & Cleaning'.")
        return

    df = st.session_state['train_df']
    col_text_idx = st.session_state.get('cleaned_text_col_idx', 1)
    col_label_idx = st.session_state.get('cleaned_label_col_idx', 0)

    # Slider pour limiter la taille
    n_samples = st.slider("Taille de l'échantillon (Tweets)", 50, 500, 100, help="Réduisez si c'est trop lent.")
    df_sample = df.iloc[:n_samples].copy()

    # Préparation des données
    data_train = []
    labels_reels = []
    textes_seuls = []  # Pour l'export

    for i in range(len(df_sample)):
        txt = str(df_sample.iloc[i, col_text_idx])
        # Gestion safe du label (si non supervisé, on met 0)
        try:
            lbl = int(df_sample.iloc[i, col_label_idx]) if col_label_idx is not None else 0
        except:
            lbl = 0

        data_train.append((lbl, txt))
        labels_reels.append(lbl)
        textes_seuls.append(txt)

    st.info(f"Analyse sur **{len(data_train)}** tweets.")

    # 2. Configuration
    st.subheader("Paramètres")
    c1, c2 = st.columns(2)
    with c1:
        k_clusters = st.slider("Nombre de clusters (K)", 2, 6, 3)
    with c2:
        method = st.selectbox("Méthode de lien", ["ward", "average", "complete"])
        if method == "ward":
            st.caption("✅ **Bonus**: Utilise TF-IDF + Distance Euclidienne.")
        else:
            st.caption("✅ **Standard**: Utilise Distance 'Mots Communs' (Jaccard).")

    # --- Initialisation de l'algo ---
    manager.select("Clustering")
    algo = manager.get_current()

    # 3. Lancer l'algorithme
    if st.button("🚀 Lancer le Clustering", type="primary"):
        with st.spinner("Calcul des distances et construction du dendrogramme..."):
            algo.set_params(n_clusters=k_clusters, linkage=method)
            algo.fit(data_train)

            # Sauvegarde des résultats bruts
            st.session_state['clustering_trained'] = True
            st.session_state['clustering_linkage'] = algo.get_linkage_matrix()
            st.session_state['clustering_labels'] = algo.get_labels()

            # --- CRÉATION DU CSV D'EXPORT (Immédiatement après fit) ---
            df_export = pd.DataFrame({
                "tweet": textes_seuls,
                "label_reel": labels_reels,
                "cluster_id": algo.get_labels()  # Récupère les labels générés
            })
            st.session_state['clustering_result_df'] = df_export

        st.success("Clustering terminé !")

    # --- Affichage des résultats ---
    if st.session_state.get('clustering_trained'):

        # A. Dendrogramme
        st.subheader("Visualisation : Dendrogramme")
        Z = st.session_state.get('clustering_linkage')

        if Z is not None:
            fig, ax = plt.subplots(figsize=(10, 5))
            dendrogram(Z, ax=ax, truncate_mode='lastp', p=k_clusters + 15, show_leaf_counts=True)
            plt.title(f"Dendrogramme ({method})")
            plt.xlabel("Index des Tweets / Clusters")
            plt.ylabel("Distance")
            st.pyplot(fig)

        # B. Évaluation
        st.subheader("Évaluation")
        labels_predits = st.session_state.get('clustering_labels')

        if col_label_idx is None:
            st.warning("Impossible d'évaluer : pas de colonne Label réelle sélectionnée.")
        else:
            # ARI
            ari = adjusted_rand_score(labels_reels, labels_predits)

            # Matrice de Confusion (via Pandas Crosstab)
            df_cm = pd.crosstab(
                pd.Series(labels_reels, name="Label Réel"),
                pd.Series(labels_predits, name="Cluster ID")
            )

            col_met1, col_met2 = st.columns([1, 2])
            with col_met1:
                st.metric("Adjusted Rand Index (ARI)", f"{ari:.4f}")
            with col_met2:
                st.write("**Matrice de Confusion (Label vs Cluster)**")
                st.dataframe(df_cm, use_container_width=True)

        # C. Zone de Téléchargement
        if 'clustering_result_df' in st.session_state:
            st.divider()
            st.markdown("### 📥 Télécharger les résultats")
            csv = st.session_state['clustering_result_df'].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Télécharger le CSV (Clusters)",
                data=csv,
                file_name="resultats_clustering.csv",
                mime="text/csv",
                key="download_clustering_btn"
            )

        # D. Test Interactif
        st.divider()
        st.subheader("Test Manuel")
        user_input = st.text_input("Tapez un tweet pour voir à quel cluster il appartiendrait :")

        if user_input:
            cluster_id = algo.predict_one(user_input)
            st.info(f"Ce tweet serait classé dans le **Cluster {cluster_id}**")