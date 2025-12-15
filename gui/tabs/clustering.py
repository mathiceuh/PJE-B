import streamlit as st
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score


def render(manager):
    st.header("🧶 Clustering Hiérarchique")

    # 1. Chargement des données
    if 'train_df' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord charger les données dans l'onglet 'Data & Cleaning'.")
        return

    df = st.session_state['train_df']
    col_text = st.session_state.get('cleaned_text_col_idx', 1)
    col_label = st.session_state.get('cleaned_label_col_idx', 0)

    # 2. Configuration
    st.subheader("1. Configuration")

    c1, c2, c3 = st.columns(3)
    with c1:
        n_samples = st.slider("Taille échantillon", 50, 500, 100, step=50, help="Plus c'est grand, plus c'est lent.")
    with c2:
        k_clusters = st.slider("Nombre de clusters (K)", 2, 6, 3)
    with c3:
        method = st.selectbox("Méthode de lien", ["ward", "average", "complete"])

    st.info(f"Prêt à clusteriser les **{n_samples}** premiers tweets.")

    # 3. Action : Entraînement
    # Utilisation d'une clé unique pour le bouton
    if st.button("🚀 Lancer le Clustering", type="primary", key="btn_run_cluster"):

        # A. Préparation des données (Snapshot instantané)
        df_slice = df.iloc[:n_samples].copy()

        textes = []
        labels_reels = []

        for i in range(len(df_slice)):
            try:
                t = str(df_slice.iloc[i, col_text])
                l = int(float(df_slice.iloc[i, col_label]))  # Convert safe
            except:
                t = ""
                l = 0
            textes.append(t)
            labels_reels.append(l)

        # B. Exécution de l'algorithme
        manager.select("Clustering")
        algo = manager.get_current()

        with st.spinner("Calcul des distances en cours..."):
            algo.set_params(n_clusters=k_clusters, linkage=method)
            algo.fit([(l, t) for l, t in zip(labels_reels, textes)])  # Format (label, text) générique

        # C. Sauvegarde des résultats dans un "Snapshot" sécurisé
        # On sauvegarde tout ce qui est nécessaire pour l'affichage pour éviter les désynchronisations
        st.session_state['clustering_snapshot'] = {
            "trained": True,
            "params": f"K={k_clusters}, Méthode={method}",
            "linkage_matrix": algo.get_linkage_matrix(),
            "labels_pred": algo.get_labels(),
            "labels_true": labels_reels,
            "textes": textes,
            "k": k_clusters
        }
        st.success("Clustering terminé avec succès !")

    # 4. Affichage des Résultats (basé UNIQUEMENT sur le snapshot)
    snapshot = st.session_state.get('clustering_snapshot')

    if snapshot and snapshot.get("trained"):
        st.divider()
        st.subheader(f"2. Résultats ({snapshot['params']})")

        # Récupération sécurisée des données
        labels_pred = snapshot['labels_pred']
        labels_true = snapshot['labels_true']
        textes = snapshot['textes']
        Z = snapshot['linkage_matrix']
        k = snapshot['k']

        # A. Dendrogramme
        if Z is not None:
            fig, ax = plt.subplots(figsize=(10, 4))
            dendrogram(Z, ax=ax, truncate_mode='lastp', p=k + 10, show_leaf_counts=True)
            plt.title("Dendrogramme des tweets")
            plt.xlabel("Clusters / Tweets")
            st.pyplot(fig)

        # B. Métriques & Tableau
        if len(labels_pred) == len(labels_true):
            ari = adjusted_rand_score(labels_true, labels_pred)

            c_met, c_tab = st.columns([1, 2])
            with c_met:
                st.metric("Adjusted Rand Index (ARI)", f"{ari:.4f}")
                st.caption("1.0 = Parfait, 0.0 = Aléatoire")

            with c_tab:
                df_cm = pd.crosstab(
                    pd.Series(labels_true, name="Vrai Label"),
                    pd.Series(labels_pred, name="Cluster Trouvé")
                )
                st.write("**Répartition (Matrice de Confusion)**")
                st.dataframe(df_cm, use_container_width=True)
        else:
            st.error("Erreur de dimensions de données. Veuillez relancer.")

        # C. Export CSV
        df_export = pd.DataFrame({
            "tweet": textes,
            "label_reel": labels_true,
            "cluster_id": labels_pred
        })
        csv_data = df_export.to_csv(index=False).encode('utf-8')

        st.download_button(
            "📥 Télécharger les résultats (CSV)",
            data=csv_data,
            file_name="clustering_results.csv",
            mime="text/csv"
        )

        # D. Test Manuel (Bonus)
        st.divider()
        st.write("**Tester la prédiction d'un cluster**")
        txt_test = st.text_input("Tweet à tester :")
        if txt_test:
            algo = manager.get_current()
            # On vérifie que l'algo est bien celui entraîné (juste au cas où)
            if algo and hasattr(algo, 'algo') and algo.algo.centroides is not None:
                cid = algo.predict_one(txt_test)
                st.info(f"Ce tweet appartient au **Cluster {cid}**")
            else:
                st.warning("L'algorithme doit être ré-entraîné pour prédire.")