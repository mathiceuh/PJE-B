import streamlit as st
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def render(manager):
    st.header("🧶 Clustering Hiérarchique (Texte)")

    # ============================================================
    # 1. Vérification des données
    # ============================================================
    if 'train_df' not in st.session_state:
        st.warning("⚠️ Chargez d'abord les données dans l'onglet 'Data & Cleaning'.")
        return

    df = st.session_state['train_df']
    col_text = st.session_state.get('cleaned_text_col_idx', 1)
    col_label = st.session_state.get('cleaned_label_col_idx', 0)

    # ============================================================
    # 2. Configuration
    # ============================================================
    st.subheader("1. Configuration")

    c1, c2, c3 = st.columns(3)

    with c1:
        n_samples = st.slider(
            "Taille de l'échantillon",
            min_value=50,
            max_value=500,
            value=100,
            step=50
        )

    with c2:
        k_clusters = st.slider(
            "Nombre de clusters (K)",
            min_value=2,
            max_value=6,
            value=3
        )

    with c3:
        linkage_method = st.selectbox(
            "Méthode de lien",
            ["average", "complete"],
            help="Méthodes compatibles avec les distances cosinus"
        )

    st.info(f"Clustering de **{n_samples} tweets** avec **K={k_clusters}**, méthode **{linkage_method}**.")

    # ============================================================
    # 3. Lancement du clustering
    # ============================================================
    if st.button("🚀 Lancer le clustering", type="primary"):

        # --- Préparation des données ---
        df_slice = df.iloc[:n_samples]

        textes = []
        labels_true = []

        for i in range(len(df_slice)):
            try:
                texte = str(df_slice.iloc[i, col_text])
                label = int(float(df_slice.iloc[i, col_label]))
            except Exception:
                texte = ""
                label = 0

            textes.append(texte)
            labels_true.append(label)

        # --- Exécution ---
        manager.select("Clustering")
        algo = manager.get_current()

        with st.spinner("Clustering hiérarchique en cours..."):
            algo.set_params(
                n_clusters=k_clusters,
                linkage=linkage_method
            )
            algo.fit(list(zip(labels_true, textes)))

        # --- Snapshot ---
        st.session_state["clustering_snapshot"] = {
            "trained": True,
            "params": f"K={k_clusters}, linkage={linkage_method}",
            "labels_pred": algo.get_labels(),
            "labels_true": labels_true,
            "textes": textes,
            "linkage_matrix": algo.get_linkage_matrix(),
            "k": k_clusters
        }

        st.success("Clustering terminé.")

    # ============================================================
    # 4. Affichage des résultats
    # ============================================================
    snapshot = st.session_state.get("clustering_snapshot")

    if not snapshot or not snapshot.get("trained"):
        return

    st.divider()
    st.subheader(f"2. Résultats ({snapshot['params']})")

    labels_pred = snapshot["labels_pred"]
    labels_true = snapshot["labels_true"]
    textes = snapshot["textes"]
    Z = snapshot["linkage_matrix"]
    k = snapshot["k"]

    # ============================================================
    # A. Dendrogramme
    # ============================================================
    if Z is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        dendrogram(
            Z,
            ax=ax,
            truncate_mode="lastp",
            p=k + 10,
            show_leaf_counts=True
        )
        ax.set_title("Dendrogramme (vue tronquée)")
        ax.set_xlabel("Clusters / Tweets")
        st.pyplot(fig)

    # ============================================================
    # B. Évaluation (ARI)
    # ============================================================
    if len(labels_pred) == len(labels_true):
        ari = adjusted_rand_score(labels_true, labels_pred)

        c1, c2 = st.columns([1, 2])

        with c1:
            st.metric("Adjusted Rand Index (ARI)", f"{ari:.4f}")
            st.caption("ARI ≈ 0 attendu : similarité lexicale ≠ sentiment")

        with c2:
            df_cm = pd.crosstab(
                pd.Series(labels_true, name="Label réel"),
                pd.Series(labels_pred, name="Cluster")
            )
            st.write("**Répartition des labels par cluster**")
            st.dataframe(df_cm, use_container_width=True)
    else:
        st.error("Erreur : dimensions incohérentes.")

    # ============================================================
    # C. Export CSV
    # ============================================================
    df_export = pd.DataFrame({
        "tweet": textes,
        "label_reel": labels_true,
        "cluster_id": labels_pred
    })

    csv = df_export.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Télécharger les résultats (CSV)",
        data=csv,
        file_name="clustering_hierarchique.csv",
        mime="text/csv"
    )

