# algorithms/clustering/hierarchical.py

from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances


class ClusteringHierarchique:
    def __init__(self, n_clusters=3, linkage_method="average"):
        if linkage_method == "ward":
            raise ValueError("Ward interdit avec des données textuelles (non euclidiennes).")

        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english"
        )

        self.X = None
        self.Z = None
        self.labels_ = None

    def entrainer(self, textes):
        # Vectorisation
        self.X = self.vectorizer.fit_transform(textes)

        # Distance cosinus
        dist_matrix = cosine_distances(self.X)

        # Clustering hiérarchique
        self.Z = linkage(dist_matrix, method=self.linkage_method)

        # Découpe en K clusters
        self.labels_ = fcluster(
            self.Z,
            t=self.n_clusters,
            criterion="maxclust"
        ) - 1  # clusters 0..K-1

    def recuperer_donnees_lien(self):
        return self.Z
