import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


class ClusteringHierarchique:
    def __init__(self, n_clusters=3, methode_lien="average"):
        self.n_clusters = n_clusters
        self.methode_lien = methode_lien
        self.modele_lien = None
        self.labels_ = None
        # On utilise TF-IDF par défaut pour vectoriser le texte
        self.vectoriseur = TfidfVectorizer(stop_words='english')
        self.matrice_tfidf = None
        self.centroides = None

    def _calculer_matrice_jaccard_cours(self, textes):
        """
        Implémente la distance du cours :
        D(t1, t2) = (Total_mots - Mots_communs) / Total_mots
        Optimisé avec des ensembles (sets) Python.
        """
        n = len(textes)
        matrice = np.zeros((n, n))

        # Pré-tokenisation pour gagner du temps
        tokens_list = [set(str(t).lower().split()) for t in textes]

        for i in range(n):
            for j in range(i + 1, n):  # On remplit le triangle supérieur
                mots1 = tokens_list[i]
                mots2 = tokens_list[j]

                union = len(mots1 | mots2)  # Nombre total de mots uniques
                inter = len(mots1 & mots2)  # Nombre de mots communs

                if union == 0:
                    dist = 0.0
                else:
                    dist = (union - inter) / union

                matrice[i, j] = dist
                matrice[j, i] = dist  # Symétrie

        return matrice

    def entrainer(self, textes):
        # 1. Vectorisation (toujours utile pour Ward et pour les centroïdes)
        self.matrice_tfidf = self.vectoriseur.fit_transform(textes)

        # 2. Choix du Pipeline selon le cours
        if self.methode_lien == "ward":
            # --- Pipeline WARD (BONUS) ---
            # Nécessite distances euclidiennes sur vecteurs TF-IDF
            # On laisse scipy calculer la distance euclidienne implicitement ou on lui passe la matrice
            # Pour être explicite comme dans le cours :
            dense_tfidf = self.matrice_tfidf.toarray()
            self.modele_lien = linkage(dense_tfidf, method='ward', metric='euclidean')

        else:
            # --- Pipeline AVERAGE / COMPLETE ---
            # Utilise la matrice de distance "mots communs"
            matrice_dist = self._calculer_matrice_jaccard_cours(textes)

            # Conversion en format "condensed" pour scipy (requis par squareform)
            matrice_condensed = squareform(matrice_dist)
            self.modele_lien = linkage(matrice_condensed, method=self.methode_lien)

        # 3. Découpage de l'arbre (Flat Clustering)
        self.labels_ = fcluster(self.modele_lien, self.n_clusters, criterion='maxclust')

        # 4. Calcul des centroïdes (Pour la fonctionnalité "Prédire un tweet")
        self._calculer_centroides()

    def _calculer_centroides(self):
        """Calcule le centre moyen de chaque cluster (basé sur TF-IDF)."""
        dense_tfidf = self.matrice_tfidf.toarray()
        self.centroides = []

        # fcluster retourne des labels commençant à 1, on les suit
        labels_uniques = np.unique(self.labels_)

        for label in labels_uniques:
            masque = (self.labels_ == label)
            points = dense_tfidf[masque]
            if len(points) > 0:
                moyenne = points.mean(axis=0)
                self.centroides.append(moyenne)
            else:
                self.centroides.append(np.zeros(dense_tfidf.shape[1]))

        self.centroides = np.array(self.centroides)

    def predire_un(self, texte):
        """Assigne un nouveau tweet au cluster le plus proche (Nearest Centroid)."""
        if self.centroides is None:
            return 0
        vec = self.vectoriseur.transform([texte]).toarray()
        dists = pairwise_distances(vec, self.centroides, metric='euclidean')
        # +1 car nos labels fcluster commencent à 1
        return np.argmin(dists) + 1

    def recuperer_donnees_lien(self):
        return self.modele_lien