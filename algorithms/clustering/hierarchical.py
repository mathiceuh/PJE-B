import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


class ClusteringHierarchique:
    def __init__(self, n_clusters=3, methode_lien="average"):
        self.n_clusters = n_clusters
        self.methode_lien = methode_lien
        self.modele_lien = None
        self.labels_ = None

        # On utilise TF-IDF pour vectoriser (nécessaire pour Ward et pour la prédiction finale)
        self.vectoriseur = TfidfVectorizer(stop_words='english', max_features=1000)
        self.matrice_tfidf = None
        self.centroides = None
        self.map_centroides_labels = []

    def _nettoyer_texte(self, texte):
        """Tokenisation simple : garde uniquement les mots de plus de 2 lettres."""
        if not isinstance(texte, str):
            return set()
        # Regex pour garder uniquement les mots alphanumériques
        mots = re.findall(r'\b[a-zA-Z]{3,}\b', texte.lower())
        return set(mots)

    def _calculer_matrice_jaccard(self, textes):
        """
        Calcule la distance de Jaccard (1 - Index Jaccard) entre tous les textes.
        D = 1 - (Intersection / Union)
        """
        n = len(textes)
        matrice = np.zeros((n, n))

        # Pré-traitement pour accélérer
        sets_mots = [self._nettoyer_texte(t) for t in textes]

        for i in range(n):
            for j in range(i + 1, n):
                s1 = sets_mots[i]
                s2 = sets_mots[j]

                union = len(s1 | s2)
                inter = len(s1 & s2)

                if union == 0:
                    dist = 1.0  # Si pas de mots, distance max
                else:
                    dist = 1.0 - (inter / union)

                matrice[i, j] = dist
                matrice[j, i] = dist

        return matrice

    def entrainer(self, textes):
        # 1. Vectorisation (Toujours faite pour pouvoir calculer les centroïdes après)
        self.matrice_tfidf = self.vectoriseur.fit_transform(textes)

        # 2. Construction de la hiérarchie (Linkage)
        if self.methode_lien == "ward":
            # WARD : Distance Euclidienne sur TF-IDF
            dense_tfidf = self.matrice_tfidf.toarray()
            self.modele_lien = linkage(dense_tfidf, method='ward', metric='euclidean')

        else:
            # AVERAGE / COMPLETE : Distance Jaccard sur les mots
            matrice_dist = self._calculer_matrice_jaccard(textes)
            # Conversion en format compressé pour Scipy
            matrice_condensed = squareform(matrice_dist)
            self.modele_lien = linkage(matrice_condensed, method=self.methode_lien)

        # 3. Découpage en K clusters
        self.labels_ = fcluster(self.modele_lien, self.n_clusters, criterion='maxclust')

        # 4. Calcul des centroïdes pour la prédiction
        self._calculer_centroides()

    def _calculer_centroides(self):
        """Calcule le vecteur moyen de chaque cluster trouvé."""
        if self.matrice_tfidf is None:
            return

        dense_tfidf = self.matrice_tfidf.toarray()
        self.centroides = []
        self.map_centroides_labels = []

        # On itère sur chaque label de cluster unique trouvé
        labels_uniques = np.unique(self.labels_)

        for label in labels_uniques:
            masque = (self.labels_ == label)
            points = dense_tfidf[masque]

            if len(points) > 0:
                moyenne = points.mean(axis=0)
                self.centroides.append(moyenne)
                self.map_centroides_labels.append(label)

        self.centroides = np.array(self.centroides)

    def predire_un(self, texte):
        """Trouve le cluster le plus proche pour un nouveau texte."""
        if self.centroides is None or len(self.centroides) == 0:
            return 0

        # On vectorise le texte
        vec = self.vectoriseur.transform([texte]).toarray()

        # On calcule la distance avec chaque centroïde connu
        dists = pairwise_distances(vec, self.centroides, metric='euclidean')

        # On trouve l'index du plus proche
        idx_min = np.argmin(dists)

        # On retourne le vrai label correspondant
        return self.map_centroides_labels[idx_min]

    def recuperer_donnees_lien(self):
        return self.modele_lien