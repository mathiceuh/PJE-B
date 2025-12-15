# core/wrapper.py
from algorithms.knn.knn_classifier import KNNClassifier
from algorithms.knn.distance import JaccardDistance
from algorithms.knn.voting import MajorityVote
from algorithms.keywords.annotator import KeywordAnnotator
from algorithms.bayes.bayes import NaiveBayesClassifier
from algorithms.clustering.hierarchical import ClusteringHierarchique
import random
import os


# ============================================================
# Base Class
# ============================================================
class BaseAlgorithm:
    name = "BaseAlgorithm"
    mode = "abstract"
    output_type = None

    def __init__(self): self.params = {}

    def set_params(self, **kwargs): self.params.update(kwargs)

    def fit(self, data): raise NotImplementedError

    def predict_one(self, text): raise NotImplementedError

    def predict_batch(self, texts): return [self.predict_one(t) for t in texts]


# ============================================================
# KNN Wrapper
# ============================================================
class KNNWrapper(BaseAlgorithm):
    name = "KNN"
    mode = "supervised"
    output_type = "label"

    def __init__(self, **kwargs):
        super().__init__()
        self.params = {"k": 5, "synonyms_path": "algorithms/knn/synonyms.json"}
        self.params.update(kwargs)
        self.knn = None

    def fit(self, data):
        path = self.params["synonyms_path"]
        # Vérification basique du chemin
        dist = JaccardDistance(True, path if os.path.exists(path) else "algorithms/knn/synonyms.json")
        self.knn = KNNClassifier(k=self.params["k"], distance=dist, voter=MajorityVote())
        self.knn.fit(data)

    def predict_one(self, text): return self.knn.predict_one(text)


# ============================================================
# Keyword Wrapper
# ============================================================
class KeywordWrapper(BaseAlgorithm):
    name = "Keyword Annotation"
    mode = "rule-based"
    output_type = "label"

    def __init__(self, **kwargs):
        super().__init__()
        self.params = {"keywords_path": "data/keywords.json"}
        self.params.update(kwargs)
        self.annotator = None

    def fit(self, data=None):
        self.annotator = KeywordAnnotator(self.params["keywords_path"])

    def predict_one(self, tweet):
        from algorithms.keywords.annotation import annotate_tweet
        if not self.annotator: self.fit()
        return annotate_tweet(tweet, self.annotator.params)


# ============================================================
# Naive Bayes Wrapper (Mise à jour complète)
# ============================================================
class NaiveBayesWrapper(BaseAlgorithm):
    name = "Naive Bayes"
    mode = "supervised"
    output_type = "label"
    description = "Naive Bayes classifier (Multinomial)"

    def __init__(self, **kwargs):
        super().__init__()
        # Paramètres par défaut
        self.params = {
            "smoothing": 1.0,
            "n_gram": 1,         # Par défaut : Unigrammes
            "use_binary": False  # Par défaut : Fréquence
        }
        self.params.update(kwargs)
        self.model = None

    def fit(self, data):
        """
        Reçoit les données depuis l'interface et entraîne le modèle.
        """
        # On instancie la classe NaiveBayesClassifier avec TOUS les paramètres
        self.model = NaiveBayesClassifier(
            smoothing=self.params.get("smoothing", 1.0),
            n_gram=self.params.get("n_gram", 1),
            use_binary=self.params.get("use_binary", False)
        )
        self.model.fit(data)

    def predict_one(self, text):
        if self.model is None:
            return 2
        return self.model.predict_one(text)

    def predict_batch(self, texts):
        if self.model is None:
            return [2] * len(texts)
        return self.model.predict_batch(texts)


# ============================================================
# Clustering Wrapper
# ============================================================

class ClusteringWrapper(BaseAlgorithm):
    name = "Hierarchical Clustering"
    mode = "unsupervised"
    output_type = "cluster_id"

    def __init__(self, **kwargs):
        super().__init__()
        self.params = {"n_clusters": 3, "linkage": "average"}
        self.params.update(kwargs)
        self.algo = None

    def fit(self, data):
        if self.params["linkage"] == "ward":
            raise ValueError("Ward interdit pour le clustering textuel.")

        self.algo = ClusteringHierarchique(
            self.params["n_clusters"],
            self.params["linkage"]
        )
        self.algo.entrainer([t for _, t in data])

    def get_linkage_matrix(self):
        return self.algo.recuperer_donnees_lien() if self.algo else None

    def get_labels(self):
        return self.algo.labels_ if self.algo else []

# ============================================================
# Dummy Wrappers
# ============================================================
class DummyWrapper(BaseAlgorithm):
    name = "Dummy"
    mode = "supervised"
    output_type = "label"

    def fit(self, data): pass

    def predict_one(self, text): return random.choice([0, 2, 4])


class DummyAlgo2(BaseAlgorithm):
    name = "Dummy Always 4"
    mode = "supervised"
    output_type = "label"

    def fit(self, data): pass

    def predict_one(self, text): return 4