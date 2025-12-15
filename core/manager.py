# algorithm_manager.py
from wrapper import KNNWrapper, KeywordWrapper, NaiveBayesWrapper, DummyWrapper, DummyAlgo2, ClusteringWrapper


class AlgorithmManager:

    def __init__(self, algorithms):
        # algorithms: list of (name, wrapper_instance)
        self.algorithms = {name: algo for name, algo in algorithms}
        self.current = None

    # ---- Introspection ----
    def get_available_algos(self):
        return list(self.algorithms.keys())

    def select(self, name):
        if name not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {name}")
        self.current = self.algorithms[name]

    def get_current(self):
        return self.current

    # ---- Delegated Actions ----
    def fit(self, data):
        if not self.current:
            raise ValueError("No algorithm selected")
        self.current.fit(data)

    def predict_one(self, text):
        return self.current.predict_one(text)

    def predict_batch(self, texts):
        return self.current.predict_batch(texts)

    def evaluate(self, dataset):
        return self.current.evaluate(dataset)


# ============================================================
# Export a prebuilt manager instance
# ============================================================

manager = AlgorithmManager([
    ("KNN", KNNWrapper()),
    ("Keyword", KeywordWrapper()),
    ("Naive Bayes", NaiveBayesWrapper()),
("Clustering", ClusteringWrapper()),  # <--- AJOUTER CETTE LIGNE
    ("Dummy", DummyWrapper()),
    ("Dummy Always 4", DummyAlgo2())  # Added name string and instantiated the class
])