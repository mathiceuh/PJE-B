# distance.py

from abc import ABC, abstractmethod
from math import sqrt
from synonym_mapper import SynonymMapper


# ============================================================
# 🔹 Base Class
# ============================================================
class Distance(ABC):
    """
    Abstract base class for distance algorithms.
    Uses SynonymMapper delegate for synonym normalization.
    """

    def __init__(self, use_synonyms=False, synonym_json=None):
        """
        :param use_synonyms: whether to enable synonym normalization
        :param synonym_json: path to JSON file with synonym groups
        """
        self.use_synonyms = use_synonyms
        self.synonym_mapper = None

        if use_synonyms and synonym_json:
            self.synonym_mapper = SynonymMapper(synonym_json)

    def _apply_synonyms(self, text: str) -> str:
        """Normalize text using synonym mapper if enabled."""
        if not self.use_synonyms or not self.synonym_mapper:
            return text
        return self.synonym_mapper.normalize_tweet(text)

    @abstractmethod
    def compute(self, t1: str, t2: str) -> float:
        """Compute the distance between two cleaned tweets."""
        pass


# ============================================================
# 🔹 Helper function
# ============================================================
def build_vectors(t1: str, t2: str):
    """Convert two texts into simple bag-of-words count vectors."""
    words1 = t1.split()
    words2 = t2.split()
    vocab = list(set(words1 + words2))
    v1 = [words1.count(w) for w in vocab]
    v2 = [words2.count(w) for w in vocab]
    return v1, v2


# ============================================================
# 🔹 1. Jaccard Distance
# ============================================================
class JaccardDistance(Distance):
    def compute(self, t1, t2):
        t1 = self._apply_synonyms(t1)
        t2 = self._apply_synonyms(t2)
        words1 = set(t1.split())
        words2 = set(t2.split())
        total = len(words1 | words2)
        if total == 0:
            return 0.0
        common = len(words1 & words2)
        return (total - common) / total


# ============================================================
# 🔹 2. Cosine Distance
# ============================================================
class CosineDistance(Distance):
    def compute(self, t1, t2):
        t1 = self._apply_synonyms(t1)
        t2 = self._apply_synonyms(t2)
        words1 = t1.split()
        words2 = t2.split()
        vocab = list(set(words1 + words2))
        v1 = [words1.count(w) for w in vocab]
        v2 = [words2.count(w) for w in vocab]
        dot = sum(a*b for a, b in zip(v1, v2))
        mag1 = sqrt(sum(a*a for a in v1))
        mag2 = sqrt(sum(b*b for b in v2))
        if mag1 == 0 or mag2 == 0:
            return 0.0
        sim = dot / (mag1 * mag2)
        return 1 - sim


# ============================================================
# 🔹 3. Levenshtein Distance (normalized)
# ============================================================
class LevenshteinDistance(Distance):
    def compute(self, s1, s2):
        s1 = self._apply_synonyms(s1)
        s2 = self._apply_synonyms(s2)
        n, m = len(s1), len(s2)
        if not n and not m:
            return 0.0
        dp = [[0]*(m+1) for _ in range(n+1)]
        for i in range(n+1):
            dp[i][0] = i
        for j in range(m+1):
            dp[0][j] = j
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,
                    dp[i][j-1] + 1,
                    dp[i-1][j-1] + cost
                )
        return dp[n][m] / max(n, m)


# ============================================================
# 🔹 4. Euclidean Distance
# ============================================================
class EuclideanDistance(Distance):
    def compute(self, t1, t2):
        t1 = self._apply_synonyms(t1)
        t2 = self._apply_synonyms(t2)
        v1, v2 = build_vectors(t1, t2)
        return sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


# ============================================================
# 🔹 5. Manhattan Distance
# ============================================================
class ManhattanDistance(Distance):
    def compute(self, t1, t2):
        t1 = self._apply_synonyms(t1)
        t2 = self._apply_synonyms(t2)
        v1, v2 = build_vectors(t1, t2)
        return sum(abs(a - b) for a, b in zip(v1, v2))

