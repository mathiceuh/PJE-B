from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


# ===============================================================
#  ABSTRACT BASE CLASS
# ===============================================================
class TweetColumnDetector(ABC):
    @abstractmethod
    def detect(self, df: pd.DataFrame) -> int | None:
        """
        Given a pandas DataFrame, return the detected tweet column index (int).
        Return None if no valid column found.
        """
        pass


# ===============================================================
#  HEADER-BASED DETECTOR
# ===============================================================
class HeaderBasedDetector(TweetColumnDetector):
    def __init__(self, keywords=None):
        self.keywords = keywords or [
            "text", "tweet", "message", "content", "body",
            "review", "comment", "post", "description"
        ]

    def detect(self, df: pd.DataFrame) -> int | None:
        # Create lowercase mapping for safe matching
        lower_cols = {col.lower(): col for col in df.columns.astype(str)}

        # 1️⃣ Exact match
        for key in self.keywords:
            if key in lower_cols:
                return df.columns.get_loc(lower_cols[key])

        # 2️⃣ Partial match (keyword contained in column name)
        for col_lower, original_col in lower_cols.items():
            for key in self.keywords:
                if key in col_lower:
                    return df.columns.get_loc(original_col)

        # 3️⃣ No match found
        return None


# ===============================================================
#  LENGTH-BASED DETECTOR
# ===============================================================
class LengthBasedDetector(TweetColumnDetector):
    def detect(self, df: pd.DataFrame) -> int | None:
        if df.empty:
            return None
        best_col = df.apply(lambda col: col.astype(str).str.len().mean()).idxmax()
        return df.columns.get_loc(best_col)


# ===============================================================
#  SPACE-DENSITY DETECTOR
# ===============================================================
class SpaceDensityDetector(TweetColumnDetector):
    def detect(self, df: pd.DataFrame) -> int | None:
        if df.empty:
            return None
        best_col = df.apply(lambda col: col.astype(str).str.count(" ").mean()).idxmax()
        return df.columns.get_loc(best_col)


# ===============================================================
#  TEXT VARIETY DETECTOR
# ===============================================================
class TextVarietyDetector(TweetColumnDetector):
    def detect(self, df: pd.DataFrame) -> int | None:
        if df.empty:
            return None
        ratios = df.apply(lambda col: col.nunique(dropna=True) / len(col.dropna()))
        best_col = ratios.idxmax()
        return df.columns.get_loc(best_col)


# ===============================================================
#  HYBRID DETECTOR (length + spaces + variety)
# ===============================================================
class HybridDetector(TweetColumnDetector):
    def detect(self, df: pd.DataFrame) -> int | None:
        if df.empty:
            return None

        scores = {}
        for col in df.columns:
            s = df[col].dropna().astype(str)
            if len(s) == 0:
                continue

            avg_len = s.str.len().mean()
            space_ratio = (s.str.count(" ") / s.str.len().replace(0, np.nan)).mean()
            uniq_ratio = s.nunique() / len(s)

            # Combine normalized signals
            score = 0.4 * min(avg_len / 100, 1) + 0.3 * space_ratio + 0.3 * uniq_ratio
            scores[col] = score

        if not scores:
            return None

        best_col = max(scores, key=scores.get)
        return df.columns.get_loc(best_col)

# ===============================================================
#  5️⃣ PROVIDED (MANUAL / GUI)
# ===============================================================
class ProvidedTweetDetector(TweetColumnDetector):
    """
    Simple detector that returns a user-provided tweet column index.
    Useful for GUI interaction or debugging.
    """

    def __init__(self, provided_index: int | None = None):
        self.provided_index = provided_index

    def detect(self, df: pd.DataFrame) -> int | None:
        if self.provided_index is not None and 0 <= self.provided_index < len(df.columns):
            print(f"🧭 Using provided tweet index: {self.provided_index} ({df.columns[self.provided_index]})")
            return self.provided_index
        print("⚠️ No valid index provided to ProvidedTweetDetector.")
        return None



# ===============================================================
#  FINAL COMBINED DETECTOR
# ===============================================================
class FinalTweetDetector(TweetColumnDetector):
    def __init__(self, fallback_detector: TweetColumnDetector = None):
        self.header_detector = HeaderBasedDetector()
        self.fallback_detector = fallback_detector or LengthBasedDetector()

    def detect(self, df: pd.DataFrame) -> int | None:
        # 1️⃣ Try header-based detection
        idx = self.header_detector.detect(df)
        if idx is not None:
            print(f"✅ Header-based detector found tweet column at index {idx} ({df.columns[idx]})")
            return idx

        # 2️⃣ Fallback — dynamically chosen detector
        idx = self.fallback_detector.detect(df)
        if idx is not None:
            print(f"⚙️ Falling back to {type(self.fallback_detector).__name__} → index {idx} ({df.columns[idx]})")
        else:
            print("❌ No tweet column detected.")
        return idx


