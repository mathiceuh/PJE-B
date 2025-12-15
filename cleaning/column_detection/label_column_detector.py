from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


# ===============================================================
#  BASE CLASS (common interface)
# ===============================================================
class LabelColumnDetector(ABC):
    @abstractmethod
    def detect(self, df: pd.DataFrame) -> int | None:
        """
        Given a pandas DataFrame, return the detected label column index.
        """
        pass


# ===============================================================
#  1️⃣ HEADER-BASED DETECTOR
# ===============================================================
class HeaderLabelDetector(LabelColumnDetector):
    def __init__(self, keywords=None):
        self.keywords = keywords or [
            "target", "label", "sentiment", "polarity", "class", "output", "result"
        ]

    def detect(self, df: pd.DataFrame) -> int | None:
        lower_cols = {col.lower(): col for col in df.columns.astype(str)}

        # Exact match
        for key in self.keywords:
            if key in lower_cols:
                return df.columns.get_loc(lower_cols[key])

        # Partial match
        for col_lower, original_col in lower_cols.items():
            for key in self.keywords:
                if key in col_lower:
                    return df.columns.get_loc(original_col)

        return None


# ===============================================================
#  2️⃣ VALUE-PATTERN DETECTOR
# ===============================================================
class ValuePatternLabelDetector(LabelColumnDetector):
    """
    Detect label columns based on numeric or short text categories:
    - numeric values like 0,2,4 or -1,0,1
    - text values like 'positive', 'neutral', 'negative'
    """

    def __init__(self):
        self.text_keywords = [
            "pos", "neg", "neu", "positive", "negative", "neutral",
            "good", "bad", "mauvais", "bon", "neutre"
        ]

    def detect(self, df: pd.DataFrame) -> int | None:
        scores = {}

        for col in df.columns:
            s = df[col].dropna().astype(str)
            n_unique = s.nunique()
            n_total = len(s)

            if n_total == 0:
                continue

            # Numeric detection
            numeric_ratio = sum(s.str.fullmatch(r"[-+]?\d+").fillna(False)) / n_total

            if numeric_ratio > 0.8 and 2 <= n_unique <= 10:
                scores[col] = 1.0  # strong candidate
                continue

            # Text detection (short, categorical)
            avg_len = s.str.len().mean()
            text_hits = sum(
                s.str.lower().isin(self.text_keywords)
            )
            if n_unique <= 10 and avg_len < 15 and text_hits > 0:
                scores[col] = 0.8  # strong but slightly lower confidence

        if not scores:
            return None

        # Return best column by score
        best_col = max(scores, key=scores.get)
        return df.columns.get_loc(best_col)


# ===============================================================
#  3️⃣ BACKUP (POSITIONAL HEURISTIC)
# ===============================================================
class PositionalBackupDetector(LabelColumnDetector):
    """
    Fallback rule: assume first small numeric column is label.
    """
    def detect(self, df: pd.DataFrame) -> int | None:
        for idx, col in enumerate(df.columns):
            try:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if s.nunique() <= 10:
                    return idx
            except Exception:
                continue
        return None


# ===============================================================
#  4️⃣ PROVIDED (MANUAL / GUI)
# ===============================================================
class ProvidedLabelDetector(LabelColumnDetector):
    """
    Detector that simply returns a user-provided column index.
    Useful for GUI where user manually selects the label column.
    """

    def __init__(self, provided_index: int | None = None):
        self.provided_index = provided_index

    def detect(self, df: pd.DataFrame) -> int | None:
        if self.provided_index is not None and 0 <= self.provided_index < len(df.columns):
            print(f"🧭 Using provided label index: {self.provided_index} ({df.columns[self.provided_index]})")
            return self.provided_index
        print("⚠️ No valid index provided to ProvidedLabelDetector.")
        return None


# ===============================================================
#  FINAL COMBINED DETECTOR
# ===============================================================
class FinalLabelDetector(LabelColumnDetector):
    def __init__(self, fallback_detector: LabelColumnDetector = None):
        self.header_detector = HeaderLabelDetector()
        self.value_detector = ValuePatternLabelDetector()
        self.backup_detector = PositionalBackupDetector()

    def detect(self, df: pd.DataFrame) -> int | None:
        # Try header-based
        idx = self.header_detector.detect(df)
        if idx is not None:
            print(f"✅ Header-based label detected: index {idx} ({df.columns[idx]})")
            return idx

        # Try value-pattern based
        idx = self.value_detector.detect(df)
        if idx is not None:
            print(f"⚙️ Value-pattern-based label detected: index {idx} ({df.columns[idx]})")
            return idx

        # Fallback positional heuristic
        idx = self.backup_detector.detect(df)
        if idx is not None:
            print(f"🪄 Backup heuristic used: index {idx} ({df.columns[idx]})")
            return idx

        print("❌ No label column detected.")
        return None


