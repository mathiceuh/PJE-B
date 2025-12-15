# tweet_cleaner.py
import re
import pandas as pd
from langdetect import detect, DetectorFactory
from .rules import default_rules

DetectorFactory.seed = 0  # ensure reproducibility


# ===============================================================
#  Row / Dataset Filter Classes
# ===============================================================

class RemoveMixedEmojiRow:
    """
    Remove rows where a tweet contains both positive and negative emojis.
    """

    def __init__(self):
        self.positive_emoji_pattern = re.compile(r"[😀😃😄😁😆😅😂🤣😊🙂😉😍😘😗😙😚😋😜😝😛🤗🤩❤💖💗💓💞💕💝💘👍👏🙏🤝🤟]")
        self.negative_emoji_pattern = re.compile(r"[😞😔😟😕🙁☹😣😖😫😩😭😢😠😡🤬👎😤😒😞😓😩😫😱😨😰]")

    def should_drop(self, text: str) -> bool:
        has_pos = bool(self.positive_emoji_pattern.search(text))
        has_neg = bool(self.negative_emoji_pattern.search(text))
        return has_pos and has_neg

    def apply(self, df: pd.DataFrame, tweet_idx: int) -> pd.DataFrame:
        mask = df.iloc[:, tweet_idx].astype(str).apply(self.should_drop)
        removed = mask.sum()
        print(f"🧹 Removed {removed} mixed-emoji tweets")
        return df[~mask]


class RemoveDuplicates:
    """
    Remove duplicate tweets (optionally by label as well).
    """

    def __init__(self, subset="tweet"):
        self.subset = subset  # can be "tweet" or "tweet_label"

    def apply(self, df: pd.DataFrame, tweet_idx: int, label_idx: int | None = None) -> pd.DataFrame:
        if self.subset == "tweet_label" and label_idx is not None:
            df = df.drop_duplicates(subset=[df.columns[tweet_idx], df.columns[label_idx]])
        else:
            df = df.drop_duplicates(subset=[df.columns[tweet_idx]])
        print(f"🧹 Removed duplicates → {df.shape[0]} remaining rows")
        return df


class RemoveDifferentLanguageTweets:
    """
    Detect the dominant language in the dataset and remove tweets in other languages.
    """

    def __init__(self, min_length: int = 10):
        self.min_length = min_length  # skip very short tweets that confuse detector

    def detect_language(self, text: str) -> str:
        try:
            if len(text.strip()) < self.min_length:
                return "unknown"
            return detect(text)
        except Exception:
            return "unknown"

    def apply(self, df: pd.DataFrame, tweet_idx: int) -> pd.DataFrame:
        # Detect languages
        langs = df.iloc[:, tweet_idx].astype(str).apply(self.detect_language)

        # Find dominant language
        dominant_lang = langs.value_counts().idxmax() if not langs.empty else "unknown"

        #debug
        print("\n🔍 Language detection results (first 10):")
        print(list(zip(df.iloc[:10, tweet_idx], langs[:10])))
        print("Detected counts:", langs.value_counts().to_dict())

        # Apply mask
        mask = langs == dominant_lang
        kept = mask.sum()
        total = len(df)

        print(f"🌍 Dominant language detected: '{dominant_lang}'")
        print(f"🗑 Removed {total - kept} tweets not in '{dominant_lang}'")

        return df[mask]



# ===============================================================
#  TWEET CLEANER CLASS
# ===============================================================

class TweetCleaner:
    """
    Main cleaning pipeline — applies text rules, row filters, and dataset filters.
    """

    def __init__(self, text_rules=None, row_filters=None, dataset_filters=None):
        # Use None-checking instead of truthy/falsy
        self.text_rules = default_rules() if text_rules is None else text_rules
        self.row_filters = [RemoveMixedEmojiRow()] if row_filters is None else row_filters
        self.dataset_filters = [RemoveDuplicates()] if dataset_filters is None else dataset_filters

    # -----------------------
    # Apply text cleaning rules
    # -----------------------
    def _apply_text_rules(self, df: pd.DataFrame, tweet_idx: int) -> pd.DataFrame:
        df = df.copy()
        for rule in self.text_rules:
            df.iloc[:, tweet_idx] = df.iloc[:, tweet_idx].astype(str).apply(rule.apply)
        return df

    # -----------------------
    # Apply row filters
    # -----------------------
    def _apply_row_filters(self, df: pd.DataFrame, tweet_idx: int) -> pd.DataFrame:
        for filt in self.row_filters:
            df = filt.apply(df, tweet_idx)
        return df

    # -----------------------
    # Apply dataset filters
    # -----------------------
    def _apply_dataset_filters(self, df: pd.DataFrame, tweet_idx: int, label_idx: int | None = None) -> pd.DataFrame:
        for filt in self.dataset_filters:
            if isinstance(filt, RemoveDuplicates):
                df = filt.apply(df, tweet_idx, label_idx)
            else:
                df = filt.apply(df, tweet_idx)
        return df

    # -----------------------
    # Main cleaning entry point
    # -----------------------
    def clean_dataframe(self, df: pd.DataFrame, tweet_idx: int, label_idx: int | None = None) -> pd.DataFrame:
        print("✨ Starting tweet cleaning pipeline...")

        df = self._apply_text_rules(df, tweet_idx)
        df = self._apply_row_filters(df, tweet_idx)
        df = self._apply_dataset_filters(df, tweet_idx, label_idx)

        df = df.reset_index(drop=True)
        print(f"✅ Cleaning complete. Remaining rows: {df.shape[0]}")
        return df


