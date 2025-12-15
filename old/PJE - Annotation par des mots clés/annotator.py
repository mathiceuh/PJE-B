# ===============================================================
#  keyword_annotator.py — Keyword-based tweet annotation module
# ===============================================================

import pandas as pd
import json
from annotation import AnnotationParams, annotate_tweet


class KeywordAnnotator:
    """
    Annotate a dataset using keyword-based sentiment rules.

    Supports two modes:
      - 'override' → replaces existing label column.
      - 'add'      → adds a new column ('predicted_label'),
                     creates label column if missing.
    """

    def __init__(self, json_path: str = "keywords.json"):
        # Load positive and negative word lists
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        positive_words = data.get("positive", [])
        negative_words = data.get("negative", [])
        self.params = AnnotationParams(positive_words=positive_words, negative_words=negative_words)

    # ---------------------------------------------------------------
    # Apply annotation to a dataset
    # ---------------------------------------------------------------
    def annotate(
        self,
        csv_path: str,
        output_path: str = "annotated_output.csv",
        mode: str = "add"
    ):
        """
        Annotate the dataset.

        Parameters
        ----------
        csv_path : str
            Input CSV file (cleaned).
        output_path : str
            Where to save annotated output.
        mode : str
            'override' → replace existing label column.
            'add' → add a new column 'predicted_label'
                    (creates one if label column doesn’t exist).
        """
        print(f"📂 Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)

        # Detect tweet column (assume second column)
        if df.shape[1] < 2:
            raise ValueError("CSV must have at least 2 columns (label + tweet)")
        label_col = df.columns[0]
        tweet_col = df.columns[1]

        # Annotate tweets
        print(f"🧠 Annotating tweets... (mode={mode})")
        predictions = df[tweet_col].astype(str).apply(lambda t: annotate_tweet(t, self.params))

        if mode == "override":
            # Replace label values directly
            df[label_col] = predictions
        elif mode == "add":
            # Add new column (and create label if missing)
            if "predicted_label" not in df.columns:
                df["predicted_label"] = predictions
            else:
                df["predicted_label"] = predictions
        else:
            raise ValueError("Invalid mode. Choose 'override' or 'add'.")

        # Save results
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"✅ Annotation complete → {output_path}")
        return df


class KeywordAnnotator:
    def _annotate_text(self, text):
        """Annote un seul texte."""
        text = str(text).lower()

        for label, keywords in self.keywords.items():
            if any(keyword in text for keyword in keywords):
                return int(label)  # Retourne 0, 1 ou 2

        return 2  # Neutre par défaut
