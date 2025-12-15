import threading
from charset_normalizer.api import from_path
import re

positive_words = []
negative_words = []

def read_words_from_file(filename, target_list):
    """Read word list robustly (newline or comma separated)."""
    try:
        encoding = from_path(filename).best().encoding or "utf-8"

        with open(filename, "r", encoding=encoding, errors="ignore") as f:
            text = f.read()

        # Split by newlines or commas, but NOT inside phrases
        words = re.split(r"[\n,]+", text)
        words = [w.strip().lower() for w in words if w.strip()]

        # Remove weird fragments (optional heuristic)
        words = [w for w in words if len(w) > 2 or " " in w]  # drop 'a', 'g', etc.

        target_list.extend(words)
    except Exception as e:
        print(f"❌ Error reading {filename}: {e}")

# Threading
t1 = threading.Thread(target=read_words_from_file, args=("positive.txt", positive_words))
t2 = threading.Thread(target=read_words_from_file, args=("negative.txt", negative_words))

t1.start(); t2.start()
t1.join(); t2.join()

import json

data = {
    "positive": positive_words,
    "negative": negative_words
}

def load_keywords() -> dict:
    """Load both positive and negative keywords concurrently."""
    positive_words, negative_words = [], []

    t1 = threading.Thread(target=read_words_from_file, args=("positive.txt", positive_words))
    t2 = threading.Thread(target=read_words_from_file, args=("negative.txt", negative_words))

    t1.start(); t2.start()
    t1.join(); t2.join()

    # optional export for debugging
    with open("keywords.json", "w", encoding="utf-8") as f:
        json.dump({"positive": positive_words, "negative": negative_words}, f, ensure_ascii=False, indent=2)

    return {"positive": positive_words, "negative": negative_words}