# synonym_mapper.py

import json
from typing import Dict, List


class SynonymMapper:
    """
    Handles synonym normalization from a JSON file.

    JSON structure example:
    {
        "0": ["football", "soccer", "foot"],
        "1": ["happy", "joyful", "content", "glad"],
        "2": ["distance", "écart"]
    }

    → All words in each list are treated as synonyms.
      Each word in a list is replaced by the first one (canonical form).
    """

    def __init__(self, json_path: str):
        """
        Initialize mapper from a JSON file.
        :param json_path: path to JSON file containing synonym groups
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data: Dict[str, List[str]] = json.load(f)

        self.lookup = self._build_lookup(data)

    def _build_lookup(self, data: Dict[str, List[str]]) -> Dict[str, str]:
        """Build internal mapping from every synonym → canonical form."""
        lookup = {}
        for group_id, words in data.items():
            if not words:
                continue
            canonical = words[0]  # first word = canonical form
            for word in words:
                lookup[word] = canonical
        return lookup

    def normalize_tweet(self, tweet: str) -> str:
        """
        Replace each word by its canonical synonym if it exists.
        Assumes tweet is already cleaned (tokenized by whitespace).
        """
        words = tweet.split()
        normalized = [self.lookup.get(w, w) for w in words]
        return " ".join(normalized)



