# rules.py
import re
import string


# ===============================================================
#  Base class for all rules
# ===============================================================
class CleaningRule:
    """Base class for all text cleaning rules."""
    def apply(self, text: str) -> str:
        raise NotImplementedError("Each rule must implement the apply() method.")


# ===============================================================
#  1️⃣ Lowercase rule
# ===============================================================
class ToLowercase(CleaningRule):
    """Convert text to lowercase."""
    def apply(self, text: str) -> str:
        return text.lower()


# ===============================================================
#  2️⃣ Remove URLs
# ===============================================================
class RemoveURLs(CleaningRule):
    """Remove all http/https URLs from text."""
    def __init__(self):
        self.pattern = re.compile(r"http\S+|www\.\S+")
    def apply(self, text: str) -> str:
        return self.pattern.sub("", text)


# ===============================================================
#  3️⃣ Remove mentions (@username)
# ===============================================================
class RemoveMentions(CleaningRule):
    """Remove all @mentions."""
    def __init__(self):
        self.pattern = re.compile(r"@\w+")
    def apply(self, text: str) -> str:
        return self.pattern.sub("", text)


# ===============================================================
#  4️⃣ Remove hashtags (#hashtag)
# ===============================================================
class RemoveHashtags(CleaningRule):
    """Remove hashtags (#something) but keep the word if desired."""
    def __init__(self, keep_word=False):
        # keep_word=True → "#happy" -> "happy"
        self.keep_word = keep_word
        self.pattern = re.compile(r"#(\w+)")
    def apply(self, text: str) -> str:
        if self.keep_word:
            return self.pattern.sub(r"\1", text)
        else:
            return self.pattern.sub("", text)


# ===============================================================
#  5️⃣ Remove retweet markers (RT)
# ===============================================================
class RemoveRetweetMarker(CleaningRule):
    """
    Remove retweet markers ('RT', 'rt') whether at the start or inside text.
    Also collapses leftover extra spaces.
    """
    def __init__(self):
        # (?i) → case-insensitive; \b → word boundary (ensures we match standalone RT)
        self.pattern = re.compile(r"(?i)\brt\b")
        self.space_pattern = re.compile(r"\s{2,}")

    def apply(self, text: str) -> str:
        text = self.pattern.sub("", text)
        return self.space_pattern.sub(" ", text).strip()



# ===============================================================
#  6️⃣ Remove punctuation
# ===============================================================
class RemovePunctuation(CleaningRule):
    """Remove punctuation characters from text."""
    def __init__(self):
        self.table = str.maketrans("", "", string.punctuation)
    def apply(self, text: str) -> str:
        return text.translate(self.table)


# ===============================================================
#  7️⃣ Normalize whitespace
# ===============================================================
class NormalizeWhitespace(CleaningRule):
    """Collapse multiple spaces and strip edges."""
    def __init__(self):
        self.pattern = re.compile(r"\s+")
    def apply(self, text: str) -> str:
        return self.pattern.sub(" ", text).strip()


# ===============================================================
#  8️⃣ Optional — Remove emojis (bonus)
# ===============================================================
class RemoveEmojis(CleaningRule):
    """Remove common emoji characters."""
    def __init__(self):
        self.pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+", flags=re.UNICODE)
    def apply(self, text: str) -> str:
        return self.pattern.sub("", text)


# ===============================================================
#  Helper: default rules pipeline
# ===============================================================
def default_rules(keep_hashtag_word=False):
    """Return a default list of common cleaning rules in recommended order."""
    return [
        ToLowercase(),
        RemoveURLs(),
        RemoveMentions(),
        RemoveHashtags(keep_word=keep_hashtag_word),
        RemoveRetweetMarker(),
        RemovePunctuation(),
        NormalizeWhitespace(),
    ]
