class AnnotationParams:
    """
    Holds all resources and configuration parameters
    required for the sentiment annotation algorithm.

    This class is purely a data container — it does not perform annotation itself.
    """
    def __init__(
        self,
        positive_words: list[str] | None = None,
        negative_words: list[str] | None = None,
        memo: dict[str, str] | None = None,
        lowercase: bool = True,
        use_stemming: bool = False,
        language: str = "en"
    ):
        self.positive_words = set(positive_words or [])
        self.negative_words = set(negative_words or [])
        self.memo = memo if memo is not None else {}

        # Optional configuration flags
        self.lowercase = lowercase
        self.use_stemming = use_stemming
        self.language = language

    def __repr__(self):
        return (
            f"AnnotationParams("
            f"positive_words={len(self.positive_words)}, "
            f"negative_words={len(self.negative_words)}, "
            f"memo_size={len(self.memo)}, "
            f"lowercase={self.lowercase}, "
            f"use_stemming={self.use_stemming}, "
            f"language='{self.language}')"
        )



def annotate_tweet(tweet: str, params: "AnnotationParams") -> int:
    """
    Retourne la polarité d'un tweet selon les règles définies :
    - 4 : plus de mots positifs
    - 0 : plus de mots négatifs
    - 2 : égalité ou aucun mot trouvé

    Utilise l'objet AnnotationParams pour accéder aux listes et au cache.
    """

    # Extract resources from params object
    positive_words = params.positive_words
    negative_words = params.negative_words
    memo = params.memo

    # Optional preprocessing
    if params.lowercase:
        tweet = tweet.lower()

    tweet_words = tweet.split()
    pos_count = 0
    neg_count = 0

    for word in tweet_words:
        # ✅ Use memoization
        if word in memo:
            sentiment = memo[word]
        else:
            if word in positive_words:
                sentiment = "positive"
            elif word in negative_words:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            memo[word] = sentiment  # store in cache

        if sentiment == "positive":
            pos_count += 1
        elif sentiment == "negative":
            neg_count += 1

    # Compute final polarity
    if pos_count > neg_count:
        return 4
    elif neg_count > pos_count:
        return 0
    else:
        return 2
