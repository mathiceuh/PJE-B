class KNNClassifier:
    """
    Generic and modular KNN classifier for text (tweets).
    - Designed for datasets in format (label, tweet)
    - Delegates distance computation to a Distance instance.
    - Delegates voting to a Voting instance.
    """

    def __init__(self, k=3, distance=None, voter=None):
        """
        :param k: number of neighbors to consider
        :param distance: instance of a Distance subclass (e.g., JaccardDistance)
        :param voter: instance of a Voting subclass (e.g., MajorityVote)
        """
        if distance is None:
            raise ValueError("A distance delegate must be provided.")
        if voter is None:
            raise ValueError("A voting delegate must be provided.")

        self.k = k
        self.distance = distance
        self.voter = voter
        self.base = []  # training data [(label, tweet)]

    # -------------------------
    # Training
    # -------------------------
    def fit(self, base):
        """
        Store labeled tweets.
        :param base: list of tuples (label, tweet)
        """
        self.base = base

    # -------------------------
    # Prediction (single sample)
    # -------------------------
    def predict_one(self, x):
        """
        Predict the label for one tweet (already cleaned).
        """
        neighbors = []

        for label, tweet in self.base:
            # compute distance using delegate
            d = self.distance.compute(x, tweet)

            # Fill neighbor list up to k, then replace the farthest one if closer found
            if len(neighbors) < self.k:
                neighbors.append((d, label))
            else:
                farthest_idx, farthest_d = max(
                    enumerate(neighbors), key=lambda x: x[1][0]
                )
                if d < farthest_d[0]:
                    neighbors[farthest_idx] = (d, label)

        # Ask voting delegate to decide label
        return self.voter.decide(neighbors)

    # -------------------------
    # Prediction (batch)
    # -------------------------
    def predict_batch(self, tweets):
        """
        Predict labels for multiple tweets.
        :param tweets: list of tweet strings (cleaned)
        :return: list of predicted labels
        """
        return [self.predict_one(t) for t in tweets]


