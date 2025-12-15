# voting.py

from collections import Counter
import math
from collections import defaultdict

from collections import Counter
import math


class MajorityVote:
    """
    Simple majority vote (unweighted).
    Works for both categorical and numeric labels.
    """

    def decide(self, neighbors):
        """
        neighbors: list of tuples (distance, label)
        e.g. [(0.2, 'positive'), (0.3, 'negative'), (0.1, 'positive')]
        """
        if not neighbors:
            raise ValueError("Neighbor list is empty — cannot decide.")

        # Extract all labels
        labels = [label for _, label in neighbors]

        # Count label occurrences
        counter = Counter(labels)
        most_common = counter.most_common()

        # Handle possible tie
        top_count = most_common[0][1]
        tied = [label for label, count in most_common if count == top_count]

        if len(tied) == 1:
            return tied[0]  # clear winner

        # Tie-breaking rule: pick label of closest neighbor among tied ones
        tied_neighbors = [(dist, label) for dist, label in neighbors if label in tied]
        tied_neighbors.sort(key=lambda x: x[0])  # sort by distance ascending
        return tied_neighbors[0][1]  # label of nearest among tied


class WeightedVote:
    """Weighted vote: closer neighbors count more."""
    def decide(self, neighbors):
        """
        neighbors: list of tuples (distance, label)
        weight = 1 / (distance + epsilon)
        """
        if not neighbors:
            raise ValueError("Neighbor list is empty — cannot decide.")

        weights = defaultdict(float)
        for dist, label in neighbors:
            w = 1 / (dist + 1e-6)  # avoid division by zero
            weights[label] += w

        # Sort by total weight
        sorted_labels = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top_weight = sorted_labels[0][1]
        tied = [label for label, w in sorted_labels if abs(w - top_weight) < 1e-9]

        # If no tie → return the winner
        if len(tied) == 1:
            return tied[0]

        # Tie-breaker → pick label with smallest distance among tied labels
        tied_neighbors = [(dist, label) for dist, label in neighbors if label in tied]
        tied_neighbors.sort(key=lambda x: x[0])  # smaller distance wins
        return tied_neighbors[0][1]
