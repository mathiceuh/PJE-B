import csv
import random
import collections
from voting import MajorityVote
from distance import JaccardDistance
from knn_classifier import KNNClassifier


# -------------------------------------------------
# Helper: Load (label, tweet) dataset from CSV (auto-detect delimiter)
# -------------------------------------------------
def load_dataset(csv_path):
    import csv
    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        sample = f.read(2048)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample)
        reader = csv.reader(f, dialect)

        for row in reader:
            if len(row) < 2:
                continue
            label, tweet = row[0].strip(), row[1].strip()
            if tweet:
                try:
                    data.append((int(label), tweet))
                except ValueError:
                    continue
    return data


# -------------------------------------------------
# Helper: Split into train/test
# -------------------------------------------------
def split_dataset(data, test_ratio=0.3):
    random.seed(42)  # reproducible split
    random.shuffle(data)
    split_idx = int(len(data) * (1 - test_ratio))
    return data[:split_idx], data[split_idx:]


# -------------------------------------------------
# Main test
# -------------------------------------------------
def main():
    dataset_path = "test.csv"
    synonyms_path = "synonyms.json"

    # Load dataset
    data = load_dataset(dataset_path)
    print(f"✅ Loaded {len(data)} tweets")

    if len(data) < 10:
        print("⚠️ WARNING: Very few tweets loaded — check CSV delimiter or path.\n")

    # Split dataset
    train, test = split_dataset(data, test_ratio=0.3)
    print(f"→ Training on {len(train)} | Testing on {len(test)}")

    # Show label balance
    print("\n📊 Label distribution:")
    print("Train:", collections.Counter([y for y, _ in train]))
    print("Test :", collections.Counter([y for y, _ in test]))

    # Detect possible duplicates
    train_texts = set(t for _, t in train)
    leaks = [(y, t) for y, t in test if t in train_texts]
    if leaks:
        print(f"\n⚠️ Found {len(leaks)} possible duplicates crossing train/test splits.\n")

    # Initialize KNN components
    distance = JaccardDistance(use_synonyms=True, synonym_json=synonyms_path)
    voter = MajorityVote()
    knn = KNNClassifier(k=5, distance=distance, voter=voter)
    knn.fit(train)

    # Baseline: always predict majority class
    majority_class = collections.Counter([y for y, _ in train]).most_common(1)[0][0]
    baseline_acc = sum(1 for y, _ in test if y == majority_class) / len(test) * 100

    # Evaluate
    print("\n=== 🧠 Evaluating on Test Split (30%) ===\n")
    correct = sum(1 for label, tweet in test if knn.predict_one(tweet) == label)
    total = len(test)
    accuracy = correct / total * 100

    print(f"✅ KNN Accuracy : {accuracy:.2f}% ({correct}/{total})")
    print(f"🪶 Baseline Acc : {baseline_acc:.2f}% (always predict '{majority_class}')\n")


if __name__ == "__main__":
    main()
