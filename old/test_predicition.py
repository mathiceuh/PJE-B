# final_prediction_minimal.py

import csv
import random
from core.algorithm_manager import manager

# --- Configuration ---
TEST_CSV_FILE = "../test.csv"
SAMPLE_SIZE = 10
RANDOM_SEED = 42


# ==============================
# Load Data and Sample Tweets
# ==============================
def load_data_and_sample(file_path, sample_size):
    full_dataset = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    try:
                        full_dataset.append((int(row[0]), row[1].strip()))
                    except ValueError:
                        continue  # skip headers or malformed rows
    except FileNotFoundError:
        print(f"ERROR: {file_path} not found.")
        return [], []

    if len(full_dataset) < sample_size:
        print(f"WARNING: Dataset smaller than sample size. Using all data.")
        sample_tweets = full_dataset
    else:
        random.seed(RANDOM_SEED)
        sample_tweets = random.sample(full_dataset, sample_size)

    return full_dataset, sample_tweets


# ==============================
# Main Script
# ==============================
def main():
    print("\n🔹 Sample Prediction Table\n")

    full_dataset, sample_tweets = load_data_and_sample(TEST_CSV_FILE, SAMPLE_SIZE)
    if not full_dataset:
        return

    sample_texts = [text for _, text in sample_tweets]
    algo_names = manager.get_available_algos()
    prediction_results = {}

    # 1️⃣ Fit all algorithms on the full dataset and predict sample
    for algo_name in algo_names:
        manager.select(algo_name)
        algo = manager.get_current()
        try:
            algo.fit(full_dataset)
            predictions = algo.predict_batch(sample_texts)
        except Exception as e:
            predictions = ["FAIL"] * len(sample_texts)
        prediction_results[algo_name] = predictions

    # 2️⃣ Print Table
    header = ["Tweet", "True Label"] + algo_names
    print(f"{header[0]:<75} | {header[1]:<10} | " + " | ".join([f"{name:<12}" for name in algo_names]))
    print("-" * (75 + 3 + 10 + 3 + len(algo_names) * 15))

    for i, (label, tweet) in enumerate(sample_tweets):
        row = f"{tweet[:75]:<75} | {label:<10} | "
        row += " | ".join([f"{prediction_results[name][i]:<12}" for name in algo_names])
        print(row)

    print("\n✅ Done.\n")


if __name__ == "__main__":
    main()


