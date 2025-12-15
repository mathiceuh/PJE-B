# full_dataset_evaluation.py

import csv
from core.algorithm_manager import manager
from core.evaluation import EvaluationService
from tabulate import tabulate

# ==============================
# Configuration
# ==============================
TEST_CSV_FILE = "../test.csv"

# ==============================
# Load Full Dataset
# ==============================
def load_full_dataset():
    dataset = []
    try:
        with open(TEST_CSV_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    try:
                        dataset.append((int(row[0]), row[1].strip()))
                    except ValueError:
                        continue
    except FileNotFoundError:
        print(f"ERROR: {TEST_CSV_FILE} not found.")
    return dataset


# ==============================
# Evaluate All Algorithms
# ==============================
def evaluate_all_algorithms(dataset):
    evaluator = EvaluationService()
    results = {}

    for algo_name in manager.get_available_algos():
        manager.select(algo_name)
        algo = manager.get_current()
        print(f"\n🔹 Evaluating {algo_name}...")
        metrics, summary = evaluator.evaluate_model_performance(algo, dataset)
        results[algo_name] = metrics
    return results


# ==============================
# Pretty Print Metrics
# ==============================
def print_metrics(results):
    for algo_name, metrics in results.items():
        print(f"\n=== {algo_name} Metrics ===")
        if 'error' in metrics:
            print("❌", metrics['error'])
            continue

        # Overall accuracy
        print(f"Accuracy: {metrics['accuracy']}%")

        # Per-class precision/recall/F1
        labels = metrics['labels_order']
        table = []
        for label in labels:
            table.append([
                label,
                metrics['precision'][label],
                metrics['recall'][label],
                metrics['f1_score'][label]
            ])
        print("\nPer-class metrics:")
        print(tabulate(table, headers=["Label", "Precision (%)", "Recall (%)", "F1-score (%)"], tablefmt="grid"))

        # Confusion matrix
        print("\nConfusion Matrix:")
        print(tabulate(metrics['confusion_matrix'], headers=labels, showindex=labels, tablefmt="grid"))

        # Operational info
        print("\nOperational Info:")
        for k, v in metrics['operational'].items():
            print(f"  {k}: {v}")


# ==============================
# Main
# ==============================
def main():
    dataset = load_full_dataset()
    if not dataset:
        print("No data loaded. Exiting.")
        return

    results = evaluate_all_algorithms(dataset)
    print_metrics(results)


if __name__ == "__main__":
    main()
