# evaluation.py
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    adjusted_rand_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold
import time
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional


@dataclass
class MetricsSummary:
    mode: str
    score_name: str
    score_value: float
    training_time_sec: float
    params: dict
    operational: dict


class EvaluationService:
    """
    Evaluation service for single algorithms and comparative analysis.
    Supports supervised, rule-based, and unsupervised algorithm modes.
    """

    def __init__(self, test_size: float = 0.2, random_seed: Optional[int] = None):
        self.test_size = test_size
        self.random_seed = random_seed

    # -------------------------------------------------------------
    # Single Algorithm Evaluation
    # -------------------------------------------------------------
    def evaluate_model_performance(self, algo_instance, test_data: List[Tuple]):
        """
        Evaluates a single algorithm on a labeled test dataset.
        Returns detailed metrics and operational info.
        """
        if not test_data:
            return {"error": "Empty dataset"}

        texts = [text for _, text in test_data]
        true_labels = [label for label, _ in test_data]

        # Fit time is optional, only if algo supports fit
        fit_start = time.time()
        try:
            if hasattr(algo_instance, 'fit'):
                algo_instance.fit(test_data)
        except Exception as e:
            return {"error": f"Fit failed: {type(e).__name__} - {e}"}
        fit_time = time.time() - fit_start

        # Prediction
        start_time = time.time()
        try:
            predictions = algo_instance.predict_batch(texts)
        except Exception as e:
            return {"error": f"Prediction failed: {type(e).__name__} - {e}"}
        pred_time = time.time() - start_time

        # Compute metrics
        if algo_instance.mode in ["supervised", "rule-based"]:
            metrics = self._compute_supervised_metrics(true_labels, predictions)
            score_name = "Accuracy"
            score_value = metrics['accuracy']
        elif algo_instance.mode == "unsupervised":
            metrics = self._compute_unsupervised_metrics(true_labels, predictions)
            score_name = "ARI"
            score_value = metrics['adjusted_rand_index']
        else:
            return {"error": f"Unknown algorithm mode: {algo_instance.mode}"}

        # Combine operational info
        metrics['operational'] = {
            "training_time_sec": round(fit_time, 4),
            "batch_prediction_time_sec": round(pred_time, 4),
            "avg_prediction_time_ms": round((pred_time / len(test_data)) * 1000, 4),
            "test_set_size": len(test_data)
        }

        return metrics, MetricsSummary(
            mode=algo_instance.mode,
            score_name=score_name,
            score_value=score_value,
            training_time_sec=round(fit_time, 4),
            params=algo_instance.get_params(),
            operational=metrics['operational']
        )

    # -------------------------------------------------------------
    # Comparative Analysis Across Algorithms
    # -------------------------------------------------------------
    def run_comparative_analysis(self, manager_instance, dataset: List[Tuple]):
        """
        Runs evaluation for all algorithms in the manager instance.
        Returns a dict of MetricsSummary per algorithm.
        """
        results = {}
        algo_names = manager_instance.get_available_algos()

        if not dataset or not any(label is not None for label, _ in dataset):
            return {"error": "Dataset must be labeled for comparative analysis."}

        # Stratified train/test split for supervised/rule-based algorithms
        labels = [label for label, _ in dataset]
        train_data, test_data = train_test_split(
            dataset, test_size=self.test_size, stratify=labels if len(set(labels)) > 1 else None,
            random_state=self.random_seed
        )

        for name in algo_names:
            manager_instance.select(name)
            algo = manager_instance.get_current()

            # Select data according to mode
            if algo.mode in ["supervised", "rule-based"]:
                fit_data, eval_data = train_data, test_data
            elif algo.mode == "unsupervised":
                fit_data, eval_data = dataset, dataset
            else:
                results[name] = {"error": f"Unknown mode: {algo.mode}"}
                continue

            try:
                _, summary = self.evaluate_model_performance(algo, eval_data)
                results[name] = asdict(summary)
            except Exception as e:
                results[name] = {"error": f"Evaluation failed: {type(e).__name__} - {e}"}

        return results

    # -------------------------------------------------------------
    # Metric Calculators
    # -------------------------------------------------------------
    def _compute_supervised_metrics(self, true_labels, predictions):
        labels = sorted(list(set(true_labels + predictions)))
        acc = accuracy_score(true_labels, predictions) * 100
        p, r, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, labels=labels, zero_division=0, average=None
        )
        conf = confusion_matrix(true_labels, predictions, labels=labels)
        return {
            "accuracy": round(acc, 2),
            "precision": {labels[i]: round(p[i] * 100, 2) for i in range(len(labels))},
            "recall": {labels[i]: round(r[i] * 100, 2) for i in range(len(labels))},
            "f1_score": {labels[i]: round(f1[i] * 100, 2) for i in range(len(labels))},
            "confusion_matrix": conf.tolist(),
            "labels_order": labels
        }

    def _compute_unsupervised_metrics(self, true_labels, predictions):
        ari = adjusted_rand_score(true_labels, predictions)
        true_classes = sorted(list(set(true_labels)))
        pred_clusters = sorted(list(set(predictions)))
        conf = confusion_matrix(true_labels, predictions)
        return {
            "adjusted_rand_index": round(ari, 4),
            "clustering_confusion_matrix": conf.tolist(),
            "true_classes_order": true_classes,
            "predicted_clusters_order": pred_clusters
        }
