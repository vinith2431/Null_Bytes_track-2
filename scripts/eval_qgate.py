
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

from aegis.qgate.baseline_rbf import RBFBaseline
from aegis.qgate.detector import QGate


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "qgate_dataset.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def load_test_data():
    with DATA_FILE.open("r", encoding="utf-8", newline="") as f:
        rows = [
            row for row in csv.DictReader(f)
            if row["split"] == "test"
        ]

    if not rows:
        raise ValueError("No test examples found.")

    texts = [row["text"] for row in rows]
    labels = np.array([int(row["label"]) for row in rows])

    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Test data must contain both classes.")

    return texts, labels


def calculate_metrics(y_true, scores, threshold):
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def main():
    texts, y_test = load_test_data()

    quantum = QGate.load(MODEL_DIR / "qgate_quantum.joblib")
    baseline = RBFBaseline.load(
        MODEL_DIR / "qgate_rbf_baseline.joblib"
    )
    embedder = joblib.load(MODEL_DIR / "qgate_embedder.joblib")

    # QGate scores are sigmoid-transformed margins, not calibrated
    # attack probabilities.
    quantum_scores = quantum.score(texts)

    # The RBF baseline returns raw SVM margins. For a consistent
    # [0, 1] threshold scale, apply the same sigmoid transformation.
    rbf_margins = np.asarray(baseline.score(embedder.transform(texts)))
    rbf_scores = np.empty_like(rbf_margins, dtype=float)
    positive = rbf_margins >= 0
    rbf_scores[positive] = 1 / (1 + np.exp(-rbf_margins[positive]))
    exp_values = np.exp(rbf_margins[~positive])
    rbf_scores[~positive] = exp_values / (1 + exp_values)

    results = {
        "test_size": len(y_test),
        "positive_class": "suspicious/injection",
        "warning": (
            "Only seven held-out examples are available. Metrics are "
            "highly uncertain and are not evidence of production readiness. "
            "Scores are transformed decision margins, not calibrated "
            "probabilities."
        ),
        "quantum_svm": {
            "auroc": float(roc_auc_score(y_test, quantum_scores)),
            "threshold_results": [
                calculate_metrics(y_test, quantum_scores, t)
                for t in THRESHOLDS
            ],
        },
        "rbf_svm": {
            "auroc": float(roc_auc_score(y_test, rbf_scores)),
            "threshold_results": [
                calculate_metrics(y_test, rbf_scores, t)
                for t in THRESHOLDS
            ],
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / "qgate_evaluation.json"
    output_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"\nSaved evaluation to: {output_file}")


if __name__ == "__main__":
    main()