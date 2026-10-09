
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

from aegis.qgate.detector import QGate
from aegis.qgate.baseline_rbf import RBFBaseline


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "hf_prompt_injection"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

QGATE_THRESHOLD = 0.2573042071357381
RBF_THRESHOLD = 0.6060574076713694


def sigmoid(values):
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)

    positive = values >= 0
    result[positive] = 1 / (1 + np.exp(-values[positive]))

    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1 + exp_values)

    return result


def evaluate(y_true, scores, threshold):
    predictions = (scores >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
    }, predictions


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    test = pd.read_csv(DATA / "test.csv")
    texts = test["text"].astype(str).tolist()
    y_test = test["label"].astype(int).to_numpy()

    print("Loading saved models...")
    qgate = QGate.load(MODELS / "qgate_quantum_hf.joblib")
    baseline = RBFBaseline.load(MODELS / "qgate_rbf_hf.joblib")
    embedder = joblib.load(MODELS / "qgate_rbf_hf_8d_embedder.joblib")

    print("Scoring Q-Gate...")
    qgate_scores = qgate.score(texts)

    print("Scoring classical RBF baseline...")
    X_test = embedder.transform(texts)
    rbf_scores = sigmoid(baseline.score(X_test))

    qgate_metrics, qgate_predictions = evaluate(
        y_test, qgate_scores, QGATE_THRESHOLD
    )
    rbf_metrics, rbf_predictions = evaluate(
        y_test, rbf_scores, RBF_THRESHOLD
    )

    report = {
        "dataset": "neuralchemy/Prompt-injection-dataset",
        "config": "core",
        "test_samples": len(y_test),
        "threshold_selection": "Previously selected on validation split",
        "qgate": qgate_metrics,
        "rbf_baseline": rbf_metrics,
        "caveat": (
            "Scores are not calibrated probabilities. "
            "The models use different embedding pipelines."
        ),
    }

    report_path = RESULTS / "final_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    predictions_path = RESULTS / "test_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "text", "actual_label",
            "qgate_score", "qgate_prediction",
            "rbf_score", "rbf_prediction",
        ])

        for i, text in enumerate(texts):
            writer.writerow([
                text,
                int(y_test[i]),
                float(qgate_scores[i]),
                int(qgate_predictions[i]),
                float(rbf_scores[i]),
                int(rbf_predictions[i]),
            ])

    print("\nFINAL TEST RESULTS")
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {report_path}")
    print(f"Saved: {predictions_path}")


if __name__ == "__main__":
    main()
