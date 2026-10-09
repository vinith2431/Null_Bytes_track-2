
import csv
import json
from pathlib import Path

import numpy as np

from aegis.qgate.baseline_rbf import RBFBaseline
from aegis.qgate.detector import QGate
from aegis.qgate.embed import TextEmbedder
from aegis.qgate.kernel import gram


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "qgate_dataset.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


def load_dataset():
    with DATA_FILE.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    required = {"text", "label", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Dataset must contain text, label and split columns.")

    for row in rows:
        if row["split"] not in {"train", "dev", "test"}:
            raise ValueError(f"Invalid split: {row['split']}")

    return rows


def get_partition(rows, split):
    selected = [row for row in rows if row["split"] == split]
    texts = [row["text"] for row in selected]
    labels = np.array([int(row["label"]) for row in selected])

    if not selected:
        raise ValueError(f"No examples found for split: {split}")

    if set(labels.tolist()) != {0, 1}:
        raise ValueError(f"Split '{split}' must contain both classes.")

    return texts, labels


def main():
    rows = load_dataset()

    train_texts, y_train = get_partition(rows, "train")
    dev_texts, y_dev = get_partition(rows, "dev")

    # Fit the embedding pipeline only on training texts.
    embedder = TextEmbedder()
    X_train = embedder.fit_transform(train_texts)
    X_dev = embedder.transform(dev_texts)

    # Classical baseline on the shared features.
    baseline = RBFBaseline()
    baseline.fit(X_train, y_train)

    # Quantum-kernel SVM using the same feature representation.
    quantum = QGate()
    quantum.embedder = embedder
    quantum.X_train = X_train
    quantum.model.fit(gram(X_train), y_train)
    quantum.fitted = True

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save the shared embedder and models.
    baseline.save(MODEL_DIR / "qgate_rbf_baseline.joblib")
    quantum.save(MODEL_DIR / "qgate_quantum.joblib")
    import joblib
    joblib.dump(embedder, MODEL_DIR / "qgate_embedder.joblib")

    # Record development-set decision scores for threshold selection.
    results = {
        "dataset_size": len(rows),
        "train_size": len(train_texts),
        "dev_size": len(dev_texts),
        "train_class_counts": {
            "benign": int(np.sum(y_train == 0)),
            "suspicious": int(np.sum(y_train == 1)),
        },
        "dev_class_counts": {
            "benign": int(np.sum(y_dev == 0)),
            "suspicious": int(np.sum(y_dev == 1)),
        },
        "warning": (
            "Starter dataset only. Development scores are not final "
            "held-out test metrics. QGate sigmoid scores are not "
            "calibrated probabilities."
        ),
        "development_scores": {
            "quantum_svm_margins": quantum.decision_function(dev_texts).tolist(),
            "rbf_svm_margins": baseline.score(X_dev).tolist(),
            "labels": y_dev.tolist(),
        },
    }

    with (RESULTS_DIR / "qgate_training.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=2)

    print("Training complete.")
    print(f"Quantum model: {MODEL_DIR / 'qgate_quantum.joblib'}")
    print(f"RBF baseline:  {MODEL_DIR / 'qgate_rbf_baseline.joblib'}")
    print(f"Shared embedder: {MODEL_DIR / 'qgate_embedder.joblib'}")
    print(f"Training record: {RESULTS_DIR / 'qgate_training.json'}")
    print("Reminder: development results are not held-out test metrics.")


if __name__ == "__main__":
    main()