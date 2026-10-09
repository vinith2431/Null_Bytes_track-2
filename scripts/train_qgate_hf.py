
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from aegis.qgate.embed import TextEmbedder
from aegis.qgate.baseline_rbf import RBFBaseline
from aegis.qgate.detector import QGate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "hf_prompt_injection"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

QGATE_TRAIN_SIZE = 256
RANDOM_STATE = 42


def best_threshold(y_true, scores):
    # Select the threshold on validation data only.
    candidates = np.unique(
        np.concatenate([
            np.asarray(scores, dtype=float),
            np.array([0.0]),
        ])
    )

    best = {"threshold": 0.0, "f1": -1.0}

    for threshold in candidates:
        predictions = (scores >= threshold).astype(int)
        f1 = f1_score(y_true, predictions, zero_division=0)

        if f1 > best["f1"]:
            best = {"threshold": float(threshold), "f1": float(f1)}

    return best


def main():
    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    train = pd.read_csv(DATA / "train.csv")
    validation = pd.read_csv(DATA / "validation.csv")

    train_texts = train["text"].astype(str).tolist()
    y_train = train["label"].astype(int).to_numpy()
    val_texts = validation["text"].astype(str).tolist()
    y_val = validation["label"].astype(int).to_numpy()

    # ----- Classical RBF baseline: all training samples -----
    print("\n[1/4] Fitting baseline text embedder...")
    start = time.perf_counter()

    baseline_embedder = TextEmbedder()
    X_train = baseline_embedder.fit_transform(train_texts)
    X_val = baseline_embedder.transform(val_texts)

    baseline = RBFBaseline()
    baseline.fit(X_train, y_train)

    baseline_scores = np.asarray(baseline.score(X_val)).ravel()
    baseline_threshold = best_threshold(y_val, baseline_scores)

    print(f"Baseline training time: {time.perf_counter() - start:.2f}s")
    print(f"Baseline validation F1: {baseline_threshold['f1']:.4f}")

    baseline.save(str(MODELS / "qgate_rbf_hf.joblib"))
    joblib.dump(
        baseline_embedder,
        MODELS / "qgate_rbf_hf_embedder.joblib",
    )

    # ----- Q-Gate: reproducible stratified training subset -----
    print("\n[2/4] Selecting Q-Gate training subset...")
    q_indices, _ = train_test_split(
        np.arange(len(train)),
        train_size=QGATE_TRAIN_SIZE,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    q_texts = [train_texts[i] for i in q_indices]
    y_q = y_train[q_indices]

    print(
        f"Q-Gate subset: {len(q_texts)} samples; "
        f"benign={(y_q == 0).sum()}, suspicious={(y_q == 1).sum()}"
    )

    print("\n[3/4] Training Q-Gate...")
    start = time.perf_counter()

    qgate = QGate()
    qgate.fit(q_texts, y_q)

    print(f"Q-Gate training time: {time.perf_counter() - start:.2f}s")

    print("Scoring validation split...")
    start = time.perf_counter()
    qgate_scores = np.asarray(qgate.decision_function(val_texts)).ravel()
    print(f"Q-Gate validation scoring time: {time.perf_counter() - start:.2f}s")

    qgate_threshold = best_threshold(y_val, qgate_scores)
    print(f"Q-Gate validation F1: {qgate_threshold['f1']:.4f}")

    qgate.save(str(MODELS / "qgate_quantum_hf.joblib"))

    # Save validation-derived thresholds and reproducibility details.
    report = {
        "dataset": "neuralchemy/Prompt-injection-dataset",
        "config": "core",
        "random_state": RANDOM_STATE,
        "training_samples": len(train),
        "qgate_training_samples": len(q_texts),
        "validation_samples": len(validation),
        "threshold_selection": "maximum F1 on validation split",
        "baseline": {
            "training_samples": len(train),
            "threshold": baseline_threshold["threshold"],
            "validation_f1": baseline_threshold["f1"],
        },
        "qgate": {
            "training_samples": len(q_texts),
            "threshold": qgate_threshold["threshold"],
            "validation_f1": qgate_threshold["f1"],
        },
        "note": (
            "Test split not loaded or evaluated by this script. "
            "Thresholds were selected using validation data."
        ),
    }

    output = RESULTS / "qgate_hf_training.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n[4/4] Training complete.")
    print(f"Saved report: {output}")
    print("Test split has not been evaluated.")


if __name__ == "__main__":
    main()