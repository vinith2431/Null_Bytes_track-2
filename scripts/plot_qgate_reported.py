"""[M1] Redraw the bar chart and confusion matrices for the largest reported Q-Gate run
(results/qgate_hf_training_<N>.json), next to the classical RBF baseline scored on the same test split with
M3's validation threshold. Uses M3's own plotting functions; the ROC curve is left unchanged.

Run: python -m scripts.plot_qgate_reported"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from aegis.qgate.baseline_rbf import RBFBaseline
from scripts.evaluate_qgate_final import RBF_THRESHOLD, evaluate, sigmoid
from scripts.plot_qgate_final import plot_benchmark_metrics, plot_confusion_matrices

RESULTS, MODELS, DATA = Path("results"), Path("models"), Path("data/hf_prompt_injection")


def main():
    runs = sorted(RESULTS.glob("qgate_hf_training_*.json"), key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
    path = runs[-1]
    run = json.loads(path.read_text(encoding="utf-8"))
    q = run["qgate"]
    if "rbf_baseline" not in run:
        test = pd.read_csv(DATA / "test.csv")
        X = joblib.load(MODELS / "qgate_rbf_hf_8d_embedder.joblib").transform(test["text"].astype(str).tolist())
        scores = sigmoid(RBFBaseline.load(MODELS / "qgate_rbf_hf.joblib").score(X))
        run["rbf_baseline"], _ = evaluate(test["label"].astype(int).to_numpy(), scores, RBF_THRESHOLD)
        path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    report = {"qgate": {"accuracy": q["accuracy"], "precision": q["precision"], "recall": q["recall"],
                        "f1": q["test_f1"], "confusion_matrix": q["confusion_matrix"]},
              "rbf_baseline": run["rbf_baseline"]}
    plot_benchmark_metrics(report)
    plot_confusion_matrices(report)
    print(f"redrew benchmark_metrics.png and confusion_matrices.png from {path.name}")
    print("RBF baseline:", {k: round(v, 4) for k, v in run["rbf_baseline"].items() if isinstance(v, float)},
          run["rbf_baseline"]["confusion_matrix"])


if __name__ == "__main__":
    main()
