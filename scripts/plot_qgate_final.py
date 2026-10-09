
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

REPORT_FILE = RESULTS / "final_evaluation.json"
PREDICTIONS_FILE = RESULTS / "test_predictions.csv"


def plot_benchmark_metrics(report):
    models = {
        "Q-Gate": report["qgate"],
        "RBF Baseline": report["rbf_baseline"],
    }

    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (name, values) in enumerate(models.items()):
        scores = [values[m] for m in metrics]
        bars = ax.bar(
            x + (i - 0.5) * width,
            scores,
            width,
            label=name,
        )
        ax.bar_label(bars, fmt="%.3f", padding=3)

    ax.set_title("Q-Gate vs Classical RBF Baseline")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    fig.savefig(RESULTS / "benchmark_metrics.png", dpi=200)
    plt.close(fig)


def plot_confusion_matrices(report):
    names = ["Q-Gate", "RBF Baseline"]
    keys = ["qgate", "rbf_baseline"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, name, key in zip(axes, names, keys):
        cm = np.array(report[key]["confusion_matrix"])

        image = ax.imshow(cm, interpolation="nearest")
        ax.set_title(name)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Benign", "Suspicious"])
        ax.set_yticklabels(["Benign", "Suspicious"])

        for row in range(2):
            for col in range(2):
                ax.text(
                    col, row, str(cm[row, col]),
                    ha="center", va="center",
                )

        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Test-Set Confusion Matrices")
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrices.png", dpi=200)
    plt.close(fig)


def plot_roc_curves(predictions):
    y_true = predictions["actual_label"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 6))

    curves = [
        ("Q-Gate", "qgate_score"),
        ("RBF Baseline", "rbf_score"),
    ]

    for name, score_column in curves:
        scores = predictions[score_column].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)

        ax.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {roc_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    ax.set_title("ROC Curves on Test Set")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    fig.savefig(RESULTS / "roc_curves.png", dpi=200)
    plt.close(fig)


def main():
    if not REPORT_FILE.exists() or not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            "Run scripts/evaluate_qgate_final.py first."
        )

    report = json.loads(REPORT_FILE.read_text(encoding="utf-8"))
    predictions = pd.read_csv(PREDICTIONS_FILE)

    plot_benchmark_metrics(report)
    plot_confusion_matrices(report)
    plot_roc_curves(predictions)

    print("Generated graphs:")
    print(RESULTS / "benchmark_metrics.png")
    print(RESULTS / "confusion_matrices.png")
    print(RESULTS / "roc_curves.png")


if __name__ == "__main__":
    main()
