"""Train, evaluate, and save the digit classifier."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from src.model import train_and_evaluate


def main() -> None:
    model, metrics, y_test, predictions = train_and_evaluate()

    artifact_path = Path("artifacts/digit_classifier.joblib")
    metrics_path = Path("reports/metrics.json")
    figure_path = Path("reports/confusion_matrix.png")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, artifact_path)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap="Blues")
    plt.title("Handwritten Digit Classifier")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=160)
    plt.close()

    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {artifact_path}")
    print(f"Saved confusion matrix to {figure_path}")


if __name__ == "__main__":
    main()
