"""Training and evaluation functions for the digit classifier."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_split_data():
    digits = load_digits()
    return train_test_split(
        digits.data,
        digits.target,
        test_size=0.25,
        random_state=42,
        stratify=digits.target,
    )


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", SVC(kernel="rbf", C=5, probability=True, random_state=42)),
        ]
    )


def train_and_evaluate() -> tuple[Pipeline, dict[str, Any], np.ndarray, np.ndarray]:
    x_train, x_test, y_train, y_test = load_split_data()
    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "test_images": int(len(x_test)),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "f1_macro": round(float(f1_score(y_test, predictions, average="macro")), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    return model, metrics, y_test, predictions
