"""Predict one sample from the built-in digits dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.datasets import load_digits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--model", type=Path, default=Path("artifacts/digit_classifier.joblib")
    )
    args = parser.parse_args()

    digits = load_digits()
    if not 0 <= args.sample_index < len(digits.data):
        raise ValueError(f"sample-index must be between 0 and {len(digits.data) - 1}")

    model = joblib.load(args.model)
    sample = digits.data[args.sample_index].reshape(1, -1)
    predicted = int(model.predict(sample)[0])
    confidence = float(model.predict_proba(sample).max())
    actual = int(digits.target[args.sample_index])

    print(f"Predicted digit: {predicted}")
    print(f"Actual digit: {actual}")
    print(f"Confidence: {confidence:.1%}")


if __name__ == "__main__":
    main()
