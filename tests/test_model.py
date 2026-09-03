from src.model import train_and_evaluate


def test_classifier_reaches_strong_baseline() -> None:
    _, metrics, _, _ = train_and_evaluate()
    assert metrics["accuracy"] > 0.95
    assert metrics["f1_macro"] > 0.95
