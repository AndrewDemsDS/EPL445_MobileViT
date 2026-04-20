"""Tests for evaluation metrics."""

import numpy as np

from src.evaluation.metrics import compute_all_metrics


def test_perfect_classification():
    """All metrics should be 1.0 for perfect predictions."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    class_names = ["car", "bus", "truck"]

    metrics = compute_all_metrics(y_true, y_pred, class_names=class_names)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["macro_precision"] == 1.0
    assert metrics["macro_recall"] == 1.0

    for cls in class_names:
        assert metrics["per_class"][cls]["f1"] == 1.0
        assert metrics["per_class"][cls]["recall_sensitivity"] == 1.0
        assert metrics["per_class"][cls]["specificity"] == 1.0


def test_random_classification_low_metrics():
    """Metrics should be low for random predictions."""
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 3, size=100)
    y_pred = rng.randint(0, 3, size=100)
    class_names = ["a", "b", "c"]

    metrics = compute_all_metrics(y_true, y_pred, class_names=class_names)

    # With 3 classes and random preds, accuracy should be near 1/3
    assert 0.1 < metrics["accuracy"] < 0.6


def test_roc_auc_with_probabilities():
    """ROC data and AUC should be computed when probabilities are provided."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    # Simulate confident softmax
    y_prob = np.array([
        [0.9, 0.05, 0.05],
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.05, 0.9, 0.05],
        [0.1, 0.1, 0.8],
        [0.05, 0.05, 0.9],
    ])
    class_names = ["car", "bus", "truck"]

    metrics = compute_all_metrics(y_true, y_pred, y_prob, class_names)

    assert "roc" in metrics
    assert "auc_macro" in metrics
    assert metrics["auc_macro"] >= 0.9  # should be very high for perfect preds


def test_confusion_matrix_shape():
    """Confusion matrix should be num_classes × num_classes."""
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3])
    class_names = ["car", "bus", "truck", "background"]

    metrics = compute_all_metrics(y_true, y_pred, class_names=class_names)
    cm = np.array(metrics["confusion_matrix"])

    assert cm.shape == (4, 4)
