import matplotlib.figure
import numpy as np
import pytest
import seaborn as sns

from openadmet.models.eval.binary import PosthocBinaryMetrics, PosthocBinaryPlots
from openadmet.models.eval.classification import (
    ClassificationMetrics,
    ClassificationPlots,
)
from openadmet.models.eval.eval_base import get_eval_class
from openadmet.models.eval.regression import RegressionMetrics, RegressionPlots


def test_get_eval_class():
    """Verify that evaluation classes can be retrieved by name from the registry."""
    get_eval_class("RegressionMetrics")
    get_eval_class("PosthocBinaryMetrics")
    get_eval_class("ClassificationMetrics")


def test_regression_metrics():
    """
    Validate calculation of standard regression metrics (MSE, MAE, R2).
    
    This test uses simple synthetic data to ensure that the mathematical implementations
    of these metrics are correct and return the expected values.
    """
    y_true = np.array([3, -0.5, 2, 7]).reshape(-1, 1)
    y_pred = np.array([2.5, 0.0, 2, 8]).reshape(-1, 1)

    rm = RegressionMetrics()
    metrics = rm.evaluate(y_true, y_pred)

    assert metrics["task_0"]["mse"]["value"] == pytest.approx(0.375, abs=0.001)
    assert metrics["task_0"]["mae"]["value"] == pytest.approx(0.5, abs=0.001)
    assert metrics["task_0"]["r2"]["value"] == pytest.approx(0.94860, abs=0.001)


def test_regression_plots():
    """
    Verify that regression plotting functions return valid figure objects.
    
    This ensures that regression plots (JointGrid for parity, Figure for CI) are generated
    without error, which is important for model reporting.
    """
    y_true = np.array([3, -0.5, 2, 7]).reshape(-1, 1)
    y_pred = np.array([2.5, 0.0, 2, 8]).reshape(-1, 1)

    rm = RegressionPlots()
    plot_data = rm.evaluate(y_true, y_pred)

    assert isinstance(plot_data, dict)
    assert "task_0_regplot" in plot_data
    assert "task_0_ciplot" in plot_data
    assert isinstance(plot_data["task_0_regplot"], sns.axisgrid.JointGrid)
    assert isinstance(plot_data["task_0_ciplot"], matplotlib.figure.Figure)


def test_classification_metrics():
    """
    Validate calculation of classification metrics (Accuracy, Precision, Recall, F1, AUC).
    
    This ensures that for binary classification tasks, the metrics are computed correctly based on
    predicted probabilities and ground truth labels.
    """
    y_true = [0, 1, 1, 0]

    # We pass probabilities of the class, not the class itself
    # Classes would be [0, 1, 1, 1]
    y_pred = [[1, 0], [0, 1], [0, 1], [0, 1]]

    cm = ClassificationMetrics()
    metrics = cm.evaluate(y_true, y_pred)

    assert metrics["accuracy"]["value"] == pytest.approx(0.75)
    assert metrics["precision"]["value"] == pytest.approx(0.667, abs=0.001)
    assert metrics["recall"]["value"] == pytest.approx(1.0)
    assert metrics["f1"]["value"] == pytest.approx(0.8)
    assert metrics["roc_auc"]["value"] == pytest.approx(0.75)
    assert metrics["pr_auc"]["value"] == pytest.approx(0.833, abs=0.001)


def test_classification_plots():
    """
    Verify that classification plotting functions (ROC, PR curves) return valid figure objects.
    """
    y_true = [0, 1, 1, 0]
    y_pred = [[1, 0], [0, 1], [0, 1], [0, 1]]

    cp = ClassificationPlots()
    cp.evaluate(y_true, y_pred)

    assert isinstance(cp.plot_data, dict)
    assert "roc_curve" in cp.plot_data
    assert "pr_curve" in cp.plot_data
    assert isinstance(cp.plot_data["roc_curve"], matplotlib.figure.Figure)
    assert isinstance(cp.plot_data["pr_curve"], matplotlib.figure.Figure)


def test_posthoc_eval_metrics():
    """
    Test post-hoc binary metrics utility functions.
    
    Verifies that we can calculate precision and recall at a specific cutoff threshold from
    regression-like outputs (or probabilities).
    """
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    cutoff = 4.0
    pem = PosthocBinaryMetrics()
    precision, recall = pem.get_precision_recall(y_pred, y_true, cutoff)
    assert precision == 1.0
    assert recall == 1.0


def test_posthoc_binary_metrics_evaluate():
    """
    Test the full evaluate method of PosthocBinaryMetrics.

    Verifies that it returns the expected dictionary structure with precision and recall
    for each task at the given cutoff.
    """
    y_true = np.array([3, -0.5, 2, 7]).reshape(-1, 1)
    y_pred = np.array([2.5, 0.0, 2, 8]).reshape(-1, 1)
    cutoff = 4.0

    pbm = PosthocBinaryMetrics()
    metrics = pbm.evaluate(y_true, y_pred, cutoff=cutoff)

    # Check structure
    assert "cutoff" in metrics
    assert metrics["cutoff"] == cutoff
    assert "task_0" in metrics
    assert "precision" in metrics["task_0"]
    assert "recall" in metrics["task_0"]

    # Check values
    # For cutoff 4.0:
    # y_true > 4.0: [F, F, F, T] -> [0, 0, 0, 1]
    # y_pred > 4.0: [F, F, F, T] -> [0, 0, 0, 1]
    # Precision: 1.0, Recall: 1.0
    assert metrics["task_0"]["precision"]["value"] == pytest.approx(1.0)
    assert metrics["task_0"]["recall"]["value"] == pytest.approx(1.0)

    # Test with a different cutoff where predictions might be wrong
    cutoff_2 = 1.0
    # y_true > 1.0: [T, F, T, T] -> [1, 0, 1, 1] (3 positives)
    # y_pred > 1.0: [T, F, T, T] -> [1, 0, 1, 1]
    metrics_2 = pbm.evaluate(y_true, y_pred, cutoff=cutoff_2)
    assert metrics_2["task_0"]["precision"]["value"] == pytest.approx(1.0)
    assert metrics_2["task_0"]["recall"]["value"] == pytest.approx(1.0)


def test_posthoc_binary_plots_evaluate():
    """
    Test the evaluate method of PosthocBinaryPlots.

    Verifies that it returns a dictionary of matplotlib figures for confusion matrix
    and classification scatter plots.
    """
    y_true = np.array([3, -0.5, 2, 7]).reshape(-1, 1)
    y_pred = np.array([2.5, 0.0, 2, 8]).reshape(-1, 1)
    cutoff = 4.0

    pbp = PosthocBinaryPlots()
    # Use non-interactive backend to avoid opening windows during test
    import matplotlib

    matplotlib.use("Agg")

    plots = pbp.evaluate(y_true, y_pred, cutoff=cutoff)

    assert isinstance(plots, dict)
    assert "task_0_confusion_matrix" in plots
    assert "task_0_classification_scatter" in plots
    assert isinstance(plots["task_0_confusion_matrix"], matplotlib.figure.Figure)
    assert isinstance(plots["task_0_classification_scatter"], matplotlib.figure.Figure)
