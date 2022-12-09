import torch
from torchmetrics.classification import (
    MulticlassCalibrationError,
    ConfusionMatrix,
    MulticlassF1Score,
    MulticlassAUROC,
    MulticlassROC,
)
import numpy as np


def per_class_accuracy(
    predictions: torch.Tensor, ground_truth: torch.Tensor
) -> torch.Tensor:
    pass


def ece_score(
    num_classes: int,
    n_bins: int,
    prediction_probabilities: torch.Tensor,
    ground_truth: torch.Tensor,
) -> float:
    multiclass_calib_metric = MulticlassCalibrationError(num_classes, n_bins, norm="l1")
    return multiclass_calib_metric(prediction_probabilities, ground_truth).numpy()


def accuracy(class_predictions: torch.Tensor, ground_truth: torch.Tensor) -> float:
    assert len(class_predictions) == len(ground_truth)
    assert class_predictions[0].size() == ground_truth[0].size()
    correct_preds = torch.where(class_predictions == ground_truth, 1, 0).sum()
    return correct_preds / len(class_predictions)


def confusion_matrix(
    num_classes: int,
    prediction_probabilities: torch.Tensor,
    ground_truth: torch,
    normalized: bool,
) -> np.ndarray:
    if normalized:
        confusion_m = ConfusionMatrix(
            task="multiclass", threshold=0.5, num_classes=num_classes, normalize="true"
        )
    else:
        confusion_m = ConfusionMatrix(
            task="multiclass", threshold=0.5, num_classes=num_classes
        )

    return confusion_m(prediction_probabilities, ground_truth).numpy()


def f1_score(
    num_classes: int,
    prediction_probabilities: torch.Tensor,
    ground_truth: torch.Tensor,
) -> np.ndarray:
    f1_metric = MulticlassF1Score(num_classes, average=None)
    return f1_metric(prediction_probabilities, ground_truth).numpy()


def auroc(
    num_classes: int,
    num_thresholds: int,
    prediction_probabilities: torch.Tensor,
    ground_truth: torch.Tensor,
) -> np.ndarray:
    auroc_metric = MulticlassAUROC(
        num_classes, average="macro", threshold=num_thresholds
    )
    return auroc_metric(prediction_probabilities, ground_truth).numpy()


def roc_score(
    num_classes: int,
    num_thresholds: int,
    prediction_probabilities: torch.Tensor,
    ground_truth: torch.Tensor,
) -> tuple[np.ndarray]:
    roc_metric = MulticlassROC(num_classes, num_thresholds)
    fpr, tpr, thresholds = roc_metric(prediction_probabilities, ground_truth)
    return fpr.numpy(), tpr.numpy(), thresholds.numpy()
