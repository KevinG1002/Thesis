import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
from .metrics import *


class Plot(object):
    def __init__(self, experiment_cfg: dict):
        """
        Init method from https://stackoverflow.com/questions/1305532/how-to-convert-a-nested-python-dict-to-object
        """
        for k, v in experiment_cfg.items():
            if isinstance(v, (list, np.ndarray, str)):
                setattr(self, k, v)
            elif isinstance(v, dict):
                Plot(v)

        assert (
            self.model_name
        ), "Can't plot because need the model name to title our plot."
        assert (
            self.experiment_name
        ), "Can't plot because need the experiment name to title our plot and keep track of experiments"

    def plot_roc_curve(self):
        plt.clf()
        assert self.thresholds, "Can't plot ROC curve without thresholds"
        assert self.roc_scores, "Can't plot ROC curve without ROC scores."

        plt.plot(self.thresholds, self.roc_scores)
        plt.xlabel("Probability thresholds")
        plt.ylabel("ROC Scores")
        plt.yticks(np.arange(0, 1, 0.1))
        plt.xticks(np.arange(0, 1, 0.1))
        plt.title(f"{self.model_name} on {self.dataset_name} - ROC Curves")
        plt.tight_layout()
        plt.savefig(
            os.path.dirname(f"{self.dir_path}")
            + f"/{self.experiment_name}_ROC_curves.png",
            dpi=400,
        )
        plt.close()

    def plot_confusion_matrix(self):
        plt.clf()
        assert (
            self.confusion_matrix
        ), "Can't plot confusion matrix without confusion matrix array"
        plt.imshow(self.confusion_matrix, cmap="hot", interpolation="nearest")
        colorbar = plt.colorbar()
        colorbar.set_label("Counts")
        plt.title(f"{self.model_name} on {self.dataset_name} - Confusion Matrix")
        plt.tight_layout()
        plt.savefig(
            os.path.dirname(f"{self.dir_path}")
            + f"/{self.experiment_name}_Confusion_Matrix.png",
            dpi=400,
        )
        plt.close()

    def val_loss_plot(self):
        plt.clf()
        assert self.val_loss, "Need val_loss data to plot loss during training."
        plt.plot(self.val_loss)
        plt.xlabel("Epochs")
        plt.ylabel("Average Loss")
        plt.title(f"{self.model_name} on {self.dataset_name} - Validation Loss Plot")
        plt.savefig(
            os.path.dirname(f"{self.dir_path}")
            + f"/{self.experiment_name}_val_loss_plot.png",
            dpi=400,
        )
        plt.close()

    def val_log_odds_plot(self):
        plt.clf()
        assert (
            self.val_log_odds
        ), "Need val_log_odds data to plot log odds during training"
        plt.plot(self.val_log_odds)
        plt.xlabel("Epochs")
        plt.ylabel("Average Log-Odds Ratio")
        plt.title(
            f"{self.model_name} on {self.dataset_name} - Validation Log-Odd Ratio Plot"
        )
        plt.savefig(
            os.path.dirname(f"{self.dir_path}")
            + f"/{self.experiment_name}_val_log_odds_plot.png",
            dpi=400,
        )
        plt.close()

    def val_acc_plot(self):
        plt.clf()
        assert self.val_acc, "Need val_acc data to plot loss during training."
        plt.plot(self.val_acc)
        plt.xlabel("Epochs")
        plt.ylabel("Validation Accuracy")
        plt.title(
            f"{self.model_name} on {self.dataset_name} - Validation Accuracy Plot"
        )
        plt.savefig(
            os.path.dirname(f"{self.dir_path}")
            + f"/{self.experiment_name}_val_acc_plot.png",
            dpi=400,
        )
        plt.close()

    def weight_histogram_plot(self, trained_model: nn.Module):
        max_num_plots_in_col = 3
        col_idx = 0
        plt.clf()
        assert isinstance(
            trained_model, nn.Module
        ), "Need a Module object to compute weight histogram."
        fig = plt.figure()
        fig, ax = plt.subplots()
        for idx, layer, params in enumerate(trained_model.named_parameters()):
            if idx % 3 == 0 and idx > 0:
                col_idx += 1
                ax[idx % 3, col_idx].histogram()


def plot_roc_curve(experiment_cfg: dict):
    plt.clf()
    plt.plot(thresholds, roc_scores)
    plt.xlabel("Probability thresholds")
    plt.ylabel("ROC Scores")
    plt.yticks(np.arange(0, 1, 0.1))
    plt.xticks(np.arange(0, 1, 0.1))
    plt.title(
        f"{experiment_cfg.model_name} on {experiment_cfg.dataset_name} - ROC Curves"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.dirname(f"{experiment_cfg.dir_path}")
        + f"/{experiment_cfg.experiment_name}_ROC_curves.png",
        dpi=400,
    )


def plot_confusion_matrix(experiment):
    pass


def val_loss_plot():
    pass


def log_odds_ratio():
    pass


def val_accuracy_plot():
    pass


def weight_histogram_plot():
    pass
